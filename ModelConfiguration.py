import torch
import numpy as np
import vtk
import SimpleITK as sitk

def getDevice(device=None):
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    return(device)

def adaptData(ctImage, device):
    """
    Data for final model needs to be in batch x channel x height x width x depth format.
    """
    imageData = sitk.GetArrayFromImage(ctImage)
    if len(imageData.shape)==3:
        imageData = np.expand_dims(imageData,axis=0)
        imageData = np.expand_dims(imageData,axis=0)
    elif len(imageData.shape)==4:
        imageData = np.expand_dims(imageData,axis=0)

    return(torch.tensor(imageData, device = device, dtype = torch.float32))

def landmarkPrediction(heatmap, ctImage):

    landmarks = vtk.vtkPolyData()
    landmarks.SetPoints(vtk.vtkPoints())
    origin = ctImage.GetOrigin()
    spacing = ctImage.GetSpacing()
    for p in range(4):

        coords = np.flip(np.where(heatmap[0].cpu().detach().numpy()[p]==heatmap[0].cpu().detach().numpy()[p].max()))

        coords = origin + spacing * coords.ravel()

        landmarks.GetPoints().InsertNextPoint(coords[0], coords[1], coords[2])

    return(landmarks)

def boneLabeling(segmentation, ctImage, binaryImage):
    bonelabels = segmentation.cpu().detach().numpy()
    bonelabels = np.argmax(bonelabels[0, :, :, : ,:], axis=0).astype(np.int16)

    similarityTransform = sitk.AffineTransform(3)
    similarityTransform.SetIdentity()

    bonelabels = sitk.GetImageFromArray(bonelabels.astype(np.uint16))
    bonelabels.CopyInformation(ctImage)
    bonelabels = sitk.Resample(bonelabels, binaryImage, similarityTransform, sitk.sitkNearestNeighbor)

    labelsArray = sitk.GetArrayFromImage(bonelabels)
    labelsArray[sitk.GetArrayViewFromImage(binaryImage)<= 0] = 0
    bonelabels = sitk.GetImageFromArray(labelsArray)
    bonelabels.CopyInformation(binaryImage)

    return(bonelabels)

def adaptModel(modelPath, device):
    model = torch.jit.load(modelPath)
    model.to(device=device) # Sending to device
    model.eval()   
    return(model)

def runModel(model, ctImage, binaryImage, imageData):
    heatmap_pred, segmentation_pred, _, _ = model(imageData)

    landmarks = landmarkPrediction(heatmap_pred, ctImage)
    boneLabels = boneLabeling(segmentation_pred, ctImage, binaryImage)

    return(landmarks, boneLabels)
