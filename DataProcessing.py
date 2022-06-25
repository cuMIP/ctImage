import SimpleITK as sitk
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
import torchio as tio
import skimage
import skimage.measure

def FloodFillHull(image):
    points = np.transpose(np.where(image))
    hull = ConvexHull(points)
    deln = Delaunay(points[hull.vertices]) 
    idx = np.stack(np.indices(image.shape), axis = -1)
    out_idx = np.nonzero(deln.find_simplex(idx) + 1)
    out_img = np.zeros(image.shape)
    out_img[out_idx] = 1
    return out_img

def CreateHeadMask(ctImage, hounsfieldThreshold = -200):
    """
    Returns a binary image mask of the head from an input CT image

    """

    headMask = sitk.GetArrayFromImage(ctImage)

    # Getting the head
    headMask = (headMask > hounsfieldThreshold).astype(np.uint8)

    headMask = skimage.measure.label(headMask)
    largestLabel = np.argmax(np.bincount(headMask.flat)[1:])+1
    headMask = (headMask == largestLabel).astype(np.uint8)

    headMask = sitk.GetImageFromArray(headMask)
    headMask.SetOrigin(ctImage.GetOrigin())
    headMask.SetSpacing(ctImage.GetSpacing())
    headMask.SetDirection(ctImage.GetDirection())

    return headMask

def CreateBoneMask(ctImage, headMaskImage=None, minimumThreshold=100, maximumThreshold=200, verbose=False):
    """
    Uses adapting thresholding to create a binary mask of the cranial bones from an input CT image.
    [Dangi et al., Robust head CT image registration pipeline for craniosynostosis skull correction surgery, Healthcare Technology Letters, 2017]

    """

    # If a head mask is not provided
    if headMaskImage is None:

        if verbose:
            print('Creating head mask.')

        headMaskImage = CreateHeadMask(ctImage)


    ctImageArray = sitk.GetArrayFromImage(ctImage)
    headMaskImageArray = sitk.GetArrayViewFromImage(headMaskImage)

    # Appling the mask to the CT image
    ctImageArray[headMaskImageArray == 0] = 0

    # Extracting the bones
    minObjects = np.inf
    optimalThreshold = 0
    for threshold in range(minimumThreshold, maximumThreshold+1, 10):

        if verbose:
            print('Optimizing skull segmentation. Threshold {:03d}.'.format(threshold), end='\r')

        labels = skimage.measure.label(ctImageArray >= threshold)
        nObjects = np.max(labels)

        if nObjects < minObjects:
            minObjects = nObjects
            optimalThreshold = threshold
    if verbose:
        print('The optimal threshold for skull segmentation is {:03d}.'.format(optimalThreshold))
    
    ctImageArray = ctImageArray >= optimalThreshold

    ctImageArray = skimage.measure.label(ctImageArray)
    largestLabel = np.argmax(np.bincount(ctImageArray.flat)[1:])+1
    ctImageArray = (ctImageArray == largestLabel).astype(np.uint)
    
    ctImageArray = sitk.GetImageFromArray(ctImageArray)
    ctImageArray.SetOrigin(ctImage.GetOrigin())
    ctImageArray.SetSpacing(ctImage.GetSpacing())
    ctImageArray.SetDirection(ctImage.GetDirection())

    return ctImageArray

def ResampleAndMaskImage(ctImage, binaryImage, outputImageSize = np.array([96, 96, 96], dtype=np.int16)):
    binary = sitk.GetArrayFromImage(binaryImage)
    convexMask = FloodFillHull(binary)
    convexMaskImage = sitk.GetImageFromArray(convexMask)
    convexMaskImage.CopyInformation(binaryImage)
    convexMaskImage = sitk.Cast(convexMaskImage, sitk.sitkUInt32)
    
    normalize = tio.RescaleIntensity(out_min_max = (0, 1), p = 1)
    filter = sitk.MaskImageFilter()
    ctImage = normalize(ctImage)
    ctImage = filter.Execute(ctImage, convexMaskImage)

    templateImageArray = np.zeros(outputImageSize, dtype=np.float32)
    templateImage = sitk.GetImageFromArray(templateImageArray)

    templateImage.SetOrigin(ctImage.GetOrigin())
    spacing = np.array(ctImage.GetSpacing())*np.array(ctImage.GetSize())/outputImageSize
    templateImage.SetSpacing(spacing)

    transform = sitk.AffineTransform(3)
    transform.SetIdentity()
    resampledCTImage = sitk.Resample(ctImage, templateImage, transform.GetInverse(), sitk.sitkLinear)

    return(resampledCTImage)
