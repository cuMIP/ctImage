import DataProcessing
import SimpleITK as sitk
import ModelConfiguration

### process example CT image

ctImage = sitk.ReadImage('./ExampleCTImage.mha)')
binaryImage = DataProcessing.CreateBoneMask(ctImage)
ctImage = DataProcessing.ResampleAndMaskImage(ctImage, binaryImage)

### model
modelPath = './MiccaiFinalModel.dat'
device = ModelConfiguration.getDevice()
model = ModelConfiguration.adaptModel(modelPath, device)
imageData = ModelConfiguration.adaptData(ctImage, device)

landmarks, boneLabels = ModelConfiguration.runModel(model, ctImage, binaryImage, imageData)
