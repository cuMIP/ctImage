import torch
import numpy as np

### example CT array

ctImage = np.load('./ctImage.npy')

### model
modelPath = './MiccaiFinalModel.dat'
model = torch.jit.load(modelPath)

if torch.cuda.is_available():
    device = torch.device('cuda')
    numberOfDevices = torch.cuda.device_count()
else:
    device = torch.device('cpu')
    numberOfDevices = 1

model.to(device=device) # Sending to device
model.eval()

ctImage = np.expand_dims(ctImage,axis=0)
ctImage = np.expand_dims(ctImage,axis=0)
ctImage=torch.tensor(ctImage, device = device, dtype = torch.float32)
heatmap_pred, segmentation_pred, _, _ = model(ctImage)
