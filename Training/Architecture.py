import os
import numpy as np
import torch
from Training import NumpyDataLoader as NumpyDataLoader
import torch.utils.data as data
from Training import EncodingModule as EncodingModule

from torch.utils.tensorboard import SummaryWriter

def initialize_weights(model):
    # Initializes weights
    for m in model.modules():
        if isinstance(m, (torch.nn.Conv3d, torch.nn.ConvTranspose3d)):
            torch.nn.init.xavier_uniform(m.weight)

class FCN(torch.nn.Module):

    def __init__(self, input=None):

        super().__init__()

        ## Configuration variables
        self.addDistanceAndDirection = True
        self.addEncodingModule = False
        self.addSELoss = True
        self.epochs=300
        self.batchSize=10
        self.learningRate=0.001
        self.randomizedTraining = True
        self.trainingSaveFrequency = 5
        self.validationFrequency = 2
        self.regularization = 1e-8
        self.lambdaHeatmapLoss = 1
        self.lambdaSegmentationLoss = 1
        self.lambdaCorrelationLoss = 1
        self.lambdaBone = 30
        self.lambdaContext = 1
        
        ## Definition of model layers

        self.down_1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1), 
            torch.nn.BatchNorm3d(num_features=32), 
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3, padding=1), 
            torch.nn.BatchNorm3d(num_features=32), 
            torch.nn.ReLU(),
        )
        self.down_1.apply(initialize_weights)

        self.down_2 = torch.nn.Sequential(
            torch.nn.MaxPool3d(kernel_size=2, stride=2), # 48 x 48 x 48
            torch.nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU(),
        ) 
        self.down_2.apply(initialize_weights)

        self.down_3 = torch.nn.Sequential(
            torch.nn.MaxPool3d(kernel_size=2, stride=2), # 24 x 24 x 24
            torch.nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1), 
            torch.nn.BatchNorm3d(num_features=128), 
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1), 
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU(),
        ) 
        self.down_3.apply(initialize_weights)

        ## encoding layers
        self.encmodule = EncodingModule.EncModule(64, 32, se_loss= self.addSELoss)

        ## decoder 
        self.upres_2 = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1),
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU()
        )
        self.upres_2.apply(initialize_weights)

        # Concatenation with down branch

        self.up_2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=128), 
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU(),
        )
        self.up_2.apply(initialize_weights)


        self.upres_1 = torch.nn.Sequential(
            torch.nn.Dropout(),
            torch.nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=2, output_padding=1),
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU()
        )
        self.upres_1.apply(initialize_weights)

        # Concatenation with down branch

        self.up_1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=96, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU(),
        ) # N x 96 x 96 x 96 x (3L)
        self.up_1.apply(initialize_weights)

        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=64, out_channels=6, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=6), 
            torch.nn.Softmax(dim=1),
        ) ## for segmentation maps (8 bone pieces, including background)
        self.conv1.apply(initialize_weights)

        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(num_features=64), 
            torch.nn.ReLU(),
            torch.nn.Conv3d(in_channels=64, out_channels=4, kernel_size=3, padding=1),
        ) ## for landmark heatmaps (4 landmarks)
        self.conv2.apply(initialize_weights)

        self.fullyConnected1 = torch.nn.Sequential(
            torch.nn.Linear(24 * 24 * 24 * 64, 64),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.ReLU(),
            torch.nn.Linear(64,36),
            torch.nn.Tanh()
        ) 

    def forward(self, x):

        down1 = self.down_1(x)
        down2 = self.down_2(down1)
        down3 = self.down_3(down2)
        if (self.addEncodingModule) & (self.addSELoss == False):
            down3 = self.encmodule(down3)

        if (self.addEncodingModule) & (self.addSELoss):
            down3, se_pred = self.encmodule(down3) 

        up2 = self.upres_2(down3)
        up2 = torch.cat((down2, up2), 1)
        up2 = self.up_2(up2)
        up1 = self.upres_1(up2)
        up1 = torch.cat((down1, up1), 1)
        up1 = self.up_1(up1)

        segmentationMap = self.conv1(up1)
        heatmap = self.conv2(up1)

        output = [heatmap, segmentationMap]

        if self.addDistanceAndDirection:
            flatMat = down3.view(-1, 24*24*24*64)
            distance = self.fullyConnected1(flatMat)
            output.append(distance)

        if self.addSELoss:
            output.append(se_pred)

        return(output)
        
    def trainNetwork(self, CTPath, HeatmapPath, modelPath, DisplacementPath = None, SegmentationPath = None, logPath=None, resumeTraining = False, verbose=False):
        
        torch.manual_seed(0)

        TrainingData = NumpyDataLoader.Dataset(CTPath, HeatmapPath, SegmentationPath, DisplacementPath,
            AddCorrelation = self.addDistanceAndDirection, split = "train")
        ValidationData = NumpyDataLoader.Dataset(CTPath, HeatmapPath, SegmentationPath, DisplacementPath,
            AddCorrelation = self.addDistanceAndDirection, split = "validation")

        nTrainingImages = TrainingData.__len__()
        nValidationImages = ValidationData.__len__()

        # Configuring the device
        if torch.cuda.is_available():
            device = torch.device('cuda')
            numberOfDevices = torch.cuda.device_count()
        else:
            device = torch.device('cpu')
            numberOfDevices = 1
        
        model = self
        model.to(device=device)

        if numberOfDevices > 1:
            # Parallelizing between two GPUs
            model = torch.nn.DataParallel(model, 
                device_ids=list(range(numberOfDevices)), output_device="cuda:0")       
        
        if verbose:
            print(f'Using {numberOfDevices} device(s) of type {device}')

        ## Configuring optimization

        optimizer = torch.optim.SGD(model.parameters(), lr=self.learningRate, momentum=0.9, weight_decay=self.regularization)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1)
        loss_heatmaps = lambda input, target: ((input - target) ** 2).mean()

        def loss_segmentation(input, target):
            vals = 0
            for i in range(6):
                if i==0:
                    vals -= ((target[:,i,:,:,:]==1) * (torch.log(input[:,i,:,:,:]))).mean()
                else:
                    vals -= self.lambdaBone * ((target[:,i,:,:,:]==1) * (torch.log(input[:,i,:,:,:]))).mean()
            vals /= 6
            return vals

        def calculateLandmarkError(input, target, point):
            input = input.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            error = 0
            for subject in range(input.shape[0]):
                try:
                    error += np.sqrt(np.sum((np.array(np.where(input[subject,point,:,:,:]==input[subject,point,:,:,:].max())) - np.array(np.where(target[subject,point,:,:,:]==target[subject,point,:,:,:].max())))**2))
                except Exception as ex:
                    pass
            return(error)

        def calculateLabelAccuracy(input, target, bone = None):
            input = input.cpu().detach().numpy()
            target = target.cpu().detach().numpy()
            smooth = 1e-4
            if bone is None:
                dice = 0
                for subject in range(input.shape[0]):
        
                    label_pred = np.argmax(input[subject, :, :, : ,:], axis=0)
                    for index in range(6):
                        bone_pred = label_pred==index
                        bone_true = target[subject, index, :,:,:]==1
                        intersection = np.logical_and(bone_pred, bone_true)
                        dice += (2. * np.count_nonzero(intersection) + smooth) / (np.count_nonzero(bone_pred) + np.count_nonzero(bone_true) + smooth) / 6

            else:
                dice = 0
                for subject in range(input.shape[0]):
                    label_pred = np.argmax(input[subject, :, :, : ,:], axis=0)
                    bone_pred = label_pred==bone
                    bone_true = target[subject, bone, :,:,:]==1
                    intersection = np.logical_and(bone_pred, bone_true)
                    dice += (2. * np.count_nonzero(intersection) + smooth) / (np.count_nonzero(bone_pred) + np.count_nonzero(bone_true) + smooth)

            return dice

        # Creating the loggers
        if logPath is not None:
            writer = SummaryWriter(os.path.join(logPath, 'Train'))
            writerValidate = SummaryWriter(os.path.join(logPath, 'Validation'))

        else:
            writer = None

        ## Optimizing
        TrainingDataLoader = data.DataLoader(dataset=TrainingData, batch_size=self.batchSize, shuffle=True, pin_memory= True)
        ValidationDataLoader = data.DataLoader(dataset=ValidationData, batch_size=self.batchSize, shuffle=True, pin_memory= True) 

        BestValidationLoss= 1e4
        BestLandmarkError = np.zeros(4)
        BestLabelingAccuracy = np.zeros(9)

        if resumeTraining:
            checkpoint = torch.load(os.path.join(modelPath, 'Model_Best.pt'))
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            epoch = checkpoint['epoch']

        start = 0
        if resumeTraining:
            start = epoch

        for epoch in range(start, start + self.epochs):

            model.train()

            epoch_loss = 0
            epoch_segmentationLoss = 0
            epoch_heatmapLoss = 0
            epoch_targetLoss = 0

            if self.addDistanceAndDirection:
                epoch_correlationLoss = 0

            if self.addSELoss:
                epoch_SELoss = 0

            epoch_landmarkAccuracyP1 = 0
            epoch_landmarkAccuracyP2 = 0
            epoch_landmarkAccuracyP3 = 0
            epoch_landmarkAccuracyP4 = 0

            epoch_labelAccuracyAll = 0
            epoch_labelAccuracyB0 = 0
            epoch_labelAccuracyB1 = 0
            epoch_labelAccuracyB2 = 0
            epoch_labelAccuracyB3 = 0
            epoch_labelAccuracyB4 = 0
            epoch_labelAccuracyB5 = 0
            
            # Iterating through all minibatches
            for (batch_idx, batch) in enumerate(TrainingDataLoader):
                # Sending batch data to device
                x = batch[0]
                x=torch.tensor(x, device = device, dtype = torch.float32)
                yHeatmap = batch[1] * 100
                yHeatmap=torch.tensor(yHeatmap, device = device, dtype = torch.float32)
                ySegmentation = batch[2]
                ySegmentation = torch.tensor(ySegmentation, device=device, dtype=torch.int64)

                if (self.addDistanceAndDirection) & (self.addSELoss):
                    xDisplacement = batch[3]
                    xDisplacement = torch.tensor(xDisplacement, device = device, dtype = torch.float32)
                    yDistance = batch[4]
                    yDistance = torch.tensor(yDistance, device = device, dtype = torch.float32)

                elif (self.addDistanceAndDirection == False) & (self.addSELoss):
                    xDisplacement = batch[3]
                    xDisplacement = torch.tensor(xDisplacement, device = device, dtype = torch.float32)

                elif (self.addDistanceAndDirection) & (self.addSELoss == False):
                    yDistance = batch[3]
                    yDistance = torch.tensor(yDistance, device = device, dtype = torch.float32)


                if self.addDistanceAndDirection & self.addSELoss:
                    heatmap_pred, segmentation_pred, distance_pred, se_pred = model(x)
                elif self.addDistanceAndDirection & (self.addSELoss == False):
                    heatmap_pred, segmentation_pred, distance_pred = model(x)
                elif (self.addDistanceAndDirection == False) & (self.addSELoss):
                    heatmap_pred, segmentation_pred, se_pred = model(x)
                else:
                    heatmap_pred, segmentation_pred = model(x)

                ## Quantifying losses
                if self.addDistanceAndDirection & self.addSELoss:
                    heatmapLoss = self.lambdaHeatmapLoss * loss_heatmaps(heatmap_pred, yHeatmap)
                    segmentationLoss = self.lambdaSegmentationLoss * loss_segmentation(segmentation_pred, ySegmentation)
                    distanceLoss = self.lambdaCorrelationLoss * loss_heatmaps(distance_pred, yDistance)
                    seLoss = self.lambdaContext * loss_heatmaps(se_pred, xDisplacement)
                    loss = heatmapLoss + segmentationLoss + distanceLoss + seLoss
                elif self.addDistanceAndDirection & (self.addSELoss == False):
                    heatmapLoss = self.lambdaHeatmapLoss * loss_heatmaps(heatmap_pred, yHeatmap)
                    segmentationLoss = self.lambdaSegmentationLoss * loss_segmentation(segmentation_pred, ySegmentation)
                    distanceLoss = self.lambdaCorrelationLoss * loss_heatmaps(distance_pred, yDistance)
                    loss = heatmapLoss + segmentationLoss + distanceLoss
                elif (self.addDistanceAndDirection == False) & self.addSELoss:
                    heatmapLoss = self.lambdaHeatmapLoss * loss_heatmaps(heatmap_pred, yHeatmap)
                    segmentationLoss = self.lambdaSegmentationLoss * loss_segmentation(segmentation_pred, ySegmentation)
                    seLoss = self.lambdaContext * loss_heatmaps(se_pred, xDisplacement)
                    loss = heatmapLoss + segmentationLoss + seLoss
                else:
                    heatmapLoss = loss_heatmaps(heatmap_pred, yHeatmap)
                    segmentationLoss = self.lambdaSegmentationLoss * loss_segmentation(segmentation_pred, ySegmentation)
                    loss =  heatmapLoss + segmentationLoss
                
                labelsLoss = segmentationLoss.data
                LandmarkLoss = heatmapLoss.data
                targetLoss = labelsLoss + LandmarkLoss

                batchLoss = loss.data
                epoch_loss += batchLoss * x.shape[0] / nTrainingImages

                epoch_segmentationLoss += labelsLoss * x.shape[0] / nTrainingImages
                epoch_heatmapLoss += LandmarkLoss * x.shape[0] / nTrainingImages
                epoch_targetLoss += targetLoss * x.shape[0] / nTrainingImages

                if self.addDistanceAndDirection:
                    epoch_correlationLoss += distanceLoss * x.shape[0] / nTrainingImages
                if self.addSELoss:
                    epoch_SELoss += seLoss * x.shape[0] / nTrainingImages
            
                # Evaluating classification performance for training

                errorP1 = calculateLandmarkError(heatmap_pred, yHeatmap, 0)
                errorP2 = calculateLandmarkError(heatmap_pred, yHeatmap, 1)
                errorP3 = calculateLandmarkError(heatmap_pred, yHeatmap, 2)
                errorP4 = calculateLandmarkError(heatmap_pred, yHeatmap, 3)
                epoch_landmarkAccuracyP1 += errorP1/nTrainingImages
                epoch_landmarkAccuracyP2 += errorP2/nTrainingImages
                epoch_landmarkAccuracyP3 += errorP3/nTrainingImages
                epoch_landmarkAccuracyP4 += errorP4/nTrainingImages

                accuracyAll = calculateLabelAccuracy(segmentation_pred, ySegmentation)
                accuracyB0 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 0)
                accuracyB1 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 1)
                accuracyB2 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 2)
                accuracyB3 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 3)
                accuracyB4 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 4)
                accuracyB5 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 5)
                # accuracyB6 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 6)
                # accuracyB7 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 7)
                epoch_labelAccuracyAll += accuracyAll/nTrainingImages
                epoch_labelAccuracyB0 += accuracyB0/nTrainingImages
                epoch_labelAccuracyB1 += accuracyB1/nTrainingImages
                epoch_labelAccuracyB2 += accuracyB2/nTrainingImages
                epoch_labelAccuracyB3 += accuracyB3/nTrainingImages
                epoch_labelAccuracyB4 += accuracyB4/nTrainingImages
                epoch_labelAccuracyB5 += accuracyB5/nTrainingImages
                # epoch_labelAccuracyB6 += accuracyB6/nTrainingImages
                # epoch_labelAccuracyB7 += accuracyB7/nTrainingImages

                if verbose:
                    if self.addDistanceAndDirection & self.addSELoss:
                        print(' - Batch {:05d}/{:05d}. Loss: {:.5f} = {:.5f} + {:.5f} + {:.5f} + {:.5f}'.format(batch_idx, len(TrainingDataLoader), batchLoss.data, 
                            LandmarkLoss, labelsLoss, distanceLoss.data, seLoss.data), end='\r')
                    elif self.addDistanceAndDirection & (self.addSELoss == False):
                        print(' - Batch {:05d}/{:05d}. Loss: {:.5f} = {:.5f} + {:.5f} + {:.5f}'.format(batch_idx, len(TrainingDataLoader), batchLoss.data, 
                            LandmarkLoss, labelsLoss, distanceLoss.data), end='\r')
                    elif (self.addDistanceAndDirection == False) & self.addSELoss:
                        print(' - Batch {:05d}/{:05d}. Loss: {:.5f} = {:.5f} + {:.5f} + {:.5f}'.format(batch_idx, len(TrainingDataLoader), batchLoss.data, 
                            LandmarkLoss, labelsLoss, seLoss.data), end='\r')
                    else:
                        print(' - Batch {:05d}/{:05d}. Loss: {:.5f}= {:.5f} + {:.5f}'.format(batch_idx, len(TrainingDataLoader), batchLoss.data, LandmarkLoss, labelsLoss), end='\r')
                # Backpropagating
                optimizer.zero_grad()
                loss.backward()

                # Updating values
                torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
                optimizer.step()

            # Checking if the learning rate should be decreased
            scheduler.step(epoch_loss)
            
            if epoch % self.validationFrequency == 0 or epoch == self.epochs-1:
                ## validation      

                epoch_lossTest = 0
                epoch_segmentationLossTest = 0
                epoch_heatmapLossTest = 0
                epoch_targetLossTest = 0

                if self.addDistanceAndDirection:
                    epoch_correlationLossTest = 0
                
                if self.addSELoss:
                    epoch_SELossTest = 0

                epoch_landmarkAccuracyP1Test = 0
                epoch_landmarkAccuracyP2Test = 0
                epoch_landmarkAccuracyP3Test = 0
                epoch_landmarkAccuracyP4Test = 0

                epoch_labelAccuracyAllTest = 0
                epoch_labelAccuracyB0Test = 0
                epoch_labelAccuracyB1Test = 0
                epoch_labelAccuracyB2Test = 0
                epoch_labelAccuracyB3Test = 0
                epoch_labelAccuracyB4Test = 0
                epoch_labelAccuracyB5Test = 0

                ## Evaluating on test images
                model.eval()
                with torch.no_grad():
                    # epochValidateAccuracy = 0

                    ## Evaluating test groups
                    for (batch_idx, batch) in enumerate(ValidationDataLoader):
                        x = batch[0]
                        x=torch.tensor(x, device = device, dtype = torch.float32)
                        yHeatmap = batch[1] *100
                        yHeatmap=torch.tensor(yHeatmap, device = device, dtype = torch.float32)
                        ySegmentation = batch[2]
                        ySegmentation = torch.tensor(ySegmentation, device=device, dtype=torch.int64)

                        if (self.addDistanceAndDirection) & (self.addSELoss):
                            xDisplacement = batch[3]
                            xDisplacement = torch.tensor(xDisplacement, device = device, dtype = torch.float32)
                            yDistance = batch[4]
                            yDistance = torch.tensor(yDistance, device = device, dtype = torch.float32)

                        elif (self.addDistanceAndDirection == False) & (self.addSELoss):
                            xDisplacement = batch[3]
                            xDisplacement = torch.tensor(xDisplacement, device = device, dtype = torch.float32)

                        elif (self.addDistanceAndDirection) & (self.addSELoss == False):
                            yDistance = batch[3]
                            yDistance = torch.tensor(yDistance, device = device, dtype = torch.float32)

                       
                        if self.addDistanceAndDirection & self.addSELoss:
                            heatmap_pred, segmentation_pred, distance_pred, se_pred = model(x)
                        elif self.addDistanceAndDirection & (self.addSELoss == False):
                            heatmap_pred, segmentation_pred, distance_pred = model(x)
                        elif (self.addDistanceAndDirection == False) & (self.addSELoss):
                            heatmap_pred, segmentation_pred, se_pred = model(x)
                        else:
                            heatmap_pred, segmentation_pred = model(x)

                        # Quantifying loss
                            
                        ## Quantifying losses
                        # segmentationLoss = loss_segmentation(segmentation_pred, ySegmentation)
                        if self.addDistanceAndDirection & self.addSELoss:
                            heatmapLoss = self.lambdaHeatmapLoss * loss_heatmaps(heatmap_pred, yHeatmap)
                            segmentationLoss = self.lambdaSegmentationLoss * loss_segmentation(segmentation_pred, ySegmentation)
                            distanceLoss = self.lambdaCorrelationLoss * loss_heatmaps(distance_pred, yDistance)
                            # seLoss = self.lambdaContext * loss_heatmaps(se_pred, xDisplacement)
                            seLoss = self.lambdaContext * loss_heatmaps(se_pred, xDisplacement)
                            # directionLoss = loss_heatmaps(direction_pred, yDirection)
                            loss = heatmapLoss + segmentationLoss + distanceLoss + seLoss
                        elif self.addDistanceAndDirection & (self.addSELoss == False):
                            heatmapLoss = self.lambdaHeatmapLoss * loss_heatmaps(heatmap_pred, yHeatmap)
                            segmentationLoss = self.lambdaSegmentationLoss * loss_segmentation(segmentation_pred, ySegmentation)
                            distanceLoss = self.lambdaCorrelationLoss * loss_heatmaps(distance_pred, yDistance)
                            # directionLoss = loss_heatmaps(direction_pred, yDirection)
                            loss = heatmapLoss + segmentationLoss + distanceLoss
                        elif (self.addDistanceAndDirection == False) & self.addSELoss:
                            heatmapLoss = self.lambdaHeatmapLoss * loss_heatmaps(heatmap_pred, yHeatmap)
                            segmentationLoss = self.lambdaSegmentationLoss * loss_segmentation(segmentation_pred, ySegmentation)
                            # seLoss = self.lambdaContext * loss_heatmaps(se_pred, xDisplacement)
                            seLoss = self.lambdaContext * loss_heatmaps(se_pred, xDisplacement)
                            # directionLoss = loss_heatmaps(direction_pred, yDirection)
                            loss = heatmapLoss + segmentationLoss + seLoss
                        else:
                            heatmapLoss = loss_heatmaps(heatmap_pred, yHeatmap)
                            segmentationLoss = self.lambdaSegmentationLoss * loss_segmentation(segmentation_pred, ySegmentation)
                            loss =  heatmapLoss + segmentationLoss
                            
                        labelsLoss = segmentationLoss.data
                        LandmarkLoss = heatmapLoss.data
                        targetLoss = labelsLoss + LandmarkLoss

                        batchLoss = loss.data
                        epoch_lossTest += batchLoss * x.shape[0] / nValidationImages

                        epoch_segmentationLossTest += labelsLoss * x.shape[0] / nValidationImages
                        epoch_heatmapLossTest += LandmarkLoss * x.shape[0] / nValidationImages
                        epoch_targetLossTest += targetLoss * x.shape[0] / nValidationImages

                        if self.addDistanceAndDirection:
                            epoch_correlationLossTest +=  distanceLoss * x.shape[0] / nValidationImages
                        if self.addSELoss:
                            epoch_SELossTest += seLoss * x.shape[0] / nValidationImages

                        errorP1 = calculateLandmarkError(heatmap_pred, yHeatmap, 0)
                        errorP2 = calculateLandmarkError(heatmap_pred, yHeatmap, 1)
                        errorP3 = calculateLandmarkError(heatmap_pred, yHeatmap, 2)
                        errorP4 = calculateLandmarkError(heatmap_pred, yHeatmap, 3)
                        epoch_landmarkAccuracyP1Test += errorP1/nValidationImages
                        epoch_landmarkAccuracyP2Test += errorP2/nValidationImages
                        epoch_landmarkAccuracyP3Test += errorP3/nValidationImages
                        epoch_landmarkAccuracyP4Test += errorP4/nValidationImages

                        accuracyAll = calculateLabelAccuracy(segmentation_pred, ySegmentation)
                        accuracyB0 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 0)
                        accuracyB1 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 1)
                        accuracyB2 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 2)
                        accuracyB3 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 3)
                        accuracyB4 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 4)
                        accuracyB5 = calculateLabelAccuracy(segmentation_pred, ySegmentation, 5)

                        epoch_labelAccuracyAllTest += accuracyAll/nValidationImages
                        epoch_labelAccuracyB0Test += accuracyB0/nValidationImages
                        epoch_labelAccuracyB1Test += accuracyB1/nValidationImages
                        epoch_labelAccuracyB2Test += accuracyB2/nValidationImages
                        epoch_labelAccuracyB3Test += accuracyB3/nValidationImages
                        epoch_labelAccuracyB4Test += accuracyB4/nValidationImages
                        epoch_labelAccuracyB5Test += accuracyB5/nValidationImages

            ## Logging the epoch

            if logPath is not None:
                writer.add_scalar('Loss', epoch_loss, epoch)
                writer.add_scalar('Segmentation loss', epoch_segmentationLoss, epoch)
                writer.add_scalar('Heatmap loss', epoch_heatmapLoss, epoch)
                writer.add_scalar('P1 error', epoch_landmarkAccuracyP1, epoch)
                writer.add_scalar('P2 error', epoch_landmarkAccuracyP2, epoch)
                writer.add_scalar('P3 error', epoch_landmarkAccuracyP3, epoch)
                writer.add_scalar('P4 error', epoch_landmarkAccuracyP4, epoch)
                writer.add_scalar('Overall accuracy', epoch_labelAccuracyAll, epoch)
                writer.add_scalar('Background accuracy', epoch_labelAccuracyB0, epoch)
                writer.add_scalar('Bone 1 accuracy', epoch_labelAccuracyB1, epoch)
                writer.add_scalar('Bone 2 accuracy', epoch_labelAccuracyB2, epoch)
                writer.add_scalar('Bone 3 accuracy', epoch_labelAccuracyB3, epoch)
                writer.add_scalar('Bone 4 accuracy', epoch_labelAccuracyB4, epoch)
                writer.add_scalar('Bone 5 accuracy', epoch_labelAccuracyB5, epoch)

                if self.addDistanceAndDirection:
                    writer.add_scalar('Correlation error', epoch_correlationLoss, epoch)
                if self.addSELoss:
                    writer.add_scalar('Context loss', epoch_SELoss, epoch)
                if epoch % self.validationFrequency == 0 or epoch == self.epochs-1:
                    writerValidate.add_scalar('Loss', epoch_lossTest, epoch)
                    writerValidate.add_scalar('Segmentation loss', epoch_segmentationLossTest, epoch)
                    writerValidate.add_scalar('Heatmap loss', epoch_heatmapLossTest, epoch)
                    writerValidate.add_scalar('P1 error', epoch_landmarkAccuracyP1Test, epoch)
                    writerValidate.add_scalar('P2 error', epoch_landmarkAccuracyP2Test, epoch)
                    writerValidate.add_scalar('P3 error', epoch_landmarkAccuracyP3Test, epoch)
                    writerValidate.add_scalar('P4 error', epoch_landmarkAccuracyP4Test, epoch)
                
                    writerValidate.add_scalar('Overall accuracy', epoch_labelAccuracyAllTest, epoch)
                    writerValidate.add_scalar('Background accuracy', epoch_labelAccuracyB0Test, epoch)
                    writerValidate.add_scalar('Bone 1 accuracy', epoch_labelAccuracyB1Test, epoch)
                    writerValidate.add_scalar('Bone 2 accuracy', epoch_labelAccuracyB2Test, epoch)
                    writerValidate.add_scalar('Bone 3 accuracy', epoch_labelAccuracyB3Test, epoch)
                    writerValidate.add_scalar('Bone 4 accuracy', epoch_labelAccuracyB4Test, epoch)
                    writerValidate.add_scalar('Bone 5 accuracy', epoch_labelAccuracyB5Test, epoch)

                    if self.addDistanceAndDirection:
                        writerValidate.add_scalar('Correlation error', epoch_correlationLossTest, epoch)
                    if self.addSELoss:
                        writerValidate.add_scalar('Context loss', epoch_SELossTest, epoch)

            if verbose:
                print('#### Epoch {:05d}/{:05d}. Training Loss: {:.5f}'.format(epoch, self.epochs, epoch_loss))
                print('#### Epoch {:05d}/{:05d}. Validation Loss: {:.5f}'.format(epoch, self.epochs, epoch_lossTest))
            
            if epoch % self.trainingSaveFrequency == 0 or epoch == self.epochs-1:
                
                if verbose:
                    print(f'Saving model to: {modelPath}')

                if numberOfDevices > 1: # model is a DataParallel object
                    torch.save(model.module, os.path.join(modelPath, 'Model_{}'.format(epoch)))
                else:
                    torch.save(model, os.path.join(modelPath, 'Model_{}'.format(epoch)))

            if epoch % self.validationFrequency == 0 or epoch == self.epochs-1:
                if epoch_targetLossTest<BestValidationLoss:
                    checkpoint = {
                            'epoch': epoch + 1,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'loss': loss,
                        }
                    
                    BestLandmarkError[0] = epoch_landmarkAccuracyP1Test
                    BestLandmarkError[1] = epoch_landmarkAccuracyP2Test
                    BestLandmarkError[2] = epoch_landmarkAccuracyP3Test
                    BestLandmarkError[3] = epoch_landmarkAccuracyP4Test
                    np.savetxt(os.path.join(modelPath, 'LandmarkError.txt'), BestLandmarkError)

                    BestLabelingAccuracy[0] = epoch_labelAccuracyAllTest
                    BestLabelingAccuracy[1] = epoch_labelAccuracyB0Test
                    BestLabelingAccuracy[2] = epoch_labelAccuracyB1Test
                    BestLabelingAccuracy[3] = epoch_labelAccuracyB2Test
                    BestLabelingAccuracy[4] = epoch_labelAccuracyB3Test
                    BestLabelingAccuracy[5] = epoch_labelAccuracyB4Test
                    BestLabelingAccuracy[6] = epoch_labelAccuracyB5Test

                    np.savetxt(os.path.join(modelPath, 'LabelAccuracy.txt'), BestLabelingAccuracy)

                    BestValidationLoss = epoch_targetLossTest
                    if numberOfDevices > 1: # model is a DataParallel object
                        torch.save(model.module, os.path.join(modelPath, 'Model_Best'))
                        torch.save(checkpoint, os.path.join(modelPath, 'Model_Best.pt'))
                    else:
                        torch.save(model, os.path.join(modelPath, 'Model_Best'))
                        torch.save(checkpoint, os.path.join(modelPath, 'Model_Best.pt'))

        # Closing the logger
        if logPath is not None:
            writer.close()
            writerValidate.close()