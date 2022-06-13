import numpy as np
import os
import glob
import torch.utils.data as data

class Dataset(data.Dataset):

    def __init__(self, root_path_CT,root_path_heatmap, root_path_segmentation, root_path_displacement = None, AddCorrelation = False, split = 'train'):
        
        self.split = split
        self.AddCorrelation = AddCorrelation

        self.data_numpy_list_CT = np.array([x for x in glob.glob(os.path.join(root_path_CT, '*.npy'))])
        self.data_numpy_list_heatmap = np.array([x for x in glob.glob(os.path.join(root_path_heatmap, '*.npy'))])
        self.data_numpy_list_segmentation = np.array([x for x in glob.glob(os.path.join(root_path_segmentation, '*.npy'))])

        def by_size(words, size):
            return [word for word in words if len(word) < size]

        self.data_numpy_list_CT = np.array(by_size(self.data_numpy_list_CT, 100))
        self.data_numpy_list_heatmap = np.array(by_size(self.data_numpy_list_heatmap, 100))
        self.data_numpy_list_segmentation = np.array(by_size(self.data_numpy_list_segmentation, 105))
        if root_path_displacement is not None:
            self.data_numpy_list_displacement = np.array([x for x in glob.glob(os.path.join(root_path_displacement, '*.npy'))])
            self.data_numpy_list_displacement = np.array(by_size(self.data_numpy_list_displacement, 105))
        nTotal = len(self.data_numpy_list_CT)
        index = np.arange(nTotal)
        np.random.seed(0)
        np.random.shuffle(index)
        self.data_numpy_list_CT = self.data_numpy_list_CT[index]
        if root_path_displacement is not None:
            self.data_numpy_list_displacement = self.data_numpy_list_displacement[index]
        self.data_numpy_list_heatmap = self.data_numpy_list_heatmap[index]
        self.data_numpy_list_segmentation = self.data_numpy_list_segmentation[index]

        if self.split is 'train':
            self.data_numpy_list_CT = self.data_numpy_list_CT[0: int(0.8*nTotal)]
            if root_path_displacement is not None:
                self.data_numpy_list_displacement = self.data_numpy_list_displacement[0: int(0.8*nTotal)]
            self.data_numpy_list_heatmap = self.data_numpy_list_heatmap[0: int(0.8*nTotal)]
            self.data_numpy_list_segmentation = self.data_numpy_list_segmentation[0: int(0.8*nTotal)]
        elif self.split is 'validation':
            self.data_numpy_list_CT = self.data_numpy_list_CT[int(0.8*nTotal):int(0.9*nTotal)]
            if root_path_displacement is not None:
                self.data_numpy_list_displacement = self.data_numpy_list_displacement[int(0.8*nTotal):int(0.9*nTotal)]
            self.data_numpy_list_heatmap = self.data_numpy_list_heatmap[int(0.8*nTotal):int(0.9*nTotal)]
            self.data_numpy_list_segmentation = self.data_numpy_list_segmentation[int(0.8*nTotal):int(0.9*nTotal)]
        else:
            self.data_numpy_list_CT = self.data_numpy_list_CT[int(0.9*nTotal):]
            if root_path_displacement is not None:
                self.data_numpy_list_displacement = self.data_numpy_list_displacement[int(0.9*nTotal):]
            self.data_numpy_list_heatmap = self.data_numpy_list_heatmap[int(0.9*nTotal):]
            self.data_numpy_list_segmentation = self.data_numpy_list_segmentation[int(0.9*nTotal):]

    def __getitem__(self, index):

        CT = np.load(self.data_numpy_list_CT[index])
        if len(CT.shape)==3:
            CT = np.expand_dims(CT,axis=0)
        
        if hasattr(self, "data_numpy_list_displacement"):
            displacement = np.load(self.data_numpy_list_displacement[index])
        heatmap = np.load(self.data_numpy_list_heatmap[index])

        segmentation = np.load(self.data_numpy_list_segmentation[index])
        
        outputMaps = [CT, heatmap, segmentation]
        if hasattr(self, "data_numpy_list_displacement"):
            outputMaps.append(displacement)
        if self.AddCorrelation == True:
            coords = np.zeros((4,3))
            for map in range(4):
                coords[map,:] = (np.reshape(np.where(heatmap[map] == (heatmap[map]).max()),-1))/96

            distanceAndDirections = np.zeros((12,3))

            distanceAndDirections[0] = coords[1] -coords[0]
            distanceAndDirections[1] = coords[2] -coords[0]
            distanceAndDirections[2] = coords[3] -coords[0]
            distanceAndDirections[3] = coords[2] -coords[1]
            distanceAndDirections[4] = coords[3] -coords[1]
            distanceAndDirections[5] = coords[3] -coords[2]
            
            distanceAndDirections[6] = coords[0] -coords[1]
            distanceAndDirections[7] = coords[0] -coords[2]
            distanceAndDirections[8] = coords[0] -coords[3]
            distanceAndDirections[9] = coords[1] -coords[2]
            distanceAndDirections[10] = coords[1] -coords[3]
            distanceAndDirections[11] = coords[2] -coords[3]

            outputMaps.append(distanceAndDirections.ravel())

        return outputMaps

    def __len__(self):
        return len(self.data_numpy_list_CT)
