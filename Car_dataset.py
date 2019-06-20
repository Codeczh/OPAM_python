import os
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
from PIL import Image



class Car196(torch.utils.data.Dataset):
    """
    Arguments:
        _root                [str]                   root directory of the dataset
        _train               [bool]                  load train/test dataset
        _transform           [callable]              a function/transform that takes in an image and transform it
        _train_data          [list of np.ndarray]
        _train_labels        [list of int]
        _test_data           [list of np.ndarray]
        _test_labels         [list of int]
    """
    def __init__(self, root, train=True, transform=None, get_raw_imgsize=False,get_nameid = False):
        """
        Load the dataset

        Arguments:
            root               [str]        root directory of the dataset
            train              [bool]       load train/test dataset, default: True
            transform          [callable]   a function/transform that takes in an image and transform it, default: None
        """
        self._root = os.path.expanduser(root)  # replace the '~' by the complete dir location
        self._train = train
        self._transform = transform
        self.get_raw_imgsize = get_raw_imgsize
        self.get_nameid =get_nameid

        #   to the ImageFolder structure
        image_path = os.path.join(root, 'images/')
        self._train_image_id=[]
        self._train_image_size = []
        self._train_data=[]
        self._train_labels=[]
        self._train_nameid=[]
        self._test_image_id=[]
        self._test_data=[]
        self._test_labels=[]
        self._test_image_size = []
        self._test_nameid=[]
        # load data
        if self._train:
            id_2_train = np.genfromtxt(os.path.join(self._root, 'train.list'), dtype=str)

            for idx in range(id_2_train.shape[0]):
                # fp = open(os.path.join(image_path, id_2_train[idx, 0]),'rb')
                # image = Image.open(fp)
                image = Image.open(os.path.join(image_path, id_2_train[idx, 0]))
                size = image.size
                label = int(id_2_train[idx, 1])  # Label starts from 0

                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()
                # fp.close()

                self._train_data.append(image_np)
                self._train_labels.append(label)
                self._train_image_id.append(idx)
                self._train_image_size.append(size)
                self._train_nameid.append(id_2_train[idx, 0])
                # print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_train.shape[0]), end='\r')
                print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_train.shape[0]))



        else:
            id_2_test = np.genfromtxt(os.path.join(self._root, 'test.list'), dtype=str)

            for idx in range(id_2_test.shape[0]):
                image = Image.open(os.path.join(image_path, id_2_test[idx, 0]))
                size = image.size
                label = int(id_2_test[idx, 1])  # Label starts from 0
                # pytorch only takes label as [0, num_classes) to calc loss, [1, num_classes] will get an error in loss calc

                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()
                self._test_data.append(image_np)
                self._test_labels.append(label)
                self._test_image_id.append(idx)
                self._test_image_size.append(size)
                self._test_nameid.append(id_2_test[idx,0])
                # print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_test.shape[0]), end='\r')
                print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_test.shape[0]))




    def __getitem__(self, index):
        """
        Get items from the dataset

        Argument:
            index       [int]               image/label index
        Return:
            image       [numpy.ndarray]     image of the given index
            label_gt    [int]               label of the given index
        """
        if self._train:
            nameid,imgsize, image, label_gt = self._train_nameid[index],self._train_image_size[index], \
                                       self._train_data[index], self._train_labels[index]
        else:
            nameid,imgsize, image, label_gt = self._test_nameid[index],self._test_image_size[index],\
                                       self._test_data[index], self._test_labels[index]

        if self._transform is not None:
            image = self._transform(image)

        if self.get_raw_imgsize:
            imgsize = torch.from_numpy(np.array(imgsize))
            if self.get_nameid:
                return image, label_gt, imgsize,nameid
            else:
                return  image, label_gt, imgsize
        else:
            if self.get_nameid:
                return image, label_gt, nameid
            else:
                return image, label_gt

    def __len__(self):
        """
        Length of the dataset

        Return:
            [int] Length of the dataset
        """
        if self._train:
            return len(self._train_data)
        else:
            return len(self._test_data)
