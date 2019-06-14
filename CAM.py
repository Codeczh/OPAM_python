#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step 1 (saliencynet.py) :
    fine tune pre-trained vgg19 on the cub200 dataset and save the best model
step 2 (object_extractor.py):
    use the saved model to extract the saliency map using CAM and get the object bounding box subsequently
"""

import os
import time
import torch
from saliencynet import SaliencyNet
import torchvision
# from Car_dataset import Car196
from torch.utils.data import DataLoader
import yaml
import numpy as np
import cv2
from scipy import ndimage
import pickle
import matplotlib.pyplot as plt
from collections import OrderedDict
from PIL import Image
import argparse

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
    def __init__(self, root, train=True,threadid=0, transform=None, get_raw_imgsize=False,get_nameid = False):
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
        self.threadid = threadid

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

            for idx in range(1000*self.threadid,min(1000*(self.threadid+1),id_2_train.shape[0])):
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

            for idx in range(1000*self.threadid,min(1000*(self.threadid+1),id_2_test.shape[0])):
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

class ObjectExtractor(object):
    def __init__(self, data_model_path,thread=0):
        super().__init__()
        self.conv_feature_blobs = []
        self._thread = thread
        self._objDict = {}  # {key: imgID, value: (objBBox, saliency, label)}
        # self._img_id = []
        # self._img_bbox = []
        # self._img_label = []
        # self._img_saliency = []

        self._path = data_model_path
        # Net
        net = SaliencyNet(pretrained=False)
        # torch.cuda.set_device(1)
        # self._net = net.cuda()

        if torch.cuda.device_count() > 1:
            self._net = torch.nn.DataParallel(net).cuda()
        elif torch.cuda.device_count() == 1:
            self._net = net.cuda()
        else:
            raise EnvironmentError('This is designed to run on GPU but no GPU is found')
        state_dict = torch.load(self._path['load_model'])
        # new_state_dict=OrderedDict()
        # for k,v in state_dict.items():
        #     name = k[7:]
        #     new_state_dict[name]=v
        self._net.load_state_dict(state_dict) # saliencynet_vgg19_best_epoch.pth

        # data
        data_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # TODO: here, only training phase involves saliency generation and bbox extraction
        train_data = Car196(self._path['root'], train=True, threadid=thread,transform=data_transform, get_raw_imgsize=True,get_nameid=True)
        self._old_train_data = Car196(self._path['root'], train=True, threadid=thread,transform=None, get_raw_imgsize=False,get_nameid=True)
        self._old_test_data = Car196(self._path['root'], train=False, threadid=thread,transform=None, get_raw_imgsize=False,get_nameid=True)
        test_data = Car196(self._path['root'], train=False, threadid=thread,transform=data_transform, get_raw_imgsize=True,get_nameid=True)
        self._train_dataloader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        self._test_dataloader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        # self._old_train_dataloader = DataLoader(old_train_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
        # self._old_test_dataloader = DataLoader(old_test_data, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    def saliency_extractor(self):
        """
        extract the saliency map and bbox and dump to pkl
        """
        if isinstance(self._net, torch.nn.DataParallel):
            self._net = self._net.module

        self._net.eval()
        if not os.path.exists(os.path.join(self._path['root'], 'heatmap_read')):
            os.mkdir(os.path.join(self._path['root'], 'heatmap_read'))
            print(os.path.join(self._path['root'], 'heatmap_read')+' have been created ...')
        if not os.path.exists(os.path.join(self._path['root'], 'heatmap_watch')):
            os.mkdir(os.path.join(self._path['root'], 'heatmap_watch'))
            print(os.path.join(self._path['root'], 'heatmap_watch')+' have been created ...')
        if not os.path.exists(os.path.join(self._path['root'], 'showbbox')):
            os.mkdir(os.path.join(self._path['root'], 'showbbox'))
            print(os.path.join(self._path['root'], 'showbbox')+' have been created ...')
        if not os.path.exists(os.path.join(self._path['root'], 'bbox')):
            os.mkdir(os.path.join(self._path['root'], 'bbox'))
            print(os.path.join(self._path['root'], 'bbox')+' have been created ...')
        if not os.path.exists(os.path.join(self._path['root'], 'datalist')):
            os.mkdir(os.path.join(self._path['root'], 'datalist'))
            print(os.path.join(self._path['root'], 'datalist')+' have been created ...')
        params = list(self._net.parameters())
        fc_weight = np.squeeze(params[-2].data.cpu().numpy())

        with torch.no_grad():
            for phase in ['train', 'test']:
                if phase=='train':
                    dataLoader = self._train_dataloader
                    old_data = self._old_train_data
                else:
                    dataLoader = self._test_dataloader
                    old_data = self._old_test_data
                num = 0
                bbox = []
                for img, label, raw_imgsize,nameid in dataLoader:
                    # Data
                    img = img.cuda()
                    # Clear the blobs of last conv layer feature output
                    self.conv_feature_blobs = []
                    # Hook
                    handler = self._net.conv.register_forward_hook(self.hook_feature)
                    # forward pass
                    score = self._net(img)  # N * 200
                    # now, self.conv_feature_blobs contains only one element which has the shape of (N, 1024, h. w)
                    # prediction
                    max_score, pred_idx = torch.max(score, 1)  # N * 200
                    # get CAM, CAMs shape: (N, (224, 224))
                    CAMs = self.return_cam(self.conv_feature_blobs[0], fc_weight, pred_idx, raw_imgsize)
                    for idx in range(len(CAMs)):
                        image_start_time = time.time()
                        cv2.imwrite(self._path['root']+'/heatmap_read/{}.png'.format(nameid[idx][0:6]), CAMs[idx])
                        plt.imshow(CAMs[idx],cmap='jet')
                        plt.xticks([])
                        plt.yticks([])
                        plt.savefig(os.path.join(self._path['root'], 'heatmap_watch', '{}.png'.format(nameid[idx][0:6])))
                        plt.clf()
                        image_end_time1 = time.time()
                        img_bbox = self.bbox_extractor_grabcut(CAMs[idx])  # (x, y, w, h)
                        img_label = label[idx].item()

                        x, y, width, height = img_bbox
                        bbox.append(str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' ' + '\n')

                        image_end_time2 = time.time()
                        num += 1
                        print('>>>>>> Processing {} image  {}  -----time:{}  {}  {}'.\
                              format(phase,num,image_end_time2-image_start_time,
                                     image_end_time1-image_start_time,image_end_time2-image_end_time1))
                    # remove the hook handler
                    handler.remove()

                print('>>>>>> Pickling object {} bounding box over ........... '.format(phase))
                with open(os.path.join(self._path['root'], 'datalist/bbox' + phase + '{}000.list'.format(self._thread)), 'w') as f:
                    f.writelines(bbox)
                total = 0
                for idx in range(len(old_data)):
                    image_start_time = time.time()
                    img,label,nameid = old_data.__getitem__(idx)
                    x, y, w, h = map(int, bbox[idx ].split())
                    rect = (img[y:y + h, x:x + w, :])
                    rect = torchvision.transforms.ToPILImage()(rect)
                    rect.save(os.path.join(self._path['root'], 'bbox', '{}.png'.format(nameid[0:6])))
                    rect.close()
                    img = img[..., ::-1]
                    showmap = cv2.rectangle(img.astype(np.uint8).copy() , (x, y), (x + w, y + h), (0, 255, 0), 1)
                    cv2.imwrite(os.path.join(self._path['root'], 'showbbox', '{}.png'.format(nameid[0:6])),
                                showmap)
                    image_end_time = time.time()
                    total += 1
                    print('>>>>>> Saving {} showbbox and bbox {}.png time:{}'.format(phase, total,image_end_time-image_start_time))






    def hook_feature(self, module, input, output):
        self.conv_feature_blobs.append(output.data.cpu().numpy())  # output : (N, 1024, h, w), numpy.ndarray

    @staticmethod
    def return_cam(feature_conv, weight, pred_class_idx,raw_imgsize):
        """
        Get the class attention mapping (CAM)
        :param feature_conv:
            numpy.ndarray           the feature output of the last convolution layer (N, 1024, h, w)
        :param weight:
            [numpy.ndarray]         the weight of the last fc layer (200 * 1024)
        :param pred_class_idx:
            [list(int)]             the list of prediction of this data batch (N, 1)
        :return:
            [list(numpy.ndarray)]   the list of output CAM, length is N, each element's shape is (224, 224)
        """
        size_upsample = (224, 224)
        batch_size, num_channel, h, w = feature_conv.shape
        output_cam = []
        for i, pred_class in enumerate(pred_class_idx):
            cam = weight[pred_class].dot(feature_conv[i].reshape((num_channel, h * w)))
            cam = cam.reshape((h, w))
            cam = cam - np.min(cam)
            cam_img = cam / np.max(cam)
            cam_img = np.uint8(255 * cam_img)
            output_cam.append(cv2.resize(cam_img, tuple(raw_imgsize[i])))
        return output_cam

    @staticmethod
    def bbox_extractor_zeren(saliency):
        """
        extract the bounding box based on the saliency map
        :param saliency:
            [numpy.ndarray] saliency map (h, w)
        :return:
            [tuple]         bounding box, elements are (x_min, y_min, h, w)
        """
        mask = saliency > saliency.mean()
        # label the connected area
        label_im, num_labels = ndimage.label(mask,
                                             structure=[[0, 1, 0],
                                                        [1, 1, 1],
                                                        [0, 1, 0]])

        # Find the largest connected component
        sizes = ndimage.sum(mask, label_im, range(num_labels + 1))
        mask_size = sizes < 1000
        remove_pixel = mask_size[label_im]
        label_im[remove_pixel] = 0
        labels = np.unique(label_im)
        label_im = np.searchsorted(labels, label_im)

        # Now that we have only one connected component, extract it's bounding box
        slice_x, slice_y = ndimage.find_objects(label_im == 1)[0]
        # roi = saliency[slice_x, slice_y]
        min_x = slice_x.start
        max_x = slice_x.stop
        min_y = slice_y.start
        max_y = slice_y.stop
        # print(min_x, min_y, max_x - min_x, max_y - min_y, roi.shape)
        return min_x, min_y, max_x - min_x, max_y - min_y

    @staticmethod
    def bbox_extractor_grabcut(saliency):
        mask = np.zeros((saliency.shape[:2]), np.uint8)
        ret, bm = cv2.threshold(saliency, 20, 255, cv2.THRESH_BINARY)
        mask[bm == 255] = cv2.GC_PR_BGD
        ret, bm = cv2.threshold(saliency, 40, 255, cv2.THRESH_BINARY)
        mask[bm == 255] = cv2.GC_PR_FGD
        rect = (0, 0, 224, 224)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)
        saliency_3 = cv2.cvtColor(saliency, cv2.COLOR_GRAY2RGB)
        cv2.grabCut(saliency_3, mask, rect, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        res = saliency * mask2[:,:]
        #image, contours, hierarchy = cv2.findContours(res, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        major = cv2.__version__.split('.')[0]
        if major == '3':
            _, contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            contours, _ = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        res_w=0
        res_h=0
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w*h >res_w*res_h:
                res_x = x
                res_y = y
                res_w = w
                res_h = h
        return res_x, res_y, res_w-1, res_h-1

    @staticmethod
    def bbox_extractor_otsu(saliency):
        # th=saliency.mean()
        ret, mask = cv2.threshold(saliency, 20, 255, cv2.THRESH_OTSU)
        res = saliency * mask
        image, contours, hierarchy = cv2.findContours(res, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        res_w=0
        res_h=0
        for i in range(0, len(contours)):
            x, y, w, h = cv2.boundingRect(contours[i])
            if w*h >res_w*res_h:
                res_x = x
                res_y = y
                res_w = w
                res_h = h
        return res_x, res_y, res_w, res_h



if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--threadid', type=int, default=0)
    #parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    thread = args.threadid

    root = os.popen('pwd').read().strip()
    root = os.path.join(root, 'CUB200')
    config = yaml.load(open(os.path.join(root, 'config.yaml'), 'r'))
    path = {
        # 'car196': os.path.join(root, 'data/cub200'),
        'root': root,
        #'car196': '/farm/czh/TIP/TIP',
        'load_model': 'saliencynet_vgg16_best_epoch.pth'
            #os.path.join(root, config['saliencynet_model'])
    }
    for k in path:
        if k is 'load_model':
            assert os.path.isfile(path[k])
        else:
            assert os.path.isdir(path[k])
    print('>>>--->>>\nUsing model:\n\t{} \n>>>--->>>'.format(path['load_model']))
    start = time.time()
    extract_manager = ObjectExtractor(path,thread)
    extract_manager.saliency_extractor()
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))

    ####commond
    # cd TIP/TIP
    # conda activate lijiang
    # python CAM.py --threadid 0 >CUB200/log/CAM/CAM0.log
