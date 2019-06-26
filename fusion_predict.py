#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torchvision
import time
from torch.utils.data import DataLoader
import yaml
import numpy as np
from PIL import Image
from patchnet_bn_rerun import PatchNet
import operator
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(0)
torch.cuda.manual_seed(0)

class Data_test(torch.utils.data.Dataset):
    '''
    Arguments:
        _root               [str]                   root directory of the dataset
        _train              [bool]                  load train/test dataset
        _transform          [callable]              a function/transform that takes in an image and transform it
    '''

    def __init__(self, root, train=True, test='image', transform=None, get_img_id=False):
        '''
        load the dataset
        :param root:
        :param train:
        :param transform:
        :param get_img_id:
        '''
        self._root = os.path.expanduser(root)  # replace the '~' by the complete dir location
        self._train = train
        self._test = test
        self._transform = transform
        self.__get_img_id = get_img_id
        #  to store id,image,label
        self._train_image_id = []
        self._train_data = []
        self._train_labels = []
        self._test_image_id = []
        self._test_data = []
        self._test_labels = []
        # load data
        if self._train:
            train_image_path = os.path.join(self._root, 'images')
            train_bbox_path = os.path.join(self._root,'bbox')
            train_part_path = os.path.join(self._root,'parts_align_9')
            id_2_train = np.genfromtxt(os.path.join(self._root, 'train.list'), dtype=str)
            #bbox
            for idx in range( id_2_train.shape[0]):
                image = Image.open(os.path.join(train_bbox_path,id_2_train[idx, 0][0:6]+'.png'))
                label = int(id_2_train[idx, 1])

                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()

                self._train_data.append(image_np)
                self._train_labels.append(label)
                self._train_image_id.append(idx)
                print('>>>>>> Processed train1/4 bbox {} / {} images'.format(idx + 1 ,id_2_train.shape[0] ))
            #image
            for idx in range( id_2_train.shape[0]):
                image = Image.open(os.path.join(train_image_path,id_2_train[idx, 0]))
                label = int(id_2_train[idx, 1])

                #   ./bbox/000001_bbox.jpg 0
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)

                image.close()

                self._train_data.append(image_np)
                self._train_labels.append(label)
                self._train_image_id.append(idx)
                print('>>>>>> Processed train2/4 image {} / {} images'.format(idx + 1 ,id_2_train.shape[0] ))
            # part1
            for idx in range( id_2_train.shape[0]):
                image = Image.open(os.path.join(train_part_path,id_2_train[idx, 0][0:6]+'_1'+id_2_train[idx,0][6:]))
                label = int(id_2_train[idx, 1])

                #   ./bbox/000001_bbox.jpg 0
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)

                image.close()

                self._train_data.append(image_np)
                self._train_labels.append(label)
                self._train_image_id.append(idx)
                print('>>>>>> Processed train3/4 part1 {} / {} images'.format(idx + 1 ,id_2_train.shape[0] ))
            # part2
            for idx in range( id_2_train.shape[0]):
                image = Image.open(os.path.join(train_part_path,id_2_train[idx, 0][0:6]+'_2'+id_2_train[idx,0][6:]))
                label = int(id_2_train[idx, 1])

                #   ./bbox/000001_bbox.jpg 0
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)

                image.close()

                self._train_data.append(image_np)
                self._train_labels.append(label)
                self._train_image_id.append(idx)
                print('>>>>>> Processed train4/4 part2 {} / {} images'.format(idx + 1 ,id_2_train.shape[0] ))
        else:
            test_image_path = os.path.join(self._root, 'images')
            id_2_test = np.genfromtxt(os.path.join(self._root, 'test.list'), dtype=str)
            test_bbox_path = os.path.join(self._root,'bbox')
            test_part_path = os.path.join(self._root, 'parts_align_9')
            if self._test=='image':
                for idx in range(id_2_test.shape[0]):
                    image = Image.open(os.path.join(test_image_path, id_2_test[idx, 0]))
                    label = int(id_2_test[idx, 1])
                    # 000046.jpg 0
                    if image.mode == 'L':
                        image = image.convert('RGB')
                    image_np = np.array(image)
                    image.close()

                    self._test_data.append(image_np)
                    self._test_labels.append(label)
                    self._test_image_id.append(idx)
                    print('>>>>>> Processed test1/4 image {} / {} images --image--test'.format(idx + 1, id_2_test.shape[0]))
            elif self._test == 'bbox':
                for idx in range(id_2_test.shape[0]):
                    image = Image.open(os.path.join(test_bbox_path, id_2_test[idx, 0][0:6]+'.png'))
                    label = int(id_2_test[idx, 1])

                    # ./bbox/000046_bbox.jpg 0
                    if image.mode == 'L':
                        image = image.convert('RGB')
                    image_np = np.array(image)

                    image.close()

                    self._test_data.append(image_np)
                    self._test_labels.append(label)
                    self._test_image_id.append(idx)
                    print('>>>>>> Processed test2/4 bbox {} / {} images --bbox--test'.format(idx + 1, id_2_test.shape[0]))
            elif self._test == 'part1':
                for idx in range(id_2_test.shape[0]):
                    image = Image.open(
                        os.path.join(test_part_path, id_2_test[idx, 0][0:6] + '_1' + id_2_test[idx, 0][6:]))
                    label = int(id_2_test[idx, 1])

                    #   ./bbox/000001_bbox.jpg 0
                    if image.mode == 'L':
                        image = image.convert('RGB')
                    image_np = np.array(image)

                    image.close()

                    self._test_data.append(image_np)
                    self._test_labels.append(label)
                    self._test_image_id.append(idx)
                    print('>>>>>> Processed train3/4 part1 {} / {} images'.format(idx + 1, id_2_test.shape[0]))
            elif self._test=='part2':
                for idx in range(id_2_test.shape[0]):
                    image = Image.open(
                        os.path.join(test_part_path, id_2_test[idx, 0][0:6] + '_2' + id_2_test[idx, 0][6:]))
                    label = int(id_2_test[idx, 1])

                    #   ./bbox/000001_bbox.jpg 0
                    if image.mode == 'L':
                        image = image.convert('RGB')
                    image_np = np.array(image)

                    image.close()

                    self._test_data.append(image_np)
                    self._test_labels.append(label)
                    self._test_image_id.append(idx)
                    print('>>>>>> Processed train4/4 part2 {} / {} images'.format(idx + 1, id_2_test.shape[0]))

    def __getitem__(self, index):
        '''
        :param index:
        :return:
        '''
        if self._train:
            img_id, image, label_gt = self._train_image_id[index], self._train_data[index], self._train_labels[index]
        else:
            img_id, image, label_gt = self._test_image_id[index], self._test_data[index], self._test_labels[index]
        if self._transform is not None:
            image = self._transform(image)
        if self.__get_img_id:
            return img_id, image, label_gt
        else:
            return image, label_gt

    def __len__(self):
        '''

        :return:
        '''
        if self._train:
            return len(self._train_data)
        else:
            return len(self._test_data)


class FusionPredict(object):
    def __init__(self, options, data_model_path):
        self._options = options
        self._path = data_model_path
        self._test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # Network
        net = PatchNet(pretrained=False,classnum = self._options['classnum'])
        # torch.cuda.set_device(1)
        # self._net = net.cuda()
        # if torch.cuda.device_count() > 1:
        #     self._net = torch.nn.DataParallel(net).cuda()
        # elif torch.cuda.device_count() == 1:
        #     self._net = net.cuda()
        # else:
        #     raise EnvironmentError('This is designed to run on GPU but no GPU is found')
        self._net = net.cuda()
        # Score Container {key: imgID, value:(score(1,200), label)}
        self._classnet_score_dict = {}
        self._objectnet_score_dict = {}
        self._partnet_score_dict = {}
        # Coefficients for each score
        # load test image
        test_image_data = Data_test(root=self._path['root'], train=False, test='image', transform=self._test_transform,
                                      get_img_id=True)
        self._image_dataloader = DataLoader(test_image_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        # load test bbox
        test_bbox_data = Data_test(root=self._path['root'], train=False, test='bbox', transform=self._test_transform, get_img_id=True)
        self._bbox_dataloader = DataLoader(test_bbox_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        # load test part
        test_part1_data = Data_test(root=self._path['root'], train=False, test='part1', transform=self._test_transform,
                                      get_img_id=True)
        test_part2_data = Data_test(root=self._path['root'], train=False, test='part2', transform=self._test_transform,
                                      get_img_id=True)
        self._part1_dataloader1 = DataLoader(test_part1_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        self._part2_dataloader2 = DataLoader(test_part2_data, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        #
        print('load data over...')

        # print('Using model: \n\t{}'.format(self._path['patchnet_vgg19bn_model']))
        self._get_classnet_score()
        print('>>> ClassNet Score Got!')

        # print('Using model: \n\t{}'.format(self._path['objectnet_vgg19bn_my_model']))
        self._get_objectnet_score()
        print('>>> ObjectNet Score Got!')

        # print('Using model: \n\t{}'.format(self._path['partnet_vgg19bn_my_model']))
        self._get_partnet_score()
        print('>>> PartNet Score Got!')

    def _get_classnet_score(self):
        #,map_location={'cuda:3': 'cuda:0'}
        self._net.load_state_dict(torch.load(os.path.join(self._path['model'],self._path['patchnet_vgg19bn_model'])))# patchnet_vgg19_bn_best_epoch.pth
        # if isinstance(self._net,torch.nn.DataParallel):
        #     self._net = self._net.moudle
        self._net.eval()
        with  torch.no_grad():
            for imgID, img, label in self._image_dataloader:
                N = img.shape[0]
                img = img.cuda()  # (N, 3, 224, 224)
                score = self._net(img)  # (N, 200)
                for i in range(N):
                    self._classnet_score_dict[imgID[i].item()] = (score[i].cpu().numpy(), label[i].item())

    def _get_objectnet_score(self):
        #,map_location={'cuda:1': 'cuda:0'}
        self._net.load_state_dict(torch.load(os.path.join(self._path['model'],self._path['objectnet_vgg19bn_model'])))  # objectnet_vgg19_bn_best_epoch.pth
        self._net.eval()
        with torch.no_grad():
            for imgID, img, label in self._bbox_dataloader:
                N = img.shape[0]
                img = img.cuda()  # (N, 3, 224, 224)
                score = self._net(img)  # (N, 200)
                for i in range(N):
                    self._objectnet_score_dict[imgID[i].item()] = (score[i].cpu().numpy(), label[i].item())

    def _get_partnet_score(self):
        #,map_location={'cuda:0': 'cuda:0'}
        self._net.load_state_dict(torch.load(os.path.join(self._path['model'],self._path['partnet_vgg19bn_model'])))  # partnet_vgg19_bn_best_epoch.pth
        self._net.eval()
        with torch.no_grad():
            for imgID, img, label in self._part1_dataloader1 :
                N = img.shape[0]
                img = img.cuda()  # (N, 3, 224, 224)
                score = self._net(img)  # (N, 200)
                for i in range(N):
                    self._partnet_score_dict[imgID[i].item()] = [score[i].cpu().numpy(), label[i].item()]
            for imgID, img, label in self._part2_dataloader2:
                N = img.shape[0]
                img = img.cuda()  # (N, 3, 224, 224)
                score = self._net(img)  # (N, 200)
                for i in range(N):
                    tmp = self._partnet_score_dict[imgID[i].item()][0]
                    tmp = (tmp + score[i].cpu().numpy()) / 2
                    self._partnet_score_dict[imgID[i].item()] = (tmp, label[i].item())

    def _score_fusion(self, img_id):
        class_score, label_class = self._classnet_score_dict[img_id]
        object_score, label_object = self._objectnet_score_dict[img_id]
        part_score, label_part = self._partnet_score_dict[img_id]
        assert label_class == label_object and label_class == label_part


        origin_score = class_score
        origin_object_score = 0.5 * class_score + 0.5 * object_score
        origin_part_score = 0.5 * class_score + 0.5 * part_score
        object_part_score = 0.5 * object_score + 0.5 * part_score
        return  origin_score,object_score, part_score, origin_object_score, origin_part_score, object_part_score

    def __score(self,img_id,alpha,beta,gama):
        score = alpha * self._classnet_score_dict[img_id][0] + beta * self._objectnet_score_dict[img_id][0] + gama * self._partnet_score_dict[img_id][0]  # (1,200)
        return score
    def predict(self):

        num_total = 0
        num_correct = 0
        num_correct_origin = 0
        num_correct_object = 0
        num_correct_part = 0
        num_correct_origin_object = 0
        num_correct_origin_part = 0
        num_correct_object_part = 0
        for imgID in self._classnet_score_dict.keys():
            # score = self._score_fusion(imgID)
            score = self.__score(imgID,0.4,0.4,0.2)
            origin_score,object_score, part_score,\
            origin_object_score, origin_part_score, object_part_score \
                                                = self._score_fusion(imgID)
            label = self._classnet_score_dict[imgID][1]
            # max_score, prediction = torch.max(score, 1)
            prediction = np.argmax(score)
            predict_origin = np.argmax(origin_score)
            predict_object = np.argmax(object_score)
            predict_part = np.argmax(part_score)
            predict_origin_object = np.argmax(origin_object_score)
            predict_origin_part = np.argmax(origin_part_score)
            predict_object_part = np.argmax(object_part_score)
            num_total += 1
            if prediction == label:
                num_correct += 1
            if predict_origin == label:
                num_correct_origin += 1
            if predict_object == label:
                num_correct_object += 1
            if predict_part == label:
                num_correct_part +=1
            if predict_origin_object == label:
                num_correct_origin_object += 1
            if predict_origin_part == label:
                num_correct_origin_part += 1
            if predict_object_part == label:
                num_correct_object_part +=1
        predict_accuracy = 100 * num_correct / num_total
        predict_accuracy_origin = 100 * num_correct_origin / num_total
        predict_accuracy_object = 100 * num_correct_object / num_total
        predict_accuracy_part = 100 * num_correct_part / num_total
        predict_accuracy_origin_object = 100 * num_correct_origin_object / num_total
        predict_accuracy_origin_part = 100 * num_correct_origin_part / num_total
        predict_accuracy_object_part = 100 * num_correct_object_part / num_total

        print('---------------------------------------------------')
        print('| Fusion Predict Accuracy is {}                    '.format(predict_accuracy))
        print('| Origin Predict Accuracy is {}                    '.format(predict_accuracy_origin))
        print('| Object Predict Accuracy is {}                    '.format(predict_accuracy_object))
        print('| Part   Predict Accuracy is {}                    '.format(predict_accuracy_part))
        print('| Origin + Object Predict Accuracy is {}           '.format(predict_accuracy_origin_object))
        print('| Origin + Part   Predict Accuracy is {}           '.format(predict_accuracy_origin_part))
        print('| Object + Part   Predict Accuracy is {}           '.format(predict_accuracy_object_part))
        print('---------------------------------------------------')
        list = {}
        i=0
        maxscore = 0
        best_alpha = 0
        best_beta = 0
        best_gama = 0
        for a in range(0, 11, 1):
            for b in range(0, 11 - a, 1):
                step_start = time.time()
                num_total = 0
                num_correct = 0
                for imgID in self._classnet_score_dict.keys():
                    # score = self._score_fusion(imgID)
                    score = self.__score(imgID,a/10,a/10,(10-a-b)/10)
                    label = self._classnet_score_dict[imgID][1]
                    # max_score, prediction = torch.max(score, 1)
                    prediction = np.argmax(score)
                    num_total += 1
                    if prediction == label:
                        num_correct += 1
                predict_accuracy = 100 * num_correct / num_total
                print('|   alpha = {}   beta = {}   gama = {}      score:{}'.format(a/10,b/10,(10-a-b)/10,predict_accuracy))
                list[i] = [a/10,b/10,(10-a-b)/10,predict_accuracy]
                i+=1
                if predict_accuracy > maxscore:
                    maxscore = predict_accuracy
                    best_alpha, best_beta, best_gama = a / 10, b / 10, (10 - a - b) / 10
                step_end = time.time()
                print('step_time-----{}'.format(step_end - step_start))
        print('|  best_score:{}  alpha = {}   beta = {}   gama = {}'.format(maxscore, best_alpha, best_beta, best_gama))
        x = sorted(list.items(),key = lambda v: operator.itemgetter(1)(v)[3])
        for i in range(len(x)):
            print(x[len(x)-1-i])
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default='CUB200')
    args = parser.parse_args()
    dataset = args.dataset
    root = os.popen('pwd').read().strip()
    root = os.path.join(root, dataset)
    config = yaml.load(open(root+'/config.yaml', 'r'))
    config['classnum'] = int(config['classnum'])
    path = {
        'root': root,
        # 'patchnet_vgg19bn_model': os.path.join(root, config['patchnet_vgg19bn_model']),
        # 'objectnet_vgg19bn_my_model': os.path.join(root, config['objectnet_vgg19bn_my_model']),
        # 'partnet_vgg19bn_my_model': os.path.join(root, config['partnet_vgg19bn_my_model']),
        'model':root+'/model',
        'patchnet_vgg19bn_model': 'patchnet_vgg19bn_rerun_best_epoch.pth',
        'objectnet_vgg19bn_model': 'objectnet_vgg19bn_rerun_best_epoch.pth',
        'partnet_vgg19bn_model': 'partnet_vgg19bn_rerun_best_epoch.pth'

    }
    start = time.time()

    fusionPredictor = FusionPredict(config, path)

    fusionPredictor.predict()

    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))

