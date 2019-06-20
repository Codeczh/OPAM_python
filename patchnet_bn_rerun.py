#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
step 1 (filternet.py):
    fine tune the pre-trained vgg19 on the cub200 training dataset and save the best model
step 2 (patch_filter.py):
    generate patches(region proposal) using selective search
    resize the proposed patches and feed into the CNN using the saved model
    filter out irrelevant patches using a threshold over the neural activation of its class in the softmax layer
"""
#  python patchnet.py >log/patchnet54.log

import os
import torch
import torchvision
import time
import datetime
from torch.utils.data import DataLoader
import yaml
import numpy as np
from PIL import Image
import math
from apex import amp
import argparse

#NVIDIA = 3   #set GPU device
torch.manual_seed(0)
torch.cuda.manual_seed(0)
#os.environ["CUDA_VISIBLE_DEVICES"]="0"


class Data(torch.utils.data.Dataset):
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
    def __init__(self, root, train=True, transform=None, get_img_id=False):
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
        self._get_img_id = get_img_id


        #   to the ImageFolder structure

        self._train_image_id=[]
        self._train_data=[]
        self._train_labels=[]
        self._test_image_id=[]
        self._test_data=[]
        self._test_labels=[]
        # load data
        if self._train:
            id_2_train = np.genfromtxt(os.path.join(self._root, 'train.list'), dtype=str)
            image_path = os.path.join(root, 'images/')
            for idx in range(id_2_train.shape[0]):
                image = Image.open(os.path.join(image_path, id_2_train[idx, 0]))
                # image = Image.open(id_2_train[idx, 0])

                label = int(id_2_train[idx, 1])  # Label starts from 0
                # pytorch only takes label as [0, num_classes) to calc loss, [1, num_classes] will get an error in loss calc

                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()

                self._train_data.append(image_np)
                self._train_labels.append(label)
                # self._train_image_id.append(idx )

                # print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_train.shape[0]), end='\r')
                print('>>>>>> Processed {} / {} images'.format(idx + 1, id_2_train.shape[0]))


            id_2_train_patch = np.genfromtxt(os.path.join(self._root, 'datalist/patchlist.list'), dtype=str)
            image_path = os.path.join(root, 'filtered_patches')
            for idx in range(id_2_train_patch.shape[0]):
                image = Image.open(os.path.join(image_path, id_2_train_patch[idx, 0]))
                #image = Image.open(id_2_train[idx, 0])

                label = int(id_2_train_patch[idx, 1])  # Label starts from 0
                # pytorch only takes label as [0, num_classes) to calc loss, [1, num_classes] will get an error in loss calc

                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()

                self._train_data.append(image_np)
                self._train_labels.append(label)
                # self._train_image_id.append(idx+ id_2_train.shape[0])

                # print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_train.shape[0]), end='\r')
                print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_train_patch.shape[0]))


        else:
            id_2_test = np.genfromtxt(os.path.join(self._root, 'test.list'), dtype=str)
            image_path = os.path.join(root, 'images/')
            for idx in range(id_2_test.shape[0]):
                image = Image.open(os.path.join(image_path, id_2_test[idx, 0]))
                #image = Image.open(id_2_test[idx, 0])
                label = int(id_2_test[idx, 1])  # Label starts from 0
                # pytorch only takes label as [0, num_classes) to calc loss, [1, num_classes] will get an error in loss calc

                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()
                self._test_data.append(image_np)
                self._test_labels.append(label)
                # self._test_image_id.append(idx)

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
            #img_id=self._train_image_id[index]
            image, label_gt =  self._train_data[index], self._train_labels[index]
        else:
            #img_id= self._test_image_id[index]
            image, label_gt = self._test_data[index], self._test_labels[index]

        if self._transform is not None:
            image = self._transform(image)
        #
        # if self._get_img_id:
        #     return img_id, image, label_gt
        # else:
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



class PatchNet(torch.nn.Module):
    """
    Per the original paper, FilterNet, which is used to remove the noisy patches and select relevant patches,
    is first pre-trained on the ImageNet and then fine-tuned on the training data

    The basis of the FilterNet is VGG-19
    The structure of FilterNet is as follows:
        conv1_1 (64) -> relu -> conv1_2 (64) -> relu -> pool1(112*112*64)
    ->  conv2_1(128) -> relu -> conv2_2(128) -> relu -> pool2(56*56*128)
    ->  conv3_1(256) -> relu -> conv3_2(256) -> relu -> conv3_3(256) -> relu -> conv3_4(256) -> relu -> pool3(28*28*256)
    ->  conv4_1(512) -> relu -> conv4_2(512) -> relu -> conv4_3(512) -> relu -> conv4_4(512) -> relu -> pool4(14*14*512)
    ->  conv5_1(512) -> relu -> conv5_2(512) -> relu -> conv5_3(512) -> relu -> conv5_4(512) -> relu -> pool5(7*7*512)
    ->  fc(4096)     -> relu -> dropout      -> fc(4096)             -> relu -> dropout
    ->  fc(200)
    """
    def __init__(self, pretrained=True, classnum = 200):
        """
        Declare all layers needed
        """
        super().__init__()
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                 m.bias = None
        self._classnum = classnum
        self._pretrained = pretrained
        basis_vgg19 = torchvision.models.vgg19_bn(pretrained=self._pretrained)
        basis_vgg19.apply(weights_init)
        self.features = basis_vgg19.features
        self.classifier = torch.nn.Sequential(*list(basis_vgg19.classifier.children())[:-1])
        self.fc = torch.nn.Linear(in_features=4096, out_features= self._classnum)

    def forward(self, x):
        """
        Forward pass of the network
        :param x:
            [torch.Tensor]  shape is N * 3 * 224 * 224
        :return:
            [torch.Tensor]  shape is N * 200
        """
        N = x.size()[0]  # N is the batch size
        assert x.size() == (N, 3, 224, 224), 'This image size should be 3 * 224 * 224 (C * H * W)'
        x = self.features(x)
        x = x.view(N, -1)
        x = self.classifier(x)
        assert x.size() == (N, 4096), 'Wrong vgg19 classifier output'
        x = self.fc(x)
        assert x.size() == (N, self._classnum), 'Wrong fc output'
        return x


class PatchNetManager(object):
    def __init__(self, options, path):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter, including base_lr, batch_size, epochs, weight_decay,
                            keys are 'base_lr', 'batch_size', 'epochs', 'weight_decay'
            path    [dict]  path of the dataset and model, keys are 'cub200' and 'model'
        """
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = options
        self._path = path
        # Network
        net = PatchNet(classnum = self._options['classnum'])
        #torch.cuda.set_device(NVIDIA)
        self._net = net.cuda()
        # if torch.cuda.device_count() > 1:
        #     self._net = torch.nn.DataParallel(net).cuda()
        # elif torch.cuda.device_count() == 1:
        #     self._net = net.cuda()
        # else:
        #     raise EnvironmentError('This is designed to run on GPU but no GPU is found')
        # Criterion
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Optimizer
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self._options['base_lr'],
                                          momentum=0.9, weight_decay=self._options['weight_decay'])
        # self._optimizer = torch.optim.Adam(self._net.parameters(), lr=self._options["base_lr"],
        #                                    weight_decay=self._options["weight_decay"])
        self._net, self._optimizer = amp.initialize(self._net, self._optimizer, opt_level="O2")

        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max', factor=0.1,
                                                                     patience=3, verbose=True, threshold=1e-4)
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.Resize(size=224),
            torchvision.transforms.RandomHorizontalFlip(),
            # torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.Resize(size=224),
            # torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # data
        train_data = Data(root=self._path['root'], train=True, transform=train_transform)
        test_data = Data(root=self._path['root'], train=False, transform=test_transform)
        self._train_loader = DataLoader(train_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(test_data, batch_size=32,
                                       shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        """
        Training the network
        """
        print("Training ... ")
        best_accuracy = 0
        best_epoch = 0
        print("Epoch\t\tTrain Loss\tTrain Accuracy\t\tTest Accuracy\tEpoch Runtime")
        for t in range(self._options["epochs"]):
            epoch_start_time = time.time()
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for img, label in self._train_loader:
                # enable training phase
                self._net.train(True)
                # put data into gpu
                img = img.cuda()
                label = label.cuda()
                # clear the existing gradient
                self._optimizer.zero_grad()
                # forward pass
                score = self._net(img)  # score's size is (N, 195)
                # calculate loss
                loss = self._criterion(score, label)
                epoch_loss.append(loss.item())
                # prediction
                max_score, prediction = torch.max(score.data, 1)
                # statistics
                num_total += label.size(0)  # y.size(0) is the batch size
                num_correct += torch.sum(prediction == label.data).item()
                # backward
                #loss.backward()
                with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                    scaled_loss.backward()
                self._optimizer.step()
            train_accuracy = 100 * num_correct / num_total
            test_accuracy = self.test(self._test_loader)
            self._scheduler.step(test_accuracy)

            epoch_end_time = time.time()
            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                best_epoch = t + 1
                print('*', end='')
                torch.save(self._net.state_dict(), os.path.join(self._path['model'], self._path['save_model']))
            print("%d\t\t%4.3f\t\t%4.2f%%\t\t\t%4.2f%%\t\t\t%4.2f" % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                                      train_accuracy, test_accuracy,
                                                                      epoch_end_time - epoch_start_time))
        print('-----------------------------------------------------------------')
        print('Best at epoch %d, test accuracy %f' % (best_epoch, best_accuracy))
        print('-----------------------------------------------------------------')

    def test(self,dataloader):
        """
        Compute the test accuracy
        :param dataloader:
            [torch.utils.data.DataLoader]   test dataloader
        :return:
            [float]                         test accuracy in percentage
        """
        # enable eval mode
        self._net.train(False)
        #dataloader = self._test_loader
        #self._net.load_state_dict(torch.load(os.path.join(self._path['model'], 'patchnet_vgg19_best_epoch.pth')))  # patchnet_vgg19_best_epoch.pth
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for img, label in dataloader:
                # Data
                img = img.cuda()
                label = label.cuda()
                # forward pass
                score = self._net(img)
                # prediction
                max_score, prediction = torch.max(score, 1)
                # statistic
                num_total += label.size(0)
                num_correct += torch.sum(prediction == label.data).item()
        # switch back to train mode
        self._net.train(True)
        #print(100 * num_correct / num_total)
        return 100 * num_correct / num_total

    def test1(self):
        """
        Compute the test accuracy
        :param dataloader:
            [torch.utils.data.DataLoader]   test dataloader
        :return:
            [float]                         test accuracy in percentage
        """
        # enable eval mode
        self._net.train(False)
        dataloader = self._test_loader
        self._net.load_state_dict(torch.load(os.path.join(self._path['model'], self._path['save_model'])))  # patchnet_vgg19_best_epoch.pth
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for img, label in dataloader:
                # Data
                img = img.cuda()
                label = label.cuda()
                # forward pass
                score = self._net(img)
                # prediction
                max_score, prediction = torch.max(score, 1)
                # statistic
                num_total += label.size(0)
                num_correct += torch.sum(prediction == label.data).item()
        # switch back to train mode
        self._net.train(False)
        print(100 * num_correct / num_total)
        return 100 * num_correct / num_total

def show_params(params, paths):
    print('|-----------------------------------------------------')
    print('| Datatime : {}'.format(datetime.datetime.now()))
    print('| Training Config : ')
    print('| base_lr: {}'.format(params['base_lr']))
    print('| weight_decay: {}'.format(params['weight_decay']))
    print('| batch_size: {}'.format(params['batch_size']))
    print('| epochs: {}'.format(params['epochs']))
    print('|-----------------------------------------------------')
    print('| Data Path & Saved Model Path')
    # print('| cub200 path: {}'.format(paths['cub200']))
    print('| model path: {}'.format(paths['model']+'/'+paths['save_model']))
    print('|-----------------------------------------------------')


def filter_net_fine_tune(dataset):
    root = os.popen('pwd').read().strip()
    root = os.path.join(root, dataset)
    config = yaml.load(open(os.path.join(root, 'config.yaml'), 'r'))
    config['weight_decay'] = float(config['weight_decay'])
    config['base_lr'] = float(config['base_lr'])
    config['classnum'] = int(config['classnum'])
    #config['batch_size'] = 4
    path = {
        # 'cub200': os.path.join(root, 'data/cub200'),
        'root': root,
        'model': os.path.join(root, 'model'),
        'save_model': 'patchnet_vgg19bn_rerun_best_epoch.pth'
    }

    # for d in path:
    #     assert os.path.isdir(path[d])

    show_params(config, path)
    manager = PatchNetManager(config, path)
    manager.train()
    #manager.test1()

if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default='CUB200')
    args = parser.parse_args()
    dataset = args.dataset
    filter_net_fine_tune(dataset)
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))
