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

import os
import torch
import torchvision
from Car_dataset import Car196
import time
from torch.utils.data import DataLoader
import yaml
import numpy as np
from PIL import Image
from apex import amp

os.environ["CUDA_VISIBLE_DEVICES"]="0"
torch.manual_seed(0)
torch.cuda.manual_seed(0)



class FilterNet(torch.nn.Module):
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
    def __init__(self, pretrained=True):
        """
        Declare all layers needed
        """
        super().__init__()
       # def weights_init(m):
       #     classname = m.__class__.__name__
       #     if classname.find('Conv') != -1:
       #         m.bias = None
        self._pretrained = pretrained
        basis_vgg19 = torchvision.models.vgg19(pretrained=self._pretrained)
       # basis_vgg19.apply(weights_init)
        self.features = basis_vgg19.features
        self.classifier = torch.nn.Sequential(*list(basis_vgg19.classifier.children())[:-1])
        self.fc = torch.nn.Linear(in_features=4096, out_features=195)

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
        assert x.size() == (N, 195), 'Wrong fc output'
        return x


class FilterNetManager(object):
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
        net = FilterNet()
        # torch.cuda.set_device(0)
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
        #train_data = Car196(root="/home/czh/Downloads/OPAM_TIP2018-master", train=True, transform=train_transform)
        #test_data = Car196(root="/home/czh/Downloads/OPAM_TIP2018-master", train=False, transform=test_transform)
        train_data = Car196(root=self._path['root'], train=True, transform=train_transform)
        test_data = Car196(root=self._path['root'], train=False, transform=test_transform)

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
        print("Epoch\tTrain Loss\tTrain Accuracy\tTest Accuracy\tEpoch Runtime")
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
                score = self._net(img)  # score's size is (N, 200)
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

    def test(self, dataloader):
        """
        Compute the test accuracy
        :param dataloader:
            [torch.utils.data.DataLoader]   test dataloader
        :return:
            [float]                         test accuracy in percentage
        """
        # enable eval mode
        self._net.train(False)
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
        return 100 * num_correct / num_total


def show_params(params, paths):
    print('|-----------------------------------------------------')
    print('| Training Config : ')
    print('| base_lr: {}'.format(params['base_lr']))
    print('| weight_decay: {}'.format(params['weight_decay']))
    print('| batch_size: {}'.format(params['batch_size']))
    print('| epochs: {}'.format(params['epochs']))
    print('|-----------------------------------------------------')
    print('| Data Path & Saved Model Path')
    # print('| cub200 path: {}'.format(paths['cub200']))
    print('| model path: {}'.format(paths['model']))
    print('|-----------------------------------------------------')


def filter_net_fine_tune():
    root = os.popen('pwd').read().strip()
    root = os.path.join(root, 'CAR196')

    config = yaml.load(open(os.path.join(root,'config.yaml'), 'r'))
    config['weight_decay'] = float(config['weight_decay'])
    config['base_lr'] = float(config['base_lr'])


    path = {
        # 'cub200': os.path.join(root, 'data/cub200'),
        'model': os.path.join(root, 'model'),
        'root': root,
        'save_model':'filternet_vgg19rerun_best_epoch.pth'
    }

    if not os.path.exists(path['model']):
        os.mkdir(path['model'])
        print(' Creating model dir ...')

    show_params(config, path)
    manager = FilterNetManager(config, path)
    manager.train()


if __name__ == '__main__':
    start = time.time()
    filter_net_fine_tune()
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))
