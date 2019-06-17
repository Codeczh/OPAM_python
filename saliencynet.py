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
import torchvision
from Car_dataset import Car196
from torch.utils.data import DataLoader
#from filternet_bn_rerun import FilterNet
import yaml
import torch.nn.functional as F
from apex import amp


torch.manual_seed(0)
torch.cuda.manual_seed(0)

#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class SaliencyNet(torch.nn.Module):
    """
    SaliencyNet
    The structure of SaliencyNet is as follows:
        conv1_1 (64) -> relu -> conv1_2 (64) -> relu -> pool1 ->
        conv2_1(128) -> relu -> conv2_2(128) -> relu -> pool2 ->
        conv3_1(256) -> relu -> conv3_2(256) -> relu -> conv3_3(256) -> relu -> conv3_4(256) -> relu -> pool3 ->
        conv4_1(512) -> relu -> conv4_2(512) -> relu -> conv4_3(512) -> relu -> conv4_4(512) -> relu -> pool4 ->
        conv5_1(512) -> relu -> conv5_2(512) -> relu -> conv5_3(512) ->
        conv(size=(3,3), stride=1, pad=1, output=1024) -> relu -> global average pooling -> dropout -> 200 way fc
    The network input image of size (3 * 224 * 224)
    """
    def __init__(self, pretrained=True):
        """
        Declare all layers needed
        """
        super().__init__()
        self._pretrained = pretrained
        self.features = torchvision.models.vgg16(pretrained=self._pretrained).features
        # Remove the layers after conv5_3 (including a relu and a pool5), output of self.features will be 14 * 14
        self.features = torch.nn.Sequential(*list(self.features.children())[:-4])
        # the output of following conv layer is (N, 1024, 14, 14)
        self.conv = torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = torch.nn.ReLU(inplace=True)
        self.gap = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.fc = torch.nn.Linear(in_features=1024, out_features=195)
        # self.fc=F.softmax()

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
        assert x.size() == (N, 512, 14, 14), 'Wrong vgg19 feature extractor output'
        x = self.conv(x)
        x = self.relu(x)
        assert x.size() == (N, 1024, 14, 14), 'Wrong conv output'
        x = self.gap(x).view(N, -1)
        assert x.size() == (N, 1024), 'Wrong global average pooling output'
        x = self.dropout(x)
        x = self.fc(x)
        assert x.size() == (N, 195), 'Wrong fc output'
        return x


class SaliencyNetManager(object):
    def __init__(self, params, paths):
        """
        Prepare the network, criterion, Optimizer and data
        Arguments:
            options [dict]  Hyperparameter, including base_lr, batch_size, epochs, weight_decay,
                            keys are 'base_lr', 'batch_size', 'epochs', 'weight_decay'
            path    [dict]  path of the dataset and model, keys are 'cub200' and 'model'
        """
        print('------------------------------------------------------------------------------')
        print('Preparing the network and data ... ')
        self._options = params
        self._path = paths
        # Network
        # if self._path['model'] is None:
        #     net = SaliencyNet()
        # else:
        #     net = SaliencyNet(pretrained=False)
        net = SaliencyNet()
        #torch.cuda.set_device(0)
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
        train_data = Car196(root=self._path['root'], train=True, transform=train_transform)
        test_data = Car196(root=self._path['root'], train=False, transform=test_transform)
        self._train_loader = DataLoader(train_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = DataLoader(test_data, batch_size=16,
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
                torch.save(self._net.state_dict(), os.path.join(self._path['model'],
                                                                'saliencynet_vgg16_best_epoch.pth'))
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


def saliency_net_fine_tune():
    root = os.popen('pwd').read().strip()
    root = os.path.join(root, 'CUB200')
    config = yaml.load(open(os.path.join(root, 'config.yaml'), 'r'))
    config['weight_decay'] = float(config['weight_decay'])
    config['base_lr'] = 0.1*float(config['base_lr'])

    path = {
        # 'cub200': os.path.join(root, 'data/cub200'),
        'model': os.path.join(root, 'model'),
        'root': root
    }


    for d in path:
        assert os.path.isdir(path[d])

    show_params(config, path)
    finetune_manager = SaliencyNetManager(config, path)
    finetune_manager.train()


if __name__ == '__main__':
    start = time.time()
    saliency_net_fine_tune()
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))
