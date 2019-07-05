# Train----use image trainset , object trainset
# Test ----  object testset,  image testset
# based on patchnet.model
import os
import torch
import torchvision
import time
import datetime
from torch.utils.data import DataLoader
from patchnet_bn_rerun import PatchNet
import yaml
import numpy as np
from PIL import Image
import math
#from apex import amp
import argparse

#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.manual_seed(0)
torch.cuda.manual_seed(0)



class Data_part(torch.utils.data.Dataset):
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
            train_part_path = os.path.join(self._root,'parts_align_9')

            #bbox
            img_path = os.path.join(self._root, 'bbox')
            id_2_train = np.genfromtxt(os.path.join(self._root, 'train.list'), dtype=str)
            for idx in range(id_2_train.shape[0]):
                image = Image.open(os.path.join(img_path, id_2_train[idx, 0][0:6] + '.png'))
                label = int(id_2_train[idx, 1])
                #  ./bbox/000001_bbox.jpg 0
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
            test_part_path = os.path.join(self._root, 'parts_align_9')
            if self._test=='image':
                for idx in range(id_2_test.shape[0]):
                    image = Image.open(os.path.join(self._root, 'images', id_2_test[idx, 0]))
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
                id_2_test = np.genfromtxt(os.path.join(self._root, 'test.list'), dtype=str)
                for idx in range(id_2_test.shape[0]):
                    image = Image.open(os.path.join(self._root, 'bbox',id_2_test[idx, 0][0:6]+'.png'))
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



class Partnet(object):
    def __init__(self, options, path):
        '''

        :param options:
        :param path:
        '''
        print('-----------------------------------------------------------------')
        print('Preparing the network and data ...')
        self._options = options
        self._path = path
        # Network
        net = PatchNet(classnum = self._options['classnum'])
        # torch.cuda.set_device(NVIDIA)
        # net.load_state_dict(torch.load(os.path.join(self._path['model'], 'patchnet_vgg19_best_epoch.pth'),map_location={'cuda:2':'cuda:0'}))
        net.load_state_dict(torch.load(os.path.join(self._path['model'], self._path['load_model']),
                                       map_location={'cuda:3': 'cuda:0'}))

        self._net = net.cuda()
        # Criterion
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Optimizer
        self._optimizer = torch.optim.SGD(self._net.parameters(), lr=self._options['base_lr'],
                                          momentum=0.9, weight_decay=self._options['weight_decay'])
        #self._net, self._optimizer = amp.initialize(self._net, self._optimizer, opt_level="O2")

        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self._optimizer, mode='max',
                                                                     factor=0.1, patience=3, verbose=True,
                                                                     threshold=1e-4)
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.RandomCrop(size=224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # data
        train_data = Data_part(root=self._path['root'], train=True, transform=train_transform)
        test_image_data = Data_part(root=self._path['root'], train=False, test='image', transform=test_transform)
        test_bbox_data = Data_part(root=self._path['root'], train=False, test='bbox', transform=test_transform)
        test_part1_data = Data_part(root=self._path['root'], train=False, test='part1', transform=test_transform)
        test_part2_data = Data_part(root=self._path['root'], train=False, test='part2', transform=test_transform)
        self._train_loader = DataLoader(train_data, batch_size=self._options['batch_size'],
                                        shuffle=True, num_workers=4, pin_memory=True)
        self._test_image_loader = DataLoader(test_image_data, batch_size=self._options['batch_size'],
                                       shuffle=False, num_workers=4, pin_memory=True)
        self._test_bbox_loader = DataLoader(test_bbox_data, batch_size=self._options['batch_size'],
                                             shuffle=False, num_workers=4, pin_memory=True)
        self._test_part1_loader = DataLoader(test_part1_data, batch_size=self._options['batch_size'],
                                            shuffle=False, num_workers=4, pin_memory=True)
        self._test_part2_loader = DataLoader(test_part2_data, batch_size=self._options['batch_size'],
                                             shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        '''
        Training the network
        '''
        print("Training ...")
        best_accuracy = 0
        best_epoch = 0
        print("Epoch\t\tTrain Loss\tTrain Accuracy\t\tTest Accuracy\tTest Image Accuracy\tEpoch Runtime")
        for t in range(self._options['epochs']):
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
                score = self._net(img)  # score's size is (N,195)
                # calculate loss
                loss = self._criterion(score, label)
                epoch_loss.append(loss.item())
                # prediction
                max_score, prediction = torch.max(score.data, 1)
                #  statistics
                num_total += label.size(0)  # y.size(0) is the batch size
                num_correct += torch.sum(prediction == label.data).item()
                # backward()
                loss.backward()
                #with amp.scale_loss(loss, self._optimizer) as scaled_loss:
                #    scaled_loss.backward()
                self._optimizer.step()
            train_accuracy = 100 * num_correct / num_total
            test_image_accuracy = self.test(self._test_image_loader)
            test_bbox_accuracy = self.test(self._test_bbox_loader)
            test_part1_accuracy = self.test(self._test_part1_loader)
            test_part2_accuracy = self.test(self._test_part2_loader)
            self._scheduler.step(test_image_accuracy)

            epoch_end_time = time.time()
            if test_image_accuracy > best_accuracy:
                best_accuracy = test_image_accuracy
                best_epoch = t + 1
                print('*', end='')
                torch.save(self._net.state_dict(), os.path.join(self._path['model'], self._path['save_model']))
            print("%d\t\t%4.3f\t\t%4.2f\t\t\t%4.2f\t\t\t%4.2f\t\t\t%4.2f\t\t\t%4.2f\t\t\t%4.2f" % (t + 1, sum(epoch_loss) / len(epoch_loss),
                                                                             train_accuracy, test_image_accuracy,
                                                                             test_bbox_accuracy,test_part1_accuracy,
                                                                             test_part2_accuracy,
                                                                             epoch_end_time - epoch_start_time))
        print('-----------------------------------------------------------------')
        print('Best at epoch %d, test accuracy %f' % (best_epoch, best_accuracy))
        print('-----------------------------------------------------------------')

    def test(self, dataloader):
        '''
        Compute the test accuracy
            :param dataloader:
            [torch.utils.data.DataLoader]   test dataloader
            :return:
            [float]                         test accuracy in percentage
        '''
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
                # predicion
                max_score, prediction = torch.max(score, 1)
                # statistic
                num_total += label.size(0)
                num_correct += torch.sum(prediction == label.data).item()
            # switch back to train mode
        self._net.train(True)
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
    config['batch_size'] = 64

    path = {
        # 'cub200': os.path.join(root, 'data/cub200'),
        'root': root,
        'model': os.path.join(root, 'model'),
        'load_model': 'patchnet_vgg19bn_rerun_best_epoch.pth',
        'save_model': 'partnet_vgg19bn_rerun_best_epoch.pth'
    }

    # for d in path:
    #     assert os.path.isdir(path[d])

    show_params(config, path)
    manager = Partnet(config, path)
    manager.train()


if __name__ == '__main__':
    start = time.time()
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default='CUB200')
    args = parser.parse_args()
    dataset = args.dataset
    filter_net_fine_tune(dataset)
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))
