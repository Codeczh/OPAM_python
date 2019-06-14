import os
import torch
import torchvision
from filternet import FilterNet
from torch.utils import data
import yaml
import time
import numpy as np
from PIL import Image
import math


torch.manual_seed(0)
torch.cuda.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Patch(data.Dataset):
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

    def __init__(self, root, transform=None, train=True):
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

        #   to the ImageFolder structure
        image_path = os.path.join(root, './images/')
        train_path = os.path.join(root, './ss_patch_9process/')
        test_path = os.path.join(root, './ss_patch_9process/')
        self._train_image_id = []
        self._train_data = []
        self._train_labels = []
        self._train_patches_data = []
        self._train_patches_lable = []
        self._train_patches_id = []
        self._test_image_id = []
        self._test_data = []
        self._test_labels = []
        self._test_patches_data = []
        self._test_patches_lable = []
        self._test_patches_id = []
        # load data
        if self._train:
            id_2_train = np.genfromtxt(os.path.join(self._root, 'train.list'), dtype=str)
            for idx in range(id_2_train.shape[0] ):
                image = Image.open(os.path.join(image_path, id_2_train[idx, 0]))
                label = int(id_2_train[idx, 1])  # Label starts from 0
                name = id_2_train[idx, 0][:6]
                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()
                self._train_data.append(image_np)
                self._train_labels.append(label)
                self._train_image_id.append(name)
                path = name + ".list"
                # train_patches = np.genfromtxt(os.path.join(train_path, name), dtype=str)
                with open(os.path.join(train_path, path), 'r') as f:
                    list = f.readlines()

                for i in range(len(list)):#, min(len(list),500)):
                    list[i] = list[i].strip('\n')
                    self._train_patches_data.append(list[i])
                    self._train_patches_lable.append(label)
                    self._train_patches_id.append(name)

                # print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_train.shape[0]), end='\r')
                print('>>>>>> Processed {} / {} train images'.format(idx + 1, id_2_train.shape[0]))



        else:
            id_2_test = np.genfromtxt(os.path.join(self._root, 'test.list'), dtype=str)
            for idx in range(id_2_test.shape[0]):
                image = Image.open(os.path.join(image_path, id_2_test[idx, 0]))
                label = int(id_2_test[idx, 1])  # Label starts from 0
                name = id_2_test[idx, 0][:6]
                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()
                self._test_data.append(image_np)
                self._test_labels.append(label)
                self._test_image_id.append(name)
                path = name + ".list"
                with open(os.path.join(test_path, path), 'r') as f:
                    list = f.readlines()

                for i in range(len(list)):#,  min(len(list),500)):
                    list[i] = list[i].strip('\n')
                    self._test_patches_data.append(list[i])
                    self._test_patches_lable.append(label)
                    self._test_patches_id.append(name)

                # print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_test.shape[0]), end='\r')
                print('>>>>>> Processed {} / {} test images'.format(idx + 1, id_2_test.shape[0]))

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
            img_id, patch_lbl, patch_data = self._train_patches_id[index], self._train_patches_lable[index], \
                                            self._train_patches_data[index]
            x, y, width, height = map(int, patch_data.split())
            for idx in range(len(self._train_image_id)):
                try:
                    if img_id == self._train_image_id[idx]:
                        image = self._train_data[idx]  # (h,w,c)
                        region = image[ y:y + height,x:x + width, :]
                        if self._transform is not None:
                            region = self._transform(region)
                        break
                except BaseException:
                    raise AssertionError('image/patch id not match')

        else:
            img_id, patch_lbl, patch_data = self._test_patches_id[index], self._test_patches_lable[index], \
                                            self._test_patches_data[index]
            x, y, width, height = map(int, patch_data.split())
            for idx in range(len(self._test_image_id)):
                try:
                    if img_id == self._test_image_id[idx]:
                        image = self._test_data[idx]  # (h,w,c)
                        region = image[y:y + height,x:x + width,  :]
                        if self._transform is not None:
                            region = self._transform(region)
                        break
                except BaseException:
                    raise AssertionError('image/patch id not match')

        return img_id, region, patch_data, patch_lbl

    def __len__(self):
        """
        Length of the dataset

        Return:
            [int] Length of the dataset
        """
        if self._train:
            return len(self._train_patches_data)
        else:
            return len(self._test_patches_data)


class PatchFilter(object):
    def __init__(self, path):
        super().__init__()
        # self._regions_data = []
        # self._regions_label = []
        # self._regions_origin_img_id = []
        self._path = path
        # Net
        net = FilterNet(pretrained=False)
        self._net = net.cuda()
        self._net.load_state_dict(torch.load(self._path['load_model'],map_location={'cuda:2':'cuda:0'}))  # filternet_vgg19_best_epoch.pth
        #, map_location={'cuda:2': 'cuda:0'}
    def patch_filter(self, threshold=0.8, phase='train'):
        """
        filter irrelevant patches based on the threshold
        :param threshold:
            used to filter patches, default : 0.8
        :param phase:
            running phase, can be either 'train' or 'test'
        """
        region_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])

        self._net.eval()

        if not os.path.exists(os.path.join(path['root'], 'filtered_patches')):
            os.mkdir(os.path.join(path['root'], 'filtered_patches'))
            print(' Creating filtered_patches dir ...')
        if not os.path.exists(os.path.join(path['root'], 'rect_list')):
            os.mkdir(os.path.join(path['root'], 'rect_list'))
            print(' Creating rect_list dir ...')
        if not os.path.exists(os.path.join(path['root'], 'datalist')):
            os.mkdir(os.path.join(path['root'], 'datalist'))
            print(' Creating datalist dir ...')

        total = 0
        print('Start processing {} Patches ...'.format(phase))
        if phase == 'train':
            patch_dataset = Patch(root=self._path['root'], transform=region_transform,
                                  train=True)
        else:
            patch_dataset = Patch(root=self._path['root'], transform=region_transform,
                                  train=False)
        patch_dataloader = data.DataLoader(patch_dataset, batch_size=256, shuffle=False, num_workers=4,
                                           pin_memory=True)
        with torch.no_grad():
            count = 0
            id_mark = " "
            list = []
            for img_id, region, rect, label in patch_dataloader:
                N = len(region)
                region = region.cuda()
                netOutput = self._net(region)
                for i in range(N):
                    s = "%06d" % int(img_id[i])
                    lbl = int(label[i])
                    listrect = []
                    res = netOutput[i].argmax()
                    if (lbl == res):
                        if id_mark != img_id[i]:
                            count = 0
                            id_mark = img_id[i]
                        else:
                            count += 1
                        list.append('{}_{}_filtered_patches.jpg'.format(s, count) + " " + str(lbl) + '\n')
                        listrect.append(rect[i] + '\n')
                        # 生成图片
                        x, y, width, height = map(int, rect[i].split())
                        # img = np.array(image[i])
                        # img = img[x:x + width, y:y + height, :]
                        img = torchvision.transforms.ToPILImage()(region[i].cpu())
                        #img = torchvision.transforms.Resize((height, width))(img)
                        img.save(self._path['root']+'/filtered_patches/{}_{}_filtered_patches.jpg'.format(s, count))
                        img.close()
                with open(self._path['root']+'/rect_list/{}.list'.format(s), 'a') as f:
                    f.writelines(listrect)
                total += N
                print('\tAlready processed {} patches: {} / {} ,img_id:{}'.format(phase, total,len(patch_dataset),img_id[0]))

            with open(self._path['root']+'/datalist/patchlist.list', 'w') as f:
                f.writelines(list)


if __name__ == '__main__':
    root = os.popen('pwd').read().strip()
    root = os.path.join(root, 'CAR196')
    config = yaml.load(open(os.path.join(root, 'config.yaml'), 'r'))
    path = {
        # 'cub200': os.path.join(root, 'data/cub200'),
        'root': root,
        'load_model': os.path.join(root, 'model','filternet_vgg19rerun_best_epoch.pth')
    }
    for k in path:
        if k is 'load_model':
            assert os.path.isfile(path[k])
        else:
            assert os.path.isdir(path[k])

    start = time.time()
    patchFilter = PatchFilter(path)
    patchFilter.patch_filter()
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))
