import os
import torch
import torchvision
import cv2
import pickle
import time
import datetime
import yaml
import numpy as np
from PIL import Image
from sklearn.cluster import SpectralClustering
from sklearn.cluster import KMeans
# from sklearn.cluster import SpectralClustering
from patchnet_bn_rerun import PatchNet
from torch.utils.data import Dataset, DataLoader
import argparse


torch.manual_seed(0)
torch.cuda.manual_seed(0)
#os.environ["CUDA_VISIBLE_DEVICES"]="0"

class Part(Dataset):
    def __init__(self, data_root, transform = None,train=True):
        super().__init__()
        self._root = os.path.expanduser(data_root)
        self._train = train
        self._transform = transform
        image_path = os.path.join(self._root,'parts_9')

        self._train_image_id = []
        self._train_image_size = []
        self._train_data = []
        self._train_labels = []
        self._test_image_id = []
        self._test_data = []
        self._test_labels = []
        self._test_image_size = []
        if self._train is True:

            id_2_train = np.genfromtxt(os.path.join(self._root, 'train.list'), dtype=str)

            for idx in range(id_2_train.shape[0] ):
                # fp = open(os.path.join(image_path, id_2_train[idx, 0]),'rb')
                # image = Image.open(fp)
                image = Image.open(os.path.join(image_path, id_2_train[idx, 0][0:6]+'_1'+id_2_train[idx, 0][6:]))
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
                self._train_image_id.append(id_2_train[idx, 0][0:6])
                self._train_image_size.append(size)

                image = Image.open(os.path.join(image_path, id_2_train[idx, 0][0:6] + '_2' + id_2_train[idx, 0][6:]))
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
                self._train_image_id.append(id_2_train[idx, 0][0:6])
                self._train_image_size.append(size)

                # print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_train.shape[0]), end='\r')
                print('>>>>>> Processed {} / {} images'.format(idx + 1, id_2_train.shape[0]))

        else:
            id_2_test = np.genfromtxt(os.path.join(self._root, 'test.list'), dtype=str)

            for idx in range(id_2_test.shape[0] ):
                # fp = open(os.path.join(image_path, id_2_train[idx, 0]),'rb')
                # image = Image.open(fp)
                image = Image.open(os.path.join(image_path, id_2_test[idx, 0][0:6] + '_1' + id_2_test[idx, 0][6:]))
                size = image.size
                label = int(id_2_test[idx, 1])  # Label starts from 0

                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()
                # fp.close()

                self._test_data.append(image_np)
                self._test_labels.append(label)
                self._test_image_id.append(id_2_test[idx, 0][0:6])
                self._test_image_size.append(size)

                image = Image.open(os.path.join(image_path, id_2_test[idx, 0][0:6] + '_2' + id_2_test[idx, 0][6:]))
                size = image.size
                label = int(id_2_test[idx, 1])  # Label starts from 0

                # convert gray scale image to RGB image
                if image.mode == 'L':
                    image = image.convert('RGB')
                image_np = np.array(image)
                image.close()
                # fp.close()

                self._test_data.append(image_np)
                self._test_labels.append(label)
                self._test_image_id.append(id_2_test[idx, 0][0:6])
                self._test_image_size.append(size)

                # print('>>>>>> Processed {} / {} images'.format(idx+1, id_2_train.shape[0]), end='\r')
                print('>>>>>> Processed {} / {} images'.format(idx + 1, id_2_test.shape[0]))



    def __getitem__(self, index):
        # imgID, partRect, label = self._partImageList[index]
        # x, y, width, height = partRect
        if self._train:
            img_id, image, label_gt = self._train_image_id[index], self._train_data[index], self._train_labels[index]
        else:
            img_id, image, label_gt = self._test_image_id[index], self._test_data[index], self._test_labels[index]

        if self._transform is not None:
            image = self._transform(image)


        return img_id, image, label_gt


    def __len__(self):
        if self._train == True:
            return len(self._train_labels)
        else:
            return len(self._test_labels)




class PartAligner(object):
    def __init__(self, options ,path):
        super().__init__()
        self._options = options
        self._path = path
        net = PatchNet(classnum = self._options['classnum'])
        net = net.cuda()
        net.load_state_dict(torch.load(os.path.join(self._path['model'],self._path['load_model'])))
        #  ,map_location={'cuda:0': 'cuda:0'}
        self._net = net
        self.conv_feature_blobs = []

        self._partpath= os.path.join(self._path['root'], 'parts_align_9')
        if not os.path.exists(self._partpath):
            os.mkdir(self._partpath)
            print('Creating {}'.format(self._partpath))
        # self._data_path =  os.path.join(os.path.expanduser(data_root),'parts')
        # print(net.features[46].weight.shape)
        # print(net.features[46].weight)
        Neuron = net.features[46].weight.view(512, 4608)
        Neuron = Neuron.data.cpu().numpy()
        # print(Neuron)
        # print(type(net.features[46].weight))
        #print(net.features[46].bias)
        #print(net)
        Neuron_dir = os.path.join(self._path['root'],'Neuron_9')
        if not os.path.exists(Neuron_dir):
            os.mkdir(Neuron_dir)
            print('Creating {}'.format(Neuron_dir))
        self._splitlist = self.neuron_spectral_cluster_kmeans(Neuron)
        # print(self._splitlist)
        # print((sum(self._splitlist)))
        # print(type(self._splitlist))
        np.savetxt(os.path.join(Neuron_dir,'sck_neuron_0.txt'),(np.where(self._splitlist==0)),fmt='%d')
        np.savetxt(os.path.join(Neuron_dir,'sck_neuron_1.txt'), (np.where(self._splitlist==1)),fmt='%d')

        self._directlist = self.neuron_spectral_cluster_direct(Neuron)
        # print(self._directlist)
        # print((sum(self._directlist)))
        # print(type(self._directlist))
        np.savetxt(os.path.join(Neuron_dir,'dir_neuron_0.txt'), (np.where(self._directlist==0)),fmt='%d')
        np.savetxt(os.path.join(Neuron_dir,'dir_neuron_1.txt'), (np.where(self._directlist==1)),fmt='%d')
        # self._net = net.cuda()
        # model_path = './model/patchnet_vgg19_best_epoch.pth'
        # self._net.load_state_dict(torch.load(model_path,map_location={'cuda:2':'cuda:0'}))
        # for key,value in enumerate(self._net):
        #     print(key,value)
        # print(net)
        # self._Npart0 = np.where(self._splitlist == 0)
        # self._Npart1 = np.where(self._splitlist == 1)

    def align_part(self):
        part_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((16, 16)),
            torchvision.transforms.ToTensor()
        ])
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.Resize((224, 224)),
            # torchvision.transforms.Resize(size=224),
            #torchvision.transforms.RandomHorizontalFlip(),
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
        phases = [ 'train','test']#
        for phase in phases:
            if phase == 'train':
                part_dataset = Part(data_root=self._path['root'], transform=train_transform, train=True)
            else:
                part_dataset = Part(data_root=self._path['root'], transform=test_transform, train=False)
                print(part_dataset.__len__())
            part_dataloader = DataLoader(part_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
            self._part0 = {}
            self._part1 = {}
            print("In {}_set, start processing ... ".format(phase))
            print(len(part_dataloader))
            # if isinstance(self._net, torch.nn.DataParallel):
            #     self._net = self._net.module
            self._net.eval()
            with torch.no_grad():
                j = 0
                self._featureact = []
                for imgID, image, label in part_dataloader:
                    # Batch Size N
                    N = label.shape[0]
                    # Data
                    image = image.cuda()
                    self.conv_feature_blobs = []
                    handler = self._net.features[-7].register_forward_hook(self.hook_feature)
                    _ = self._net.features[:-6](image)
                    feature = torch.nn.AvgPool2d((14, 14))(self.conv_feature_blobs[0]).squeeze()
                    #print(feature.shape)
                    # print(feature.squeeze().shape)
                    # for i in range(N):
                    #     self._featureact.append(feature[i])
                    handler.remove()

                    for i in range(0, N, 2):
                        assert imgID[i] == imgID[i + 1]
                        img0 = Image.open(os.path.join(self._path['root'],'parts_9', imgID[i] + '_1.jpg'))
                        img1 = Image.open(os.path.join(self._path['root'],'parts_9', imgID[i] + '_2.jpg'))
                        score_f0_n0, score_f0_n1, score_f1_n0, score_f1_n1 = 0, 0, 0, 0
                        part_0_activation = feature[i]  # .cpu().numpy()  # shape of (512, 1, 1)
                        part_1_activation = feature[i + 1]
                        # print(type(part_1_activation))
                        ##
                        cluster_index = self._splitlist  # list, len = 512
                        # print(type(cluster_index))
                        # cluster_index = torch.from_numpy(cluster_index).cuda()
                        for idx in range(len(cluster_index)):
                            if cluster_index[idx] == 0:
                                score_f0_n0 += part_0_activation[idx]
                                score_f1_n0 += part_1_activation[idx]
                            else:
                                score_f0_n1 += part_0_activation[idx]
                                score_f1_n1 += part_1_activation[idx]
                        if score_f0_n0 >= score_f1_n0 and score_f0_n1 < score_f1_n1:
                            img0.save(os.path.join(self._partpath, imgID[i] + '_1.jpg'))
                            img1.save(os.path.join(self._partpath, imgID[i] + '_2.jpg'))
                        elif score_f0_n0 < score_f1_n0 and score_f0_n1 >= score_f1_n1:
                            img1.save(os.path.join(self._partpath, imgID[i] + '_1.jpg'))
                            img0.save(os.path.join(self._partpath, imgID[i] + '_2.jpg'))
                        elif score_f0_n0 >= score_f1_n0 and score_f0_n1 >= score_f1_n1:
                            img0.save(os.path.join(self._partpath, imgID[i] + '_1.jpg'))
                            img0.save(os.path.join(self._partpath, imgID[i] + '_2.jpg'))
                        else:#score_f0_n0 < score_f1_n0 and score_f0_n1 < score_f1_n1:
                            img1.save(os.path.join(self._partpath, imgID[i] + '_1.jpg'))
                            img1.save(os.path.join(self._partpath, imgID[i] + '_2.jpg'))
                        # remove the handler
                        # handler.remove()
                        img0.close()
                        img1.close()
                        j += 1
                        print('\tAlready processed: {} / {} ---ImgID {}'.format(j, part_dataset.__len__(),imgID[i]))

    def hook_feature(self, module, input, output):
        self.conv_feature_blobs.append(output.data.cpu())  # output : (N, 512, 14, 14), numpy.ndarray





    @staticmethod
    def neuron_spectral_cluster_kmeans(neuron):
        """
        Perform spectral cluster over neurons in the penultimate (conv5_3) layer
        :param neuron:
            [numpy.ndarray] The activation of neurons in the penultimate (conv5_3) layer, shape is (512, 1, 1)
        :return:
            [list] a list, whose element denotes the cluster number of its corresponding neuron, length is 512
        """
        # compute the cosine similarity matrix by
        #           cosine = <A, B> / (|A|*|B|)
        # ui = neuron.squeeze(2)
        # print(ui.shape)
        # uj = torch.t(ui)
        # cosine_similarity_matrix = torch.matmul(ui, uj) / (torch.norm(ui) * torch.norm(uj))  # shape -> (512, 512)

        ui = neuron
        uj = np.transpose(ui)  # shape -> (4608, 512)
        cosine_similarity_matrix = np.matmul(ui, uj) / (np.matmul(np.linalg.norm(ui,2,axis=1), np.linalg.norm(uj,2,axis=0)))  # shape -> (512, 512)
        W = cosine_similarity_matrix + np.transpose(cosine_similarity_matrix)
        A = sum(cosine_similarity_matrix,0)
        D = np.diag(A)
        L = D - W
        d,v = np.linalg.eig(L)
        F = v[:,1:3]
        F = F / np.repeat(np.linalg.norm(F,2,axis=1)[:,np.newaxis],repeats=2,axis=1)
        sc = KMeans(n_clusters=2)  # 构造聚类器
        sc.fit(F)  # 聚类
        cluster_index = sc.labels_  # 获取聚类标签

        #cosine_similarity_matrix = np.exp(cosine_similarity_matrix)

        # Perform spectral clustering on the similarity matrix
        # sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100)
        # sc.fit(cosine_similarity_matrix)
        # cluster_index = list(sc.labels_)
        # print(cluster_index)
        assert len(cluster_index) == 512
        return cluster_index

    @staticmethod
    def neuron_spectral_cluster_direct(neuron):
        """
        Perform spectral cluster over neurons in the penultimate (conv5_3) layer
        :param neuron:
            [numpy.ndarray] The activation of neurons in the penultimate (conv5_3) layer, shape is (512, 1, 1)
        :return:
            [list] a list, whose element denotes the cluster number of its corresponding neuron, length is 512
        """
        # compute the cosine similarity matrix by
        #           cosine = <A, B> / (|A|*|B|)
        # ui = neuron.squeeze(2)
        # print(ui.shape)
        # uj = torch.t(ui)
        # cosine_similarity_matrix = torch.matmul(ui, uj) / (torch.norm(ui) * torch.norm(uj))  # shape -> (512, 512)

        ui = np.squeeze(neuron)  # shape -> (512, 1)
        uj = np.transpose(ui)  # shape -> (1, 512)
        #cosine_similarity_matrix = np.matmul(ui, uj) / (np.linalg.norm(ui) * np.linalg.norm(uj))  # shape -> (512, 512)
        cosine_similarity_matrix = np.matmul(ui, uj) / (
            np.matmul(np.linalg.norm(ui, 2, axis=1), np.linalg.norm(uj, 2, axis=0)))  # shape -> (512, 512)

        cosine_similarity_matrix = np.exp(cosine_similarity_matrix)
        # Perform spectral clustering on the similarity matrix
        sc = SpectralClustering(n_clusters=2, affinity='precomputed', n_init=100)
        sc.fit(cosine_similarity_matrix)
        #cluster_index = list(sc.labels_)
        cluster_index = sc.labels_
        # print(cluster_index)
        assert len(cluster_index) == 512, 'error'
        return cluster_index


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default='CUB200')
    args = parser.parse_args()
    dataset = args.dataset

    root = os.popen('pwd').read().strip()
    root = os.path.join(root,dataset)
    config = yaml.load(open(os.path.join(root,'config.yaml'), 'r'))
    config['classnum'] = int (config['classnum'])
    path = {
        #'cub200': os.path.join(root, 'data/cub200'),
        'root':root,
        'model': os.path.join(root, 'model'),
        'load_model': 'patchnet_vgg19bn_rerun_best_epoch.pth'
    }
   
    print(' Datetime : {}'.format(datetime.datetime.now()))
    print('>>>--->>>\nUsing model:\n\t{} \n>>>--->>>'.format(path['load_model']))
    start = time.time()
    part_aligner = PartAligner(config,path)
    part_aligner.align_part()
    end = time.time()
    print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))

