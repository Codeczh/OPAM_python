import os
from PIL import Image
import yaml
import numpy as np
import cv2
import time
import selectivesearch
import math
import multiprocessing
import argparse


def patch_generator(image):
    """
    Using Selective Search to generate region proposals
    :param image:
        [numpy.ndarray]         original image to generate region proposal from (H, W, C format; RGB)
                                note that, opencv process image in H,W,C format and the channels are BGR
    :return:
        [list(numpy.ndarray)]   list of proposed regions
    """
    # images stored in pickle is in H, W, C format and channels are RGB
    # convert hwc(rgb) image to opencv convention: hwc(bgr)
    img_cv = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    min_size = math.floor(min(img_cv.shape[0],img_cv.shape[1])/10)
    _, regions1 = selectivesearch.selective_search(img_cv, scale=50, sigma=0.8, min_size=min_size)# 1 0.8 20
    _, regions2 = selectivesearch.selective_search(img_cv, scale=100, sigma=0.8, min_size=min_size)
    _, regions3 = selectivesearch.selective_search(img_cv, scale=150, sigma=0.8, min_size=min_size)
    _, regions4 = selectivesearch.selective_search(img_cv, scale=300, sigma=0.8, min_size=min_size)
    regions = regions1 + regions2 + regions3 + regions4
    patches = []
    #min_size=20
    for region in regions:
        x,y,w,h = np.array(region['rect'])
        if w>=min_size and h >=min_size:
            patches.append((x,y,w,h))  # region['rect'] : tuple (x, y, w, h)
            #cv2.rectangle(img_cv, (x, y), (x + h, y + w), (0, 0,255), 1)

    patches=(set([tuple(t) for t in patches]))
    final_pathes=[]
    for patche in patches:
        x, y, w, h = patche
        final_pathes.append(str(x)+' ' + str(y)+' ' + str(w)+' ' + str(h)+' ' + '\n')
    #print(len(final_pathes))
    #cv2.imshow("img", img_cv)
    #cv2.waitKey(0)
    return final_pathes

def region_proposal(path,id):
    """
    Propose regions using selective search on each image in training/testing dataset
    """
    # propose regions on training dataset
    if not os.path.exists(os.path.join(path['root'],'ss_patch_9process')):
        os.mkdir(os.path.join(path['root'],'ss_patch_9process'))
        print(' Creating ss_patch_9process dir ...')
    image_path = os.path.join(path['root'], 'images/')
    train_num = 0
    id_2_train = np.genfromtxt(os.path.join(path['root'], 'train.list'), dtype=str)
    step = math.floor((id_2_train.shape[0]-1)/8)
    for idx in range(step*id,min(step*(id+1), id_2_train.shape[0])) :
        image = Image.open(os.path.join(image_path, id_2_train[idx, 0]))
        if image.mode == 'L':
            image = image.convert('RGB')
        image_np = np.array(image)
        image.close()
        regionList = patch_generator(image_np)

        with open(path['root']+'/ss_patch_9process/' + id_2_train[idx, 0][0:6] +'.list', 'w') as f:
            f.writelines(regionList)
        print('>>>>>>Processed train image {}/{} selective search >>>length {}       '.\
              format(id_2_train[idx, 0][0:6],id_2_train.shape[0],len(regionList)))
        train_num +=len(regionList)

    # image = Image.open(os.path.join(image_path, id_2_train[0, 0]))
    # if image.mode == 'L':
    #     image = image.convert('RGB')
    # image_np = np.array(image)
    # image.close()
    # regionList = patch_generator(image_np)
    # with open('/media/zcy/zcy_data/Cars-196/patches', 'w') as f:
    #     f.writelines(regionList)
    test_num = 0
    id_2_test = np.genfromtxt(os.path.join(path['root'], 'test.list'), dtype=str)
    step = math.floor((id_2_test.shape[0]-1)/8)
    for idx in range(step*id,min(step*(id+1), id_2_test.shape[0])):
        image = Image.open(os.path.join(image_path, id_2_test[idx, 0]))
        if image.mode == 'L':
            image = image.convert('RGB')
        image_np = np.array(image)
        image.close()
        regionList = patch_generator(image_np)

        with open(path['root'] + '/ss_patch_9process/' + id_2_test[idx, 0][0:6] + '.list', 'w') as f:
            f.writelines(regionList)
        print('>>>>>>Processed test image {}/{} selective search >>>length {}       '. \
              format(id_2_test[idx, 0][0:6], id_2_test.shape[0], len(regionList)))
        test_num += len(regionList)
    print('-------------train number:{}   test number:{}--------------'.format(train_num,test_num))
def thread(id,dataset):
    print('start {}-----'.format(id))
    root = os.popen('pwd').read().strip()
    root = os.path.join(root, dataset)
    config = yaml.load(open(root + '/config.yaml', 'r'))
    path = {
        'root': root
    }
    for k in path:
        if k is 'model':
            assert os.path.isfile(path[k])
        else:
            assert os.path.isdir(path[k])

    start = time.time()
    region_proposal(path,id)
    end = time.time()
    print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~{} Runtime: {}~~~~~~~~~~~~~~~~~~~~~~~~~~'.format(id,end - start))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--dataset', type=str, default='CUB200')
    args = parser.parse_args()
    dataset = args.dataset
    p0 = multiprocessing.Process(target=thread, args=(0,dataset))
    p1 = multiprocessing.Process(target=thread, args=(1,dataset))
    p2 = multiprocessing.Process(target=thread, args=(2,dataset))
    p3 = multiprocessing.Process(target=thread, args=(3,dataset))
    p4 = multiprocessing.Process(target=thread, args=(4,dataset))
    p5 = multiprocessing.Process(target=thread, args=(5,dataset))
    p6 = multiprocessing.Process(target=thread, args=(6,dataset))
    p7 = multiprocessing.Process(target=thread, args=(7,dataset))
    p8 = multiprocessing.Process(target=thread, args=(8,dataset))
    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()
    print("The number of CPU is:" + str(multiprocessing.cpu_count()))
    for p in multiprocessing.active_children():
        print("child p.name:=%s" % p.name + "\tp.id=%s" % str(p.pid))
        #print(p.pid)
        #print("END-----")

    print('Waiting for all subprocesses done...')
