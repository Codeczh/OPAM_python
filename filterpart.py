import os
import torch
import torchvision
import time
from torch.utils.data import DataLoader
import yaml
import numpy as np
from PIL import Image
import math
import random
import argparse

from apex import amp

torch.manual_seed(0)
torch.cuda.manual_seed(0)
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def boxoverlap(a,b): #IOU
    #a=[m,4] b=[1,4]
    o = np.zeros((len(a),1))
    for i in range(len(a)):
        x1 = max(a[i][0], b[0])
        y1 = max(a[i][1], b[1])
        x2 = min(a[i][2], b[2])
        y2 = min(a[i][3], b[3])

        w = x2-x1+1
        h = y2-y1+1
        inter = w * h
        aarea = (a[i][2]-a[i][0]+1)*(a[i][3]-a[i][1]+1)
        iou = inter/aarea
        if w<=0 or h <=0 :
            iou = 0
        o[i]=iou
    return o
def cal_area(a):
    w = a[2]-a[0]+1
    h = a[3]-a[1]+1
    s = w*h
    return s


def filtering_2_TIP(boxes, gt, imgsize, saliency, beta):
    ft_boxes = []
    truth =0
    parts = np.zeros((2,4))
    overlap = boxoverlap(boxes,gt)
    overlap = overlap.T
    overlap = overlap[0]
    # if imgsize.min() > 224:
    #     scale_map = 224 / imgsize
    # else:
    #     scale_map = imgsize.min() / imgsize
    check = 0
    #saliency = float(saliency)
    gt_area = cal_area(gt)

    ft_boxes = []
    for i in range(len(boxes)):
        if overlap[i] > 0.7:
            temp_area = cal_area(boxes[i])
            if overlap[i] * temp_area / gt_area > 0.4:
                ft_boxes.append(boxes[i])
                check = check + 1
    # check--------inter/part > 0.7 and inter/bbox > 0.4
    print(' origin check: ', check,' --- ',end='')
    if check < 2:
        check = 0
        ft_boxes = []
        for i in range(len(boxes)):
        #if overlap(i) > 0.5:
            temp_area = cal_area(boxes[i])
            if overlap[i] * temp_area / gt_area > 0.4:
                ft_boxes.append(boxes[i])
                check = check + 1
        # check-------------------       inter/bbox > 0.4
        print(' second check: ', check,' --- ',end='')

    # too much, cut to 200
    if check > 200:
        del_num = check - 200
        for i in range(del_num):
            del_box_num = random.randint(0,check - i -1)
            #del_box_num = ceil(del_box_num)
            ft_boxes.pop(del_box_num)

    # still too little, can't satisfy inter/bbox > 0.4
    if check < 2:
        check = 0
        ft_boxes = []
        #  check ----------------------inter/bbox >0.3
        for i in range(len(boxes)):
            # if overlap(i) > 0.5:
            temp_area = cal_area(boxes[i])
            if overlap[i] * temp_area / gt_area > 0.3:
                ft_boxes.append([boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],overlap[i]])
                check = check + 1
        #           inter/bbox >0.3,          choose 2 max(inter/part)
        if check >= 2:
            ps_up = ft_boxes.index(max(ft_boxes[:, 4]))
            ft_boxes[ps_up][4] = 0
            ps_up2 = ft_boxes.index(max(ft_boxes[:, 4]))
            parts[0,:] = ft_boxes[ps_up, 0:4]
            parts[1,:] = ft_boxes[ps_up2, 0:4]
            truth = 1
            print(' ###########  low   ############ ', len(boxes))
            return truth,parts
        #check<2, can't satisfy inter/bbox>0.3, so choose 2 max(inter/part)
        else:
            ps_up = overlap.tolist().index(max(overlap))
            p1 = overlap[ps_up]
            overlap[ps_up] = 0
            ps_up2 = overlap.tolist().index(max(overlap))
            p2 = overlap[ps_up2]
            parts[0] = boxes[ps_up]
            parts[1] = boxes[ps_up2]
            print(' @@@@@@@@@@@ lowest @@@@@@@@@@@@ %d %d, %d' % (p1, p2, len(boxes)))
            truth = 0
            return truth, parts
    # (check>2=)satisfy inter/bbox > 0.4, (maybe inter/part>0.7)
    else:
        truth = 1
        print(' check number %d' % (check),end='')
        box_num = len(ft_boxes)
        print(' box_num %d' % box_num, end='')
        totalnum = int((box_num - 1) * box_num / 2)
        scores = np.zeros((totalnum, 5))
        cont = 0
        for i in range(box_num - 1):
            for j in range(i + 1, box_num):
                xi1 = max(ft_boxes[i][0], gt[0])
                yi1 = max(ft_boxes[i][1], gt[1])
                xi2 = min(ft_boxes[i][2], gt[2])
                yi2 = min(ft_boxes[i][3], gt[3])
                xj1 = max(ft_boxes[j][0], gt[0])
                yj1 = max(ft_boxes[j][1], gt[1])
                xj2 = min(ft_boxes[j][2], gt[2])
                yj2 = min(ft_boxes[j][3], gt[3])
                x1 = max(xi1, xj1)
                y1 = max(yi1, yj1)
                x2 = min(xi2, xj2)
                y2 = min(yi2, yj2)
                iarea = (xi2 - xi1 + 1) * (yi2 - yi1 + 1)
                jarea = (xj2 - xj1 + 1) * (yj2 - yj1 + 1)
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                if (w <= 0 or h <= 0):
                    inter = 0
                else:
                    inter = w * h
                s = iarea + jarea - 2 * inter
                if s < 1:
                    continue
                x01 = max(ft_boxes[i][0], ft_boxes[j][0])
                y01 = max(ft_boxes[i][1], ft_boxes[j][1])
                x02 = min(ft_boxes[i][2], ft_boxes[j][2])
                y02 = min(ft_boxes[i][3], ft_boxes[j][3])

                #temp = [ft_boxes[i], ft_boxes[j], [x01, y01, x02, y02]]
                # for k in range(3):  # 0,1,2
                #     temp[k] = [temp[k][0] * scale_map[0][0], temp[k][1] * scale_map[0][1], temp[k][2] * scale_map[0][0],
                #                temp[k][3] * scale_map[0][1]]
                #temp = [[xi1,yi2,xi2,yi2],[xj1,yj1,xj2,yj2],[x1,y1,x2,y2]]

                # count1 = 0
                # count2 = 0
                # average1 = 0
                # average2 = 0
                # average = 0
                counti = cal_area(ft_boxes[i])
                countj = cal_area(ft_boxes[j])
                sumi = saliency[(ft_boxes[i][1] - 1):ft_boxes[i][3], (ft_boxes[i][0] - 1):ft_boxes[i][2]].sum()
                sumj = saliency[(ft_boxes[j][1] - 1):ft_boxes[j][3], (ft_boxes[j][0] - 1):ft_boxes[j][2]].sum()
                if (x02-x01<0)or(y02-y01<0):
                    map = (sumi+sumj)/(counti+countj)
                else:
                    countij = (x02-x01+1)*(y02-y01+1)
                    sumij = saliency[(y01-1):y02,(x01-1):x02].sum()
                    map = (sumi+sumj-sumij)/(counti+countj-countij)
                # for m in range((temp[0][1]) , (temp[0][3]) + 1):
                #     for n in range((temp[0][0]) , (temp[0][2]) + 1):
                #         average1 = average1 + saliency[m - 1][n - 1]
                #         count1 = count1 + 1
                # map1 = average1 / count1
                #
                # for m in range(int([1][1]) + 1, (temp[1][3]) + 1):
                #     for n in range((temp[1][0]) + 1, (temp[1][2]) + 1):
                #         average2 = average2 + saliency[m - 1][n - 1]
                #         count2 = count2 + 1
                # map2 = average2 / count2
                #
                # for m in range((temp[2][1]) + 1, (temp[2][3]) + 1):
                #     for n in range((temp[2][0]) + 1, (temp[2][2]) + 1):
                #         average = average - saliency[m - 1][n - 1]
                #         count2 = count2 - 1
                # map = (average + average1 + average2) / (count1 + count2)
                if map < 1:
                    continue

                score = beta * math.log(s) + math.log(map)
                scores[cont] = [i, j, score, s, map]
                cont = cont + 1

    #print(scores[:,2])
    location = scores[:,2].tolist().index(max(scores[:,2]))
    print(' cont :',cont,' --- ')
    if cont == 0:
        check = 0
        ft_boxes = []
        for i in range(len(boxes)):
            # if overlap(i) > 0.5
                temp_area = cal_area(boxes[i])
                if temp_area / gt_area > 0.3:
                    ft_boxes.append([boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3],overlap[i]])
                    check = check + 1

        if check >= 2:
            id=[]
            for i in range(len(ft_boxes)):
                id.append(ft_boxes[i][4])
            ps_up = id.index(max(id))
            ft_boxes[ps_up][4] = 0
            ps_up2 = id.index(max(id))
            parts[0,:] = ft_boxes[ps_up][ 0: 4]
            parts[1,:] = ft_boxes[ps_up2][0: 4]
        # parts = cat(1, parts, boxes(ps_up,:))
        # parts = cat(1, parts, boxes(ps_up2,:))
            print(' pass too much,s<1 or map<1,relax condition~~~~')
            truth=0
            return truth,parts
        else:
            print(' I make it ~~~~~split gt to up and down~~~~~~~~')
            truth = 0
            parts[0, :] = [gt[0],gt[1],gt[2],math.floor((gt[1]+gt[3])/2)]
            parts[1, :] = [gt[0],math.ceil((gt[1]+gt[3]+1)/2),gt[2],gt[3]]
            return truth, parts
    truth = 1
    parts[0,:] = ft_boxes[max(0, int(scores[location, 0]))]
    parts[1,:] = ft_boxes[max(1, int(scores[location, 1]))]
    return truth,parts

def filter_out_2_TIP(threadID,phase):
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~config

    root = os.popen('pwd').read().strip()
    root = os.path.join(root, 'CAR196')

    count = 0
    boxes_num = 1000
    im_size = np.zeros((1,2))
    boxes = np.zeros((boxes_num, 4))
    beta = 1
    outdir = root+'/parts_9/'
    image_path = root+'/images/'
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print(' mkdir '+outdir+' ...')

    heatdir = root+'/heatmap_read/'
    bbox_txt_dir = root+'/datalist/bbox'+phase+'%d000.list'%threadID #bboxtrain8000.list  or bboxtest8000.list
    with open(bbox_txt_dir,'r') as f:
        gt_bbox = f.readlines()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # flag =1
    # boxes_num = -7
    # name = '000000'
    # idx = 0
    # saliency = 0
    # filename = 0
    boxes = []
    id_2_name = np.genfromtxt(os.path.join(root,phase+'.list'), dtype=str)
    for i in range(len(gt_bbox)):
        idx = id_2_name[threadID*1000+i,0][0:6]
        content = np.genfromtxt(os.path.join(root,'ss_patch_9process',idx+'.list'))
        # 0 378 1023 389    x  y   w   h -> x1,y1,x2,y2
        for j in range(min(len(content),1000)):
            boxes.append([int(content[j][0]),int(content[j][1]),
                          int(content[j][0])+int(content[j][2]),
                          int(content[j][1])+int(content[j][3])])
        sal = Image.open(os.path.join(heatdir,idx+'.png'))
        saliency = np.array(sal)
        # w,h
        im_size[0,0] = saliency.shape[1] #768.1024
        im_size[0,1] = saliency.shape[0]
        #  bbox.append(str(x) + ' ' + str(y) + ' ' + str(width) + ' ' + str(height) + ' ' + '\n')
        x,y,w,h = map(int,gt_bbox[i].strip().split())
        gt = [x,y,x+w,y+h]
        print('# phase {} number----{}   #image----{}.jpg start... '.format(phase,threadID*1000+i+1, idx) ,end='')

        [truth, parts] = filtering_2_TIP(boxes, gt, im_size, saliency, beta)
        count = count + truth
        # ~~~~~~~~~~~~~part,bbox~~~~~~~~~~~~
        img = Image.open(os.path.join(image_path, id_2_name[threadID*1000+i,0]))
        # part1
        img_part1 = img.crop(parts[0])
        img_part1.save(os.path.join(outdir, idx+ '_1.jpg'))
        # part2
        img_part2 = img.crop(parts[1])
        img_part2.save(os.path.join(outdir, idx+ '_2.jpg'))
        # image
        img.save(os.path.join(outdir, idx+ '.jpg'))
        # bbox
        img_bbox = img.crop(gt)
        img_bbox.save(os.path.join(outdir, idx+ '_bbox.jpg'))
        img.close()
        img_part1.close()
        img_part2.close()
        img_bbox.close()
        sal.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--threadid', type=int, default=3)
    args = parser.parse_args()
    thread = args.threadid
    for phase in ['train','test']:
        start = time.time()
        filter_out_2_TIP(thread,phase)
        end = time.time()
        print('~~~~~~~~~~~Runtime: {}~~~~~~~~~~~'.format(end - start))

    ####commond
    # cd TIP/TIP
    # conda activate lijiang
    # python filterpart.py --threadid 0