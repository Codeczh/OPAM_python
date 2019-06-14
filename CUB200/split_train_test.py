#import torch as t
import os
from PIL import Image

root = os.popen('pwd').read().strip()
cub200 = root +'/CUB_200_2011'
with open(cub200+'/train_test_split.txt','r') as f:
    list = f.readlines()
with open(cub200+'/images.txt','r') as f:
    num_lbl_name_list = f.readlines()
#   id_2_test = np.genfromtxt(os.path.join(self._root, 'test.list'), dtype=str)
list0 = []
list1 = []
#if not os.path.exists(root+'/train'):
#    os.mkdir(cub200+'train')
#    print('creating train dir...')
#if not os.path.exists(root+'/test'):
#    os.mkdir(cub200+'test')
#    print('creating test dir...')
if not os.path.exists(root+'/images'):
    os.mkdir(root+'/images')
    print('creating images dir')

for i in range(len(list)):
    number,flag = list[i].split()
    num,path = num_lbl_name_list[i][:-1].split()
    lbl= path[:3]
    lbl = str(int(lbl)-1)
    assert number==num
    path_original = os.path.join(cub200,'images',path)
    #path_train = os.path.join(root,'/train/'+str(i+1).zfill(6)+'.jpg')
    #path_test = os.path.join(root,'/test/'+str(i+1).zfill(6)+'.jpg')
    path_dest = os.path.join(root,'images',num.zfill(6)+'.jpg')
    image = Image.open(path_original)
    image.save(path_dest)
    image.close()
    if flag =='1': #train
        list0.append(number.zfill(6)+'.jpg'+' '+lbl+'\n')
        #image.save(path_train)
        print(i+1)
    else:          #flag=='0'  test
        list1.append(number.zfill(6)+'.jpg'+' '+lbl+'\n')
        #image.save(path_test)
        print(i+1)
    
with open(root+'/train.list','w') as f:
    f.writelines(list0)
with open(root+'/test.list','w') as f:
    f.writelines(list1)
