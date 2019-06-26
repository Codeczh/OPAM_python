## OPAM_rerun
### Dataset: 
   If you want to use CAR196 dataset, download `CAR196/images` from here: 
   [CAR196](https://zhenhuangc.oss-cn-hongkong.aliyuncs.com/car196_images.tar.gz), 
   and put `/images` in CAR196 folder.  
    
While if you want CUB200 dataset, download `CUB200/images` from here:
 [CUB200](https://zhenhuangc.oss-cn-hongkong.aliyuncs.com/cub200_images.tar.gz),
  or use `CUB200/split_train_test.py` to process your original
   [CUB_200_2011](www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz) and generate these images.  

**Note:** If you want your own dataset, remember to edit your `config.yaml` and change *classnum: xxx*.    

----------------------------------------
### Environment:  
Install the environment as the 'requirements.txt'.
```
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt 
```
Besides, you should install pytorch according your CUDA version 
[https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/),
and install [apex](https://github.com/NVIDIA/apex)
```
git clone https://github.com/NVIDIA/apex  
cd apex  
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .   
```

----------------------
### Command:  
Check the `run.sh` script, and change hyperparameters accordingly. 
(when the dataset choise occurs, just choose the number of you dataset )

* 1  $  **`bash run.sh setup`**  to run `filternet.py`   `9_selective_search`
* 2  $  **`bash run.sh patch`**  to run `patch_filter.py`  `patchnet_bn_rerun.py`
* 3  $  **`bash run.sh object`** to run `saliencynet.py` `CAM.py` `objectnet_bn_rerun.py`
* 4  $  **`bash run.sh part`**   to run `filterpart.py` `align_part.py` `partnet_bn_rerun.py`
* 5  $  **`bash run.sh fusion`** to run `fusion_predict.py`

or just  run $ **`bash run.sh all`**,
the final result is in `/log/fusion_predict.log`

