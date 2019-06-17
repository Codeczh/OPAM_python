# OPAM_rerun
##**Dataset:** 
   If you want to use CAR196 dataset, download CAR196/images, train.list, test.list and config.yaml from here: [CAR196](https://zhenhuangc.oss-cn-hongkong.aliyuncs.com/car196_images.tar.gz), and put them(images, train.list, test.list and config.yaml) in CAR196 folder.  
    
While if you want CUB200 dataset, download CUB200/images, train.list, test.list and config.yaml from here: [CUB200](https://zhenhuangc.oss-cn-hongkong.aliyuncs.com/cub200_images.tar.gz), or use CUB200/split_train_test.py to process your original CUB_200_2011 and generate these files.  

**Note** to change the corresponding code(     __path{ 'root' : 'CUB200' }__ ) in all py files.

##**Environment:**

Besides, you should also install the environment as the 'requirements.txt'.


##Command:  
Then, the code can be run as the following command step by step:


>	 * 1  $  bash run.sh setup  
>	 * 2  $  bash run.sh patch  
>	 * 3  $  bash run.sh object  
>	 * 4  $  bash run.sh part  
>	 * 5  $  bash run.sh fusion  
		or just  run $ bash run.sh all    
the final result is in /log/fusion_predict.log

