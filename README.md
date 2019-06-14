# OPAM_rerun
Note: If you want to use CAR196 dataset, download CAR196/images, train.list, test.list and config.yaml from here: {链接: https://pan.baidu.com/s/10mva14K4Plccqq41PFddgA 提取码: b2m2}, and put them(images, train.list, test.list and config.yaml) in CAR196.\<br>  
If you want CUB200 dataset, download CUB200/images, train.list, test.list and config.yaml from this website too, or use CUB200/split_train_test.py to process your original CUB_200_2011 and generate these files.\<br>  
note to change the corresponding code(path{'root':'CUB200'}) in all py files\<br>
Besides, you should also install the environment as the 'requirements.txt'\<br>  
Then, the code can be run as the following command step by step:\<br>  
		 *1  $ bash run.sh setup\<br>  
		 *2  $ bash run.sh patch \<br>  
		 *3  $ bash run.sh object\<br>  
		 *4  $ bash run.sh part\<br>  
		 *5  $ bash run.sh fusion\<br>  
		 or just run $ bash run.sh all \<br>  
the final result is in /log/fusion_predict.log\<br>  

