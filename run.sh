#!/usr/bin/env bash

#Usage :
#      bash run.sh <option>
#| Command Example ----------------------------------------------------------------------------------------|
#| bash run.sh setup    - setup necessary directories( model, log) &                                       |
#| filternet and selective search                                                                          |
#| bash run.sh patch    - fine-tune filternet & patchnet                                                   |
#| bash run.sh object    - filter patch & extract object objectnet                                         |
#| bash run.sh part    - select parts & align partnet                                                      |
#| bash run.sh fusion    - fine-tune classnet & objectnet & prediction                                     |
#| bash run.sh all      - run step 1,2,3,4                                                                 |
#|---------------------------------------------------------------------------------------------------------|

export CUDA_VISIBLE_DEVICES=2

echo "What is your dataset?"
select dataset in "CUB200" "CAR196";do
	break
done


if [ "$1" == "setup" ] || [ "$1" == "all" ];then
    if [ ! -d "${dataset}/model" ]; then
        echo "model directory does not exist, creating model..."
        mkdir -p ${dataset}/model
    fi
    if [ ! -d "${dataset}/log" ]; then
        echo "log directory does not exist, creating log ..."
        mkdir -p ${dataset}/log
    fi
    echo "Start filternet ... "
    python filternet.py --dataset ${dataset} > ${dataset}/log/filternet.log
    #generate filternet_vgg19_rerun_best_epoch.pth
    echo "Start 9_selective_search ... "
    python 9_Selective_search.py --dataset ${dataset} > ${dataset}/log/9_Selective_search.log
    #start 9 processes and generate ss_patch_9process/000001.list ...
    
fi

if [ "$1" == "patch" ] || [ "$1" == "all" ];then
    echo "Start patch_filter ... "
    python patch_filter.py --dataset ${dataset} > ${dataset}/log/patch_filter.log
    # filter patches
    echo "Start patchnet ... "
    python patchnet_bn_rerun.py --dataset ${dataset} > ${dataset}/log/patchnet_bn_rerun.log
    # train patches and generate patchnet_vgg19bn_rerun_best_epoch.pth
fi

if ["$1" == "object" ] || [ "$1" == "all" ];then
    echo "Start saliencynet ... "
    python saliencynet.py --dataset ${dataset} > ${dataset}/log/saliencynet.log
    #generate saliencynet_vgg16_best_epoch.pth
    echo "Start CAM ... "
    python CAM.py --threadid 0 --dataset ${dataset} > ${dataset}/log/CAM0.log &
    python CAM.py --threadid 1 --dataset ${dataset} > ${dataset}/log/CAM1.log &
    python CAM.py --threadid 2 --dataset ${dataset} > ${dataset}/log/CAM2.log &
    python CAM.py --threadid 3 --dataset ${dataset} > ${dataset}/log/CAM3.log &
    python CAM.py --threadid 4 --dataset ${dataset} > ${dataset}/log/CAM4.log &
    wait    #  if not enough GPU memory
    python CAM.py --threadid 5 --dataset ${dataset} > ${dataset}/log/CAM5.log &
    python CAM.py --threadid 6 --dataset ${dataset} > ${dataset}/log/CAM6.log &
    python CAM.py --threadid 7 --dataset ${dataset} > ${dataset}/log/CAM7.log &
    python CAM.py --threadid 8 --dataset ${dataset} > ${dataset}/log/CAM8.log &
    # generate bbox/000001.png , heatmap_read/000001.png  and datalist/1000.list
    # generate showbbox/000001.png and heatmap_watch/000001.png
    wait
    echo "Start objectnet ... "
    python objectnet_bn_rerun.py --dataset ${dataset} > ${dataset}/log/objectnet_bn_rerun.log
    # train objects and generate objectnet_vgg19bn_rerun_best_epoch.pth
fi

if ["$1" == "part" ] || [ "$1" == "all" ];then
    echo "Start filter part ... "
    python filterpart.py --threadid 0 --dataset ${dataset} > ${dataset}/log/filterpart0.log &
    python filterpart.py --threadid 1 --dataset ${dataset} > ${dataset}/log/filterpart1.log &
    python filterpart.py --threadid 2 --dataset ${dataset} > ${dataset}/log/filterpart2.log &
    python filterpart.py --threadid 3 --dataset ${dataset} > ${dataset}/log/filterpart3.log &
    python filterpart.py --threadid 4 --dataset ${dataset} > ${dataset}/log/filterpart4.log &
    python filterpart.py --threadid 5 --dataset ${dataset} > ${dataset}/log/filterpart5.log &
    python filterpart.py --threadid 6 --dataset ${dataset} > ${dataset}/log/filterpart6.log &
    python filterpart.py --threadid 7 --dataset ${dataset} > ${dataset}/log/filterpart7.log &
    python filterpart.py --threadid 8 --dataset ${dataset} > ${dataset}/log/filterpart8.log &
    # generate parts_9/000001_1.jpg  000001_2.jpg
    wait
    echo "Start align part ... "
    python align_part.py --dataset ${dataset} > ${dataset}/log/align_part.log
    # align parts, copy and splict to parts_align_9/000001_1.jpg  000001_2.jpg
    echo "Start partnet ..."
    python partnet_bn_rerun.py --dataset ${dataset} > ${dataset}/log/partnet_bn_rerun.log
    # train parts and generate partnet_vgg19bn_rerun_best_epoch.pth
fi

if ["$1" == "fusion" ] || [ "$1" == "all" ];then
    echo "Start fusion predict ..." 
    python fusion_predict.py --dataset ${dataset} > ${dataset}/log/fusion_predict.log
    





















