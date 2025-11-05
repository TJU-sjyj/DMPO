#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=./experiments/LoRA/ViT-B_prompt_lora_8.yaml
CKPT='/path/to/ViT-B_16.npz'
WEIGHT_DECAY=0.0001
mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

for LR in 0.002
do
    for DATASET in cifar caltech101 dtd oxford_flowers102 svhn sun397 oxford_iiit_pet patch_camelyon eurosat resisc45 diabetic_retinopathy clevr_count clevr_dist dmlab kitti dsprites_loc dsprites_ori smallnorb_azi smallnorb_ele
    do 
        mkdir -p logs/${DATASET}
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=33565 supernet_train_prompt.py \
            --data-path=/path/to/dataset/vtab-1k/${DATASET} --data_set=${DATASET} --cfg=${CONFIG} --resume=${CKPT} \
            --output_dir=./saves/${DATASET}_lr-${LR}_wd-${WEIGHT_DECAY}_lora_100ep_noaug_xavier_dp01_same-transform_nomixup \
            --batch-size=8 --lr=${LR} --epochs=100 --is_LoRA --weight-decay=${WEIGHT_DECAY} --no_aug --mixup=0 --cutmix=0 \
            --direct_resize --smoothing=0 --dirname=${currenttime} \
            --dist-eval --use_dist \
        2>&1 | tee -a logs/${DATASET}/${currenttime}-${DATASET}-${LR}-lora.log 
        echo -e "\033[32m[ Please check log: \"logs/${DATASET}/${currenttime}-${DATASET}-${LR}-lora.log\" for details. ]\033[0m"
    done
done