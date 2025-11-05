#!/usr/bin/env bash         \

set -x

currenttime=`date "+%Y%m%d_%H%M%S"`

CONFIG=./experiments/LoRA/ViT-B_prompt_lora_8.yaml
CKPT='/path/to/ViT-B_16.npz'
WEIGHT_DECAY=0.05
mkdir -p logs
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \

mkdir -p logs/stanford_cars
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=33535 supernet_train_prompt.py \
    --data-path=/path/to/dataset/Stanford_Cars --data_set=stanford_cars --cfg=${CONFIG} --resume=${CKPT} \
    --output_dir=./saves/stanford_cars_lr-2e-4_wd-${WEIGHT_DECAY}_lora_100ep_noaug_xavier_dp01_same-transform_nomixup \
    --batch-size=8 --epochs=100 --is_LoRA --weight-decay=${WEIGHT_DECAY} --no_aug --mixup=0 --cutmix=0 \
    --direct_resize --smoothing=0 --dirname=${currenttime} \
    --dist-eval --use_dist \
    --opt adamw --warmup-lr 1e-7 --warmup-epochs 10 --lr 2e-4 --min-lr 1e-8 --drop-path 0 \
2>&1 | tee -a logs/stanford_cars/${currenttime}-stanford_cars-2e-4-lora.log 
echo -e "\033[32m[ Please check log: \"logs/stanford_cars/${currenttime}-stanford_cars-2e-4-lora.log\" for details. ]\033[0m"
