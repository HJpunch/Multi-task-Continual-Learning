#!/bin/sh
ARCH=$1
DESC=$2
SEED=0

ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 \


CUDA_VISIBLE_DEVICES=2 python -m examples.continual -a ${ARCH} --seed ${SEED} --margin 0.3 \
	--num-instances 4 -b 128 -j 8 --warmup-step 1000 --lr 0.00004 --alpha 3 --optimizer AdamW --weight-decay 0.0005 --scheduler cosine_lr \
	--iters 1 \
	--logs-dir logs/${ARCH}-${DESC} --config ./scripts/config_ablation5.yaml --data-config scripts/config_continual.yaml \
	--dropout_clo 0.0 --patch_size_clo 16 --stride_size_clo 16 --patch_size_bio 16 --stride_size_bio 16 \
	--train-list None \
	--validate \
	--test_feat_type f --attn_type dual_attn --fusion_loss all --fusion_branch bio+clot --vit_type base --vit_fusion_layer 2 \
	--print-freq 1 \
	--save-freq 1