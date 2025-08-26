#!/usr/bin/env bash
set -euo pipefail

mode=$1
checkpoint=$2

task='ltcc'    # cc reid

sh ./scripts/test.sh \
	transformer_dualattn_joint \
	"./logs/transformer_dualattn_joint-${mode}/checkpoints/${checkpoint}.pth.tar" \
	cc \
	"../Instruct-ReID/data/${task}/datalist/query_cc.txt" \
	"../Instruct-ReID/data/${task}/datalist/gallery_cc.txt" \
	"../Instruct-ReID/data/${task}"
