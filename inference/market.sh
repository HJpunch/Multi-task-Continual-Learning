#!/usr/bin/env bash
set -euo pipefail

mode=$1
checkpoint=$2

task='market'  # traditional reid

sh ./scripts/test.sh \
	transformer_dualattn_joint \
	"./logs/transformer_dualattn_joint-${mode}/checkpoints/${checkpoint}.pth.tar" \
	sc \
	"../Instruct-ReID/data/${task}/datalist/query.txt" \
	"../Instruct-ReID/data/${task}/datalist/gallery.txt" \
	"../Instruct-ReID/data/${task}"
