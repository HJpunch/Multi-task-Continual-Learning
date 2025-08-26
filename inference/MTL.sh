#!/usr/bin/env bash
set -euo pipefail

mode=$1
checkpoint=$2

task1='market'  # traditional reid
task2='ltcc'    # cc reid
task3='llcm'    # vi reid

sh ./scripts/test.sh \
	transformer_dualattn_joint \
	"./logs/transformer_dualattn_joint-${mode}/checkpoints/${checkpoint}.pth.tar" \
	sc \
	"../Instruct-ReID/data/${task1}/datalist/query.txt" \
	"../Instruct-ReID/data/${task1}/datalist/gallery.txt" \
	"../Instruct-ReID/data/${task1}"

sh ./scripts/test.sh \
	transformer_dualattn_joint \
	"./logs/transformer_dualattn_joint-${mode}/checkpoints/${checkpoint}.pth.tar" \
	cc \
	"../Instruct-ReID/data/${task2}/datalist/query_cc.txt" \
	"../Instruct-ReID/data/${task2}/datalist/gallery_cc.txt" \
	"../Instruct-ReID/data/${task2}"

sh ./scripts/test.sh \
	transformer_dualattn_joint \
	"./logs/transformer_dualattn_joint-${mode}/checkpoints/${checkpoint}.pth.tar" \
	cross \
	"../Instruct-ReID/data/${task3}/query.txt" \
	"../Instruct-ReID/data/${task3}/gallery.txt" \
	"../Instruct-ReID/data/${task3}"
