#!/usr/bin/env bash
set -euo pipefail

mode=$1
checkpoint=$2

task='cuhk_pedes'  # t2i reid

sh ./scripts/test.sh \
	transformer_dualattn_joint \
	"./logs/transformer_dualattn_joint-${mode}/checkpoints/${checkpoint}.pth.tar" \
	t2i \
	"<path-to-data-directory>/data/${task}/query_t2i_v2.txt" \
	"<path-to-data-directory>/data/${task}/gallery_t2i_v2.txt" \
	"<path-to-data-directory>/data/${task}"
