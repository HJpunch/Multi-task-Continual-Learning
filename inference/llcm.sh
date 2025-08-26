#!/usr/bin/env bash
set -euo pipefail

root=..

mode=$1
checkpoint=$2

task='llcm'    # vi reid

sh ./scripts/test.sh \
	transformer_dualattn_joint \
	"./logs/transformer_dualattn_joint-${mode}/checkpoints/${checkpoint}.pth.tar" \
	cross \
	"<path-to-data-directory>/data/${task}/gallery.txt" \
	"<path-to-data-directory>/data/${task}/query.txt" \
	"<path-to-data-directory>/data/${task}"
