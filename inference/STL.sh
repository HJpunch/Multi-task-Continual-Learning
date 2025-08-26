#!/usr/bin/env bash
set -euo pipefail

mode1=$1
checkpoint1=$2

mode2=$3
checkpoint2=$4

mode3=$5
checkpoint3=$6

task1='market'  # traditional reid
task2='ltcc'    # cc reid
task3='llcm'    # vi reid

sh ./inference/market.sh ${mode1} ${checkpoint1}
sh ./inference/ltcc.sh ${mode2} ${checkpoint2}
sh ./inference/llcm.sh ${mode3} ${checkpoint3}
