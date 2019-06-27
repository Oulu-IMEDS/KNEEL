#!/usr/bin/env bash

HC_DATA_ROOT=/media/lext/FAST/knee_landmarks/workdir/high_cost_data
WORKDIR=/media/lext/FAST/knee_landmarks/workdir

for EXP_FILE in $(ls hc_experiments/)
do
    python scripts/train.py --dataset_root ${HC_DATA_ROOT} --workdir ${WORKDIR} --experiment experiments/${EXP_FILE}
done