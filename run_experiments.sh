#!/usr/bin/env bash

HC_DATA_ROOT=/media/lext/FAST/knee_landmarks/workdir/high_cost_data
WORKDIR=/media/lext/FAST/knee_landmarks/workdir
EXP_DIR=hc_experiments_todo/

python scripts/experiments_runner.py --data_root ${HC_DATA_ROOT} \
                                     --workdir ${WORKDIR} \
                                     --experiment ${EXP_DIR} \
                                     --log_dir ${WORKDIR}/experiment_runs_queed \
                                     --script_path scripts/train.py

for SNP in $(ls ${WORKDIR}/snapshots/ | grep "2019_")
do
    python scripts/oof_inference.py --workdir ${WORKDIR} --snapshot $SNP
done
