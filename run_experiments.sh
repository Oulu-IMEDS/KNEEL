#!/usr/bin/env bash

WORKDIR=/media/lext/FAST/knee_landmarks/workdir
EXP_DIR=hc_experiments_todo/

python scripts/experiments_runner.py --data_root ${WORKDIR}/high_cost_data \
                                     --workdir ${WORKDIR} \
                                     --experiment ${EXP_DIR} \
                                     --log_dir ${WORKDIR}/experiment_runs_queed \
                                     --script_path scripts/train.py

#python scripts/train.py --dataset_root ${WORKDIR}/low_cost_data \
#                        --workdir ${WORKDIR} \
#                        --experiment lc_experiments/low_cost_from_scratch_multiscale_wing_mixup_no_wd_cutout5.yml 


for SNP in $(ls ${WORKDIR}/snapshots/ | grep "2019_")
do
    python scripts/oof_inference.py --workdir ${WORKDIR} --snapshot ${SNP}
done
