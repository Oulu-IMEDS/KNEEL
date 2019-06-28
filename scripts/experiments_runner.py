# Refactored idea from https://dmitryulyanov.github.io/if-you-are-lazy-to-install-slurm/
from joblib import Parallel, delayed
from queue import Queue
import os
import torch
from functools import partial
import argparse
import glob
import time
import subprocess


def experiment_worker(exp_name, queue_obj, data_root, workdir, script, n_threads, log_dir):
    gpu = queue_obj.get()
    exp_fname = exp_name.split('/')[-1].split('.yml')[0]
    print(f'Working on {exp_fname} | GPU {gpu}')

    cmd = f"python {script}"
    cmd += f" --n_threads {n_threads}"
    cmd += f" --dataset_root {data_root}"
    cmd += f" --workdir {workdir}"
    cmd += f" --experiment {exp_name}"

    my_env = os.environ.copy()
    my_env['CUDA_VISIBLE_DEVICES'] = f"{gpu}"

    time.sleep(gpu)
    with open(f'{log_dir}/{exp_fname}.log', 'w') as f_log:
        subprocess.call(cmd.split(), stdout=f_log, stderr=f_log, env=my_env)

    queue_obj.put(gpu)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads_per_task', type=int, default=6)
    parser.add_argument('--data_root', default='')
    parser.add_argument('--workdir', default='')
    parser.add_argument('--script_path', default='')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--experiment_dir', default='')
    args = parser.parse_args()

    n_gpus = torch.cuda.device_count()
    q = Queue(maxsize=n_gpus)
    for i in range(n_gpus):
        q.put(i)

    worker = partial(experiment_worker,
                     queue_obj=q,
                     data_root=args.data_root,
                     workdir=args.workdir,
                     script=args.script_path,
                     n_threads=args.n_threads_per_task,
                     log_dir=args.log_dir)

    experiments = glob.glob(os.path.join(args.experiment_dir, '*.yml'))
    os.makedirs(args.log_dir, exist_ok=True)
    Parallel(n_jobs=n_gpus, backend="threading")(delayed(worker)(exp) for exp in experiments)

