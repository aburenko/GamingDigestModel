#!/usr/bin/env bash

python main.py
# sbatch --partition=lopri --gres=gpu:1 --mem=16G --time=1-00:00:00 --signal=TERM@120 slurm_run.sh