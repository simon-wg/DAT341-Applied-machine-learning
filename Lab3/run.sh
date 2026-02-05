#! /bin/bash

#SBATCH --cpus-per-task=4

source /data/users/simongre/Lab3/.venv/bin/activate
python3 main.py
