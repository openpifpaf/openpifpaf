#!/bin/bash

#SBATCH --output="OKS-%j.log"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH -w belldevcv01
# auie #SBATCH --gres="gpu:1"

python oks.py $@
