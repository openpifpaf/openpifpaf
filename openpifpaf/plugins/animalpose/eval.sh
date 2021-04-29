#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 20
#SBATCH --mem 96G
#SBATCH --time 24:00:00
#SBATCH --account=vita
#SBATCH --gres gpu:1

pattern=$1
shift

module load gcc python cuda
source /home/bertoni/.venv/animal/bin/activate

srun /bin/bash -c "find outputs/ -name \"$pattern\" -exec python3 -m openpifpaf.eval --checkpoint {} $(printf "%s " "$@") \;"
