#! /bin/bash
#SBATCH --job-name=openai_batch # Job name
#SBATCH --output=openai_batch.txt # Standard output and error.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH -p cscc-cpu-p
#SBATCH --time=12:00:00
python openai_batch.py --input sentences.csv \
        --codes-csv codes.csv \
        --output-dir results/ \
        --model gpt-5.6-terra \
        --annotations-csv sentences.csv \
        --num-versions 3 