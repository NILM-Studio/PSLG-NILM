#!/usr/bin/env bash
#SBATCH --job-name=primitive-report
#SBATCH --partition=RTX3090
#SBATCH --nodelist=h104-slurm-a
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=runs/slurm/primitive-report-%j.log
set -euo pipefail
cd /home/scnu202438025446/pslg-nilm
python3 scripts/summarize_primitive_campaign.py --run-id primitive_source_20260915_v2
