#!/usr/bin/env bash
#SBATCH --job-name CNN
#SBATCH --nodes 1
#SBATCH --mem 120GB
#SBATCH --gres gpu:1
#SBATCH --partition gpu
#SBATCH --time 0-05:00
#SBATCH --account comsm0045
#SBATCH --mail-type=END
#SBATCH --mail-user=tg17437@bristol.ac.uk

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job ID is $SLURM_JOB_ID
echo This job runs on the following machines:
echo `echo $SLURM_JOB_NODELIST | uniq`
echo GPU number: $CUDA_VISIBLE_DEVICES

export OCL_DEVICE=1

module purge
module load languages/anaconda3/2019.07-3.6.5-tflow-1.14

python train_saliency.py
python evaluation.py --preds final_preds.pkl --gts val.pkl
python visualisation.py --preds final_preds.pkl --gts val.pkl --outdir results

echo `date`
