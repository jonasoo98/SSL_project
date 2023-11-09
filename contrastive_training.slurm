#!/bin/sh
#SBATCH --job-name=ssl_project 		# Name for the job
#SBATCH --account=ie-idi		    # Billing account
#SBATCH --time=0-03:00:00     		

#SBATCH --partition=GPUQ		    # Whether you need GPUs or CPUs
#SBATCH --gres=gpu:1          		# Number of GPUs 
#SBATCH --nodes=1             		# 1 compute nodes
#SBATCH --mem=8G  			        # 8GB memory

#SBATCH --output=output.txt    		# Log file
#SBATCH --error=output.err    		# Error file

WORKDIR=/cluster/work/jssaethe/SSL_project
cd ${WORKDIR}

echo "Job was submitted from this directory: $SLURM_SUBMIT_DIR."
echo "The name of the job is: $SLURM_JOB_NAME."
echo "The job ID is $SLURM_JOB_ID."
echo "The job ran on these nodes: $SLURM_JOB_NODELIST."
echo "Number of nodes: $SLURM_JOB_NUM_NODES."

module purge 
module load Anaconda3/2022.10
conda activate maptr


python train_contrastive.py --epochs 50
