#!/bin/bash
#SBATCH --job-name=minerva-replica-350M-data-2T-vocab-50k
#SBATCH -o /leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/logs/minerva-replica/%x-%j.out
#SBATCH -e /leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/logs/minerva-replica/%x-%j.err
#SBATCH -A mnrv_bblscp
#SBATCH -p boost_usr_prod
#SBATCH --time 24:00:00
#SBATCH -N 8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --exclusive

master_port=11111
master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

module load profile/deeplrn cuda/12.1
source /leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/llmfoundry-venv/bin/activate
cd /leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/minerva/llm-foundry/

export NPROCS=4
export MASTER_ADDR=$master_addr
export MASTER_PORT=$master_port
export WORLD_SIZE=$(($SLURM_NNODES * $NPROCS))

export NCCL_ASYNC_ERROR_HANDLING=1
export HF_HOME=/leonardo_scratch/large/userexternal/lcolosi0/.cache/hf_home/
export WANDB_MODE=offline
export HF_TOKEN=$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

srun /leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/scripts/slurm/minerva-replica/train_minerva-replica-350M-data-2T-vocab-50k.sh
