#!/bin/bash

# Set the job name
JOB_NAME=minerva-replica-350M-data-2T-vocab-50k

# Huggingface home
export HF_HOME=/leonardo_scratch/large/userexternal/$(whoami)/.cache/hf_home

# Automatically generate directory and set paths
BASE_DIR=/leonardo/home/userexternal/$(whoami)/minerva/llm-foundry
PYENV="$BASE_DIR"/llmfoundry-venv
SLURM_DIR="$BASE_DIR"/scripts/slurm/minerva-replica
TRAIN_SCRIPT="$BASE_DIR"/scripts/train/train.py
CONFIG="$BASE_DIR"/scripts/train/yamls/minerva-replica/"$JOB_NAME".yaml
LOG_DIR="$BASE_DIR"/logs/minerva-replica
mkdir -p $LOG_DIR


# --- Generate train_${JOB_NAME}.sh ---
cat <<EOF > $SLURM_DIR/train_${JOB_NAME}.sh
#!/bin/bash

echo "-----------------------------------"
echo "HOSTNAME: \$(hostname -f)"
echo "NPROCS: \$NPROCS"
echo "NODEID: \$SLURM_NODEID"
base_rank=\$((\$SLURM_NODEID * \$NPROCS))
echo "WORLD SIZE: \$WORLD_SIZE"
echo "BASE RANK: \$base_rank"
echo "MASTER addr: \$MASTER_ADDR"
echo "MASTER port: \$MASTER_PORT"

cmd="composer -v --world_size \$WORLD_SIZE --base_rank \$base_rank -n \$NPROCS --master_addr \$MASTER_ADDR --master_port \$MASTER_PORT --node_rank \$SLURM_NODEID $TRAIN_SCRIPT $CONFIG"

echo \"COMMAND: \$cmd\"
echo "-----------------------------------"
eval "\$cmd"
EOF
chmod +x $SLURM_DIR/train_${JOB_NAME}.sh

# --- Generate SLURM script train_slurm_$JOB_NAME.sh ---
cat <<EOF > $SLURM_DIR/train_slurm_${JOB_NAME}.sh
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH -o $LOG_DIR/%x-%j.out
#SBATCH -e $LOG_DIR/%x-%j.err
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
master_addr=\$(scontrol show hostnames \$SLURM_JOB_NODELIST | head -n 1)

module load profile/deeplrn cuda/12.1
source $PYENV/bin/activate
cd $BASE_DIR

export NPROCS=4
export MASTER_ADDR=\$master_addr
export MASTER_PORT=\$master_port
export WORLD_SIZE=\$((\$SLURM_NNODES * \$NPROCS))

export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1    

export NCCL_ASYNC_ERROR_HANDLING=1
export HF_HOME=$HF_HOME
export WANDB_MODE=offline
export HF_TOKEN=\$(python -c "import huggingface_hub; print(huggingface_hub.HfFolder.get_token() or '')")

srun $SLURM_DIR/train_${JOB_NAME}.sh
EOF
chmod 644 $SLURM_DIR/train_slurm_${JOB_NAME}.sh
