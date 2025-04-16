#!/bin/bash

echo "-----------------------------------"
echo "HOSTNAME: $(hostname -f)"
echo "NPROCS: $NPROCS"
echo "NODEID: $SLURM_NODEID"
base_rank=$(($SLURM_NODEID * $NPROCS))
echo "WORLD SIZE: $WORLD_SIZE"
echo "BASE RANK: $base_rank"
echo "MASTER addr: $MASTER_ADDR"
echo "MASTER port: $MASTER_PORT"

cmd="composer -v --world_size $WORLD_SIZE --base_rank $base_rank -n $NPROCS --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $SLURM_NODEID /leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/scripts/train/train.py /leonardo/home/userexternal/lcolosi0/minerva/llm-foundry/scripts/train/yamls/minerva-replica/minerva-replica-350M-data-2T-vocab-50k.yaml"

echo \"COMMAND: $cmd\"
echo "-----------------------------------"
eval "$cmd"
