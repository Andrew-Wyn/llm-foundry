#!/bin/sh

echo "-----------------------------------"
echo "HOSTNAME: $(hostname -f)"
echo "NPROCS: $NPROCS"
echo "NODEID: $SLURM_NODEID"
base_rank=$(($SLURM_NODEID * $NPROCS))
echo "WORLD SIZE: $WORLD_SIZE"
echo "BASE RANK: $base_rank"
echo "MASTER addr: $MASTER_ADDR"
echo "MASTER port: $MASTER_PORT"

cmd="composer -v --world_size $WORLD_SIZE --base_rank $base_rank -n $NPROCS --master_addr $MASTER_ADDR --master_port $MASTER_PORT --node_rank $SLURM_NODEID /leonardo/home/userexternal/lmoroni0/__Work/llm-foundry/scripts/train/train.py /leonardo/home/userexternal/lmoroni0/__Work/llm-foundry/scripts/train/yamls/pretrain/minerva-7B-test.yaml"

echo "COMMAND: $cmd"
echo "-----------------------------------"

eval "$cmd"

# composer /leonardo/home/userexternal/rorland1/llm-foundry/scripts/train/train.py /leonardo/home/userexternal/rorland1/llm-foundry/scripts/train/yamls/pretrain/minestral-350m-en-it-12B.yaml

#composer [-h] [--version] [-n NPROC] [--stdout STDOUT] [--stderr STDERR] [-v] [-m] [-c] [--world_size WORLD_SIZE]
#                [--base_rank BASE_RANK] [--node_rank NODE_RANK] [--master_addr MASTER_ADDR] [--master_port MASTER_PORT]
#                training_script ...