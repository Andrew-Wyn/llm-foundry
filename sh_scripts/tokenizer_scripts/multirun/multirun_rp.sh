#!/bin/bash

for i in {1..9}; do 
# i=9
for j in {1..3}; do

# Set the job name
PARTITION="middle"
JOB_NAME="rp_"$PARTITION"_""$i"_"$j"
# echo "Running job: $JOB_NAME"

sbatch ./$JOB_NAME.sh

done
done
