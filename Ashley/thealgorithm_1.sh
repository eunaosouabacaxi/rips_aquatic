#!/bin/bash
#$ -cwd
#$ -o thealgorithm_q1.out
#$ -j y 
#$ -l rh7,h_rt=6:00:00,h_data=64G
#$ -M trex.escalante@gmail.com
#$ -m bea

#load the job environment
. /u/local/Modules/default/init/modules.sh
module load anaconda3

# run code
python thealgorithm_q1.py > thealgorithm_q1.out