#!/bin/bash

# # List of appliances and buildings
# appliances=("kettle" "washingmachine" "microwave" "fridge" "dishwasher")
# buildings=(2 3 5)

# # 临时补一下2，3的dishwasher，因为原来的对不上（不知道为什么seq2point）论文给的错了，补完minmax了补standard
appliances=("dishwasher")
buildings=(2 3)

# Directory to store output files
output_dir="testing_process"

# Check if the output directory exists, if not, create it
mkdir -p "$output_dir"

# Loop through each appliance and building and run the command
for appliance in "${appliances[@]}"; do
    for building in "${buildings[@]}"; do
        output_file="${output_dir}/seq2seqCNN/output_REFIT_${appliance}_seq2seqCNN_B${building}_standard.txt"
        python3 test_reg.py --appliance_name "$appliance" --building "$building" --dataset REFIT > "$output_file"
    done
done
