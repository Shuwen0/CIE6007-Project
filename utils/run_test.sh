#!/bin/bash

# List of appliances and buildings
appliances=("kettle" "washingmachine" "microwave" "fridge" "dishwasher")
buildings=(2 3 5)

# Directory to store output files
output_dir="testing_process"

# Check if the output directory exists, if not, create it
mkdir -p "$output_dir"

# Loop through each appliance and building and run the command
for appliance in "${appliances[@]}"; do
    for building in "${buildings[@]}"; do
        output_file="${output_dir}/output_REFIT_${appliance}_s2p_Ts2p_B${building}.txt"
        python3 test_reg.py --appliance_name "$appliance" --building "$building" --dataset REFIT > "$output_file"
    done
done
