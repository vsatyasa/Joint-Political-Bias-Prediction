#!/bin/bash

# Define arrays of values for each argument
models=("lstm")
epochs=(50)
hidden_nodes=(100 200)
learning_rates=(0.001)
versions=("baseline" "separate_class" "joint_class")

# Iterate over all combinations of argument values and run the Python file
for model in "${models[@]}"; do
  for epoch in "${epochs[@]}"; do
    for node in "${hidden_nodes[@]}"; do
      for lr in "${learning_rates[@]}"; do
        for version in "${versions[@]}"; do
          python train.py \
            --model "$model" \
            --epochs "$epoch" \
            --hidden_nodes "$node" \
            --learning_rate "$lr" \
            --version "$version" 
        done
      done
    done
  done
done




