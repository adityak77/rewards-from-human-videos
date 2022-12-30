#!/bin/bash
# Run CEM planner with different environment seeds.

NUM_SEEDS=10

for (( i=1; i<=$NUM_SEEDS; i++ ))
do
    python cem_plan_open_loop.py --seed $i "$@"
done