#!/bin/bash
# Run CEM planner with different environment seeds.

NUM_SEEDS=5

for (( i=1; i<=$NUM_SEEDS; i++ ))
do
    python cem_plan_inpaint_egohos.py --seed $i "$@"
done
