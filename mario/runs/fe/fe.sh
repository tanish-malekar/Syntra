#!/bin/bash
# sequentially test two checkpoints on the 400 fresh seeds, one GPU
export HF_HOME=/data01/tmalekar/mario/hf PYTHONPATH=/data01/tmalekar/mario/code
P=/data01/tmalekar/mario/venv/bin/python
for c in r128_rl_best r128_rl_final; do
  $P -u -m mario.train.evaluate --checkpoint /data01/tmalekar/mario/init/$c --episodes 400 --seed-offset 5000 --label ${c}_indep400 --run-dir /data01/tmalekar/mario/runs/fe
done

