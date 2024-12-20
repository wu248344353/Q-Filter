#!/bin/bash

python experiment.py --seed 124  --env halfcheetah --dataset expert --eta 1.0 --grad_norm 15.0 --exp_name qt --save_path ./save/  \
    --max_iters 500   --num_steps_per_iter 2000 --lr_decay --early_stop  --k_rewards --use_discount --q_percent 0.02

python experiment.py --seed 124  --env hopper --dataset expert  --eta 1.0 --grad_norm 9.0 --exp_name qt --save_path ./save/ \
   --max_iters 500  --num_steps_per_iter 2000 --lr_decay --early_stop --k_rewards --use_discount   --q_percent  0.02

python experiment.py --seed 124 --env walker2d --dataset expert --eta 1.0 --grad_norm 5.0 --exp_name qt --save_path ./save/ \
  --max_iters 500  --num_steps_per_iter 2000 --lr_decay --early_stop --k_rewards --use_discount   --q_percent  0.02

python experiment.py --seed 124 --env halfcheetah --dataset medium --eta 1.0 --grad_norm 15.0 --exp_name qt --save_path ./save/  \
    --max_iters 500  --num_steps_per_iter 2000 --lr_decay --early_stop --k_rewards --use_discount  --q_percent 0.05 --K 5

python experiment.py --seed 124 --env hopper --dataset medium --eta 1.0 --grad_norm 9.0 --exp_name qt --save_path ./save/  \
    --max_iters 500  --num_steps_per_iter 2000 --lr_decay --early_stop --k_rewards --use_discount  --q_percent 0.05

python experiment.py --seed 124 --env walker2d --dataset medium --eta 1.0 --grad_norm 5.0 --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 2000 --lr_decay --early_stop --k_rewards --use_discount  --q_percent 0.05

python experiment.py --seed 124 --env halfcheetah --dataset medium-replay  --eta 1.0 --grad_norm 15.0  --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 2000 --lr_decay --early_stop --k_rewards --use_discount  --q_percent 0.05 --K 5

python experiment.py --seed 124 --env hopper --dataset medium-replay  --eta 1.0 --grad_norm 9.0 --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 2000 --lr_decay --early_stop --k_rewards --use_discount  --q_percent 0.05

python experiment.py --seed 123 --env walker2d --dataset medium-replay --eta 1.0 --grad_norm 5.0 --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 2000 --lr_decay --early_stop --k_rewards --use_discount  --q_percent 0.05

python experiment.py --seed 123     --env pen --dataset human    --eta 0.1 --grad_norm 9.0     --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay --early_stop  --k_rewards --use_discount --K 5  --reward_scale 20 \

python experiment.py --seed 123     --env pen --dataset cloned       --eta 0.1 --grad_norm 9.0     --exp_name qt --save_path ./save/   \
     --max_iters 500 --num_steps_per_iter 1000 --lr_decay     --early_stop --k_rewards --use_discount  --K 5 --reward_scale 20 \

python experiment.py --seed 123 \
    --env hammer --dataset human   \
    --eta 0.1 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 60 \

python experiment.py --seed 123 \
    --env hammer --dataset cloned   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 30 \

python experiment.py --seed 123 \
    --env door --dataset human   \
    --eta 0.005 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 60 \

python experiment.py --seed 123 \
    --env door --dataset cloned   \
    --eta 0.001 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 60 \

python experiment.py --seed 123 \
    --env kitchen --dataset complete   \
    --eta 0.001 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 100 \

python experiment.py --seed 123 \
    --env kitchen --dataset partial   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset open   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset open-dense   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123     --env maze2d --dataset umaze       --eta 8.0 --grad_norm 20.0     --exp_name qt --save_path ./save/        --max_iters 100 --num_steps_per_iter 1000 --lr_decay     --early_stop --k_rewards --use_discount  --early_epoch 50  \

python experiment.py --seed 123 \
    --env maze2d --dataset umaze-dense   \
    --eta 3.0 --grad_norm 5.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset medium   \
    --eta 5.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset medium-dense   \
    --eta 5.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset large   \
    --eta 4.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env maze2d --dataset large-dense   \
    --eta 4.0 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset umaze   \
    --eta 0.05 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset umaze-diverse   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 500 --num_steps_per_iter 1000 --lr_decay \
    --early_stop --k_rewards --use_discount  --early_epoch 50 \

python experiment.py --seed 123 \
    --env antmaze --dataset medium-diverse   \
    --eta 0.01 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay --num_eval_episodes 10 \
    --early_stop --k_rewards --use_discount  --early_epoch 80 \

python experiment.py --seed 123 \
    --env antmaze --dataset large-diverse   \
    --eta 0.005 --grad_norm 9.0 \
    --exp_name qt --save_path ./save/    \
    --max_iters 100 --num_steps_per_iter 1000 --lr_decay --num_eval_episodes 10 \
    --early_stop --k_rewards --use_discount  --early_epoch 80 --reward_tune cql_antmaze 
