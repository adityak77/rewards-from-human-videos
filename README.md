# rewards-from-human-videos

Reproducing DVD with current versions of Metaworld, Mujoco-Py, and TensorFlow 2.0.

## Setup for DVD reproduction

- Download Something Something from [here](https://developer.qualcomm.com/software/ai-datasets/something-something).

- Install Mujoco 2.0 and mujoco-py. Instructions for this are [here](https://github.com/openai/mujoco-py#install-mujoco).

- Clone this repository

- Create and activate conda environment 

```
conda env create -f conda_env_setup.yml
conda activate dvd_t2t
```

- Add references to modified versions of `Metaworld`, `tensor2tensor`, `dvd/sim_envs`, `pytorch_mmpi`.

```
cd metaworld
pip install -e .

cd tensor2tensor
pip install -e .

cd dvd/sim_envs
pip install -e .
```

## Reproducing DVD

For details as to what the commands do/what the arguments are, refer to [the original repo](https://github.com/anniesch/dvd/blob/main/README.md).

- We can currently reproduce the training command as follows (you might need to use a different version of Pillow - see troubleshooting section below):

```
cd dvd
python train.py --num_tasks 6 --traj_length 0 --log_dir path/to/train/model/output --similarity --batch_size 24 --im_size 120 --seed 0 --lr 0.01 --pretrained --human_data_dir path/to/smthsmth/sm/20bn-something-something-v2 --sim_dir demos/ --human_tasks 5 41 44 46 93 94 --robot_tasks 5 41 93 --add_demos 60 --gpus 0
```

- We can currently collect robot demos as follows:

```
python collect_data.py --xml env1 --task_num 94
```

## Testing learned reward function

Using `collect_data` script above, we can generate sample trajectories in the `env` directory. We can evaluate the rewards for each of these trajectories against a demo video and get the average reward.

```
python reward_inference.py --eval_path data/file/from/collect_data/script --demo_path path/to/demo
```

Run inference with human demos on DVD tasks:

```
python cem_plan_open_loop.py --num_tasks 2 --task_id 5 --dvd --demo_path demos/task5 --checkpoint /path/to/discriminator/model
```

Run inference using ground truth (my engineered) rewards:

```
python cem_plan_open_loop.py --num_tasks 2 --task_id 5 --engineered_rewards
```

## Adding State and Visual Dynamics model

State dynamics model using PETS:

```
conda activate dvd_pets
cd dvd
git checkout state_history
python cem_plan_learned_dynamics.py --task_id 5 --engineered_rewards --learn_dynamics_model

OR 

git checkout visual_dynamics
python cem_plan_state_dynamics.py --task_id 5 --engineered_rewards --learn_dynamics_model
```

Training visual dynamics model using pydreamer

```
conda activate dvd_pydreamer
cd pydreamer
git checkout visual_dynamics

CUDA_VISIBLE_DEVICES=0,1 python train.py --configs defaults tabletop --run_name tabletop
```

Inference with CEM closed loop using visual dynamics model

```
cd dvd
[ADD HERE]
```

## Preparing inpainted data

One can inpaint using the `data_inpaint.py` script as follows:

```
conda activate e2fgvi
cd dvd

python data_inpaint.py --human_data_dir /path/to/smthsmth/sm --human_tasks 5 41 94 --opts MODEL.WEIGHTS detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl
```

To do this with EgoHOS segmentations instead of Detectron segmentations:
```
conda activate dvd_e2fgvi_detectron_egohos
cd dvd

python data_inpaint_egohos.py --human_data_dir /path/to/smthsmth/sm --human_tasks 5 41 94
```


## Training and inference on human-only inpainted data

- We might want to train on only human data. In that case, we must set `add_demos` to `0`. Adding the `--inpaint` flag will indicate that we are using inpainted videos and will modify the log file name appropriately.

```
conda activate dvd_t2t
cd dvd

python train.py --num_tasks 6 --traj_length 0 --log_dir path/to/train/model/output --similarity --batch_size 24 --im_size 120 --seed 0 --lr 0.01 --pretrained --human_data_dir path/to/smthsmth/sm/20bn-something-something-v2 --human_tasks 5 41 44 46 93 94 --add_demos 0 --inpaint --gpus 0
```

- To run CEM planning on human-only inpainted data

```
conda activate dvd_e2fgvi_detectron
cd dvd

python cem_plan_inpaint.py --task_id 5 --dvd --demo_path demos/task5 --checkpoint /path/to/trained/reward/model
```

- To do this with EgoHOS segmentations instead of Detectron segmentations:
```
conda activate dvd_e2fgvi_detectron_egohos
cd dvd

python cem_plan_inpaint_egohos.py --task_id 5 --dvd --demo_path demos/task5 --checkpoint /path/to/trained/reward/model
```



## Troubleshooting

For training, make sure the following versions of `protobuf` and `pillow` are installed.

```
pip install protobuf==3.9.2 pillow==6.1.0
```
