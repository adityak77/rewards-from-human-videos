# rewards-from-human-videos

Reproducing DVD with current versions of Metaworld, Mujoco-Py, and TensorFlow 2.0.

## Setup for DVD reproduction

- Download Something Something from [here](https://developer.qualcomm.com/software/ai-datasets/something-something). (LEAP Lab only) can access this in `rome` or `paris` and `rsync` to other machines as necessary.

- Install Mujoco 2.0 and mujoco-py. Instructions for this are [here](https://github.com/openai/mujoco-py#install-mujoco).

- Clone this repository

- Create and activate conda environment 

```
conda env create -f conda_env_setup.yml
conda activate dvd_t2t
```

- Add references to modified versions of `Metaworld` and `tensor2tensor` and `dvd/sim_envs`

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

- We can currently reproduce the training command as follows:

```
cd dvd
python train.py --num_tasks 6 --traj_length 0 --log_dir "test/" --similarity --batch_size 24 --im_size 120 --seed 0 --lr 0.01 --pretrained --human_data_dir ../smthsmth/sm/20bn-something-something-v2 --sim_dir demos/ --human_tasks 5 41 44 46 93 94 --robot_tasks 5 41 93 --add_demos 60 --gpus 0
```

- We can currently collect robot demos as follows:

```
python collect_data.py --xml env1 --task_num 94
```

## TODO

In order to run inference using DVD, we need to use Stochastic Variational Video Predictor (SV2P) from `tensor2tensor`. In order to do this, you need to register the problem `dvd/human_problem.py` first. To do this, run something like 

```
DATA_DIR = data_dir
OUTPUT = sv2p_output

cd dvd

tensor2tensor/tensor2tensor/bin/t2t-trainer --t2t_usr_dir=~/rewards-from-human-videos/dvd/human_updated \
--schedule=train --alsologtostderr --generate_data --tmp_dir "tmp/" --data_dir=$DATA_DIR --output_dir=$OUTPUT \
--problem=human_updated --model=next_frame_sv2p --hparams_set=next_frame_sv2p --train_steps=200000 --eval_steps=100 \
--hparams="video_num_input_frames=1, video_num_target_frames=15, batch_size=2"
```

This command currently does not work. There are many version issues that require changes in either TF 2.0 or TF 1.0. Once this is resolved, we should be able to run

```
cd dvd
python sv2p_plan.py --num_epochs 100 --num_tasks 6 --task_num 94 --seed 0 --sample_sz 100 --similarity 1 \
--num_demos 3 --model_dir pretrained/dvd_human_tasks_6_robot_tasks_3.pth.tar --xml env1 --cem_iters 0 --root ./ \
--sv2p_root ../tensor2tensor/tensor2tensor/
```




## Troubleshooting

For training, make sure the following versions of `protobuf` and `pillow` are installed.

```
pip install protobuf==3.9.2 pillow==6.1.0
```
