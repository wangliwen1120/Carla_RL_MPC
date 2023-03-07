import numpy as np
# import pandas as pd
import time
import gym
import Carla_gym
import argparse
import os
from config import cfg, log_config_to_file, cfg_from_list, cfg_from_yaml_file
from SAC.SAC_learner import SAC_Learner, SACConfig


def parse_args_cfgs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/config.yaml',
                        help='specify the config for training')
    parser.add_argument('--num_timesteps', type=float, default=1e7)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--agent_id', type=int, default=1),
    parser.add_argument('-p', '--carla_port', metavar='P', default=2000, type=int,
                        help='TCP port to listen to (default: 2000)')
    parser.add_argument('--tm_port', default=8000, type=int,
                        help='Traffic Manager TCP port to listen to (default: 8000)')
    parser.add_argument('--carla_host', metavar='H', default='127.0.0.1',
                        help='IP of the host server (default: 127.0.0.1)')
    parser.add_argument('--play_mode', type=int, help='Display mode: 0:off, 1:2D, 2:3D ', default=0)
    parser.add_argument('--carla_res', metavar='WIDTHxHEIGHT', default='1280x720',
                        help='window resolution (default: 1280x720)')

    args = parser.parse_args()
    args.num_timesteps = int(args.num_timesteps)

    if args.test and args.cfg_file is None:
        path = 'logs/agent_{}/'.format(args.agent_id)
        conf_list = [cfg_file for cfg_file in os.listdir(path) if '.yaml' in cfg_file]
        args.cfg_file = path + conf_list[0]
    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.EXP_GROUP_PATH = '/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'
    return args, cfg


if __name__ == '__main__':
    train = True
    args, env_cfg = parse_args_cfgs()

    if train:
        train_name = 'train_1'
        # SAC agent
        env_name = 'gym_env-v0'
        SAC_cfg = SACConfig(env_name, train_name, total_timesteps=50000)

        agent_learner = SAC_Learner(SAC_cfg, env_cfg, args)
        agent_learner.env_agent_initialize()
        agent_learner.train()
        agent_learner.save()


    else:

        train_name_1 = 'train_2'
        env_adv_name = 'gym_env-v0'

        SAC_cfg = SACConfig(env_adv_name, train_name_1)
        agent_learner = SAC_Learner(SAC_cfg, env_cfg, args)
        agent_learner.env_agent_initialize()
        agent_learner.load(agent_learner.model_path)
        agent_learner.eval()
