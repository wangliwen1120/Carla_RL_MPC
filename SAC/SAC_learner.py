# -*-coding:utf-8-*-
import os
import sys
import Carla_gym

curr_path = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在绝对路径
parent_path = os.path.dirname(curr_path)  # 父路径
root_path = os.getcwd()
sys.path.append(parent_path)  # 添加路径到系统路径
import re
import gym
import torch
import datetime
# from SoftActorCritic.env_wrapper import NormalizedActions
from SAC.env import NormalizedActions
from SAC.agent import SAC
from common.utils import save_results, make_dir
from common.plot import plot_rewards

import argparse
import numpy as np

from torch.utils.tensorboard import SummaryWriter


def mkdir(path):
    """
    创建指定的文件夹
    :param path: 文件夹路径，字符串格式
    :return: True(新建成功) or False(文件夹已存在，新建失败)
    """

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        # os.makedirs(path)
        return path
    else:
        path, path_origin = directory_check(path)
        # os.makedirs(path)
        return path


def mkdir_origin(path):
    """
    创建指定的文件夹
    :param path: 文件夹路径，字符串格式
    :return: True(新建成功) or False(文件夹已存在，新建失败)
    """

    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        # os.makedirs(path)
        return path
    else:
        path, path_origin = directory_check(path)
        # os.makedirs(path)
        return path_origin


def directory_check(directory_check):
    temp_directory_check = directory_check
    i = 1
    while i:

        if os.path.exists(temp_directory_check):
            search = '_'
            numList = [m.start() for m in re.finditer(search, temp_directory_check)]
            numList[-1]
            temp_directory_check = temp_directory_check[0:numList[-1] + 1] + str(i)
            i = i + 1
        else:
            return temp_directory_check, temp_directory_check[0:numList[-1] + 1] + str(i - 2)


class SACConfig:
    def __init__(self, env_name='default', train_name='default', total_timesteps=30000):
        self.env_name = env_name
        self.algo = 'SAC'
        self.train_name = train_name
        self.base_path = root_path + "/Results/RL_Results/" + self.env_name + '/' + self.train_name
        self.result_path = root_path + "/Results/RL_Results/" + self.env_name + '/' + self.train_name + '/results/'  # path to save results
        self.model_path = root_path + "/Results/RL_Results/" + self.env_name + '/' + self.train_name + '/models/'  # path to save models
        self.log_path = root_path + "/Results/RL_Results/runs_info/" + self.env_name + '/runs/' + self.train_name  # path to save logs
        self.ma_window = 10
        self.train_eps = 50000
        self.train_steps = 2000
        self.eval_eps = 50000
        self.eval_steps = 2000
        self.total_timesteps = total_timesteps
        self.gamma = 0.99
        self.soft_tau = 5e-3
        self.soft_q_lr = 3e-4
        self.policy_lr = 3e-4
        self.capacity = 1000000

        self.eval_total_timesteps = 60000
        self.hidden_dim = 256
        self.batch_size = 128
        self.alpha_lr = 3e-4
        self.AUTO_ENTROPY = True
        self.DETERMINISTIC = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SAC_Learner:

    def __init__(self, SAC_cfg, env_cfg, args):

        self.rewards = None
        self.ma_rewards = None
        self.SAC_cfg = SAC_cfg
        self.base_path = self.SAC_cfg.base_path
        self.origin_base_path = mkdir_origin(self.SAC_cfg.base_path)
        self.origin_model_path = self.origin_base_path + '/models/'
        self.origin_result_path = self.origin_base_path + '/results/'
        self.model_path = self.base_path + '/models/'
        self.result_path = self.base_path + '/results/'
        self.adv_scenario_data_path = None
        self.env = None
        self.cfg = env_cfg
        self.args = args
        self.agent = None
        # tensorboard
        self.log_path = mkdir(self.SAC_cfg.log_path)
        self.writer = SummaryWriter(self.log_path)  # default at runs folder if not sepecify path
        self.curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time

    def env_agent_initialize(self, seed=1):

        self.env = gym.make(self.SAC_cfg.env_name)
        self.env.seed(seed)
        print('Env is starting')
        if self.args.play_mode:
            self.env.enable_auto_render()
        self.env.begin_modules(self.args)
        action_dim = self.env.action_space.shape[0]
        state_dim = self.env.observation_space.shape[1]

        self.agent = SAC(state_dim, action_dim, self.SAC_cfg)
        print(self.SAC_cfg.algo + ' algorithm is starting')

    def load(self, model_path):
        self.agent.load(model_path)

    def train(self):

        print('Start to train !')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        rewards = []
        ma_rewards = []  # moveing average reward
        total_nums = 0
        mean_reward = 0.0
        returns = np.array([0])
        i_step = 1
        for i_ep in range(self.SAC_cfg.train_eps):

            state = self.env.reset()

            # 每一个回合随机采样一条对抗场景参数轨迹
            next_state, reward, done, _ = self.env.step([0])

            eps_reward = 0.
            for i_step in range(self.SAC_cfg.train_steps):

                total_nums = total_nums + 1
                action = self.agent.policy_net.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                self.agent.memory.push(state, action, reward, next_state, done)
                self.agent.update(reward_scale=1., auto_entropy=self.SAC_cfg.AUTO_ENTROPY,
                                  target_entropy=-1. * self.env.action_space.shape[0], gamma=self.SAC_cfg.gamma,
                                  soft_tau=self.SAC_cfg.soft_tau)
                state = next_state
                eps_reward += reward
                if i_ep == 0 or done:
                    break
            # mean_reward = eps_reward / i_step
            if (i_ep + 1) % 1 == 0:

                rewards.append(eps_reward)
                if len(rewards) > self.SAC_cfg.ma_window:
                    # mean_reward = np.mean(rewards[-self.SAC_cfg.ma_window])
                    # ma_rewards.append(mean_reward)
                    mean_reward = eps_reward
                else:
                    mean_reward = eps_reward
                print(f"Episode:{i_ep + 1}/{self.SAC_cfg.train_eps}, Reward:{mean_reward:.3f}")
                print(f'总步数：{total_nums}')
                self.writer.add_scalar("Reward", mean_reward, total_nums)

            if total_nums >= self.SAC_cfg.total_timesteps:
                break
        print('Complete training！')
        return rewards, ma_rewards

    def eval(self):
        print('Start to eval !')
        print(f'Env: {self.SAC_cfg.env_name}, Algorithm: {self.SAC_cfg.algo}, Device: {self.SAC_cfg.device}')
        rewards = []
        total_nums = 0
        ma_rewards = []  # moveing average reward
        for i_ep in range(self.SAC_cfg.eval_eps):

            state = self.env.reset()

            next_state, reward, done, _ = self.env.step([0])

            eps_reward = 0.0
            for i_step in range(self.SAC_cfg.train_steps):

                total_nums = total_nums + 1
                action = self.agent.policy_net.get_action(state)
                next_state, reward, done, _ = self.env.step(action)

                state = next_state
                eps_reward += reward
                if i_ep == 0 or done:
                    break
            # mean_reward = eps_reward / i_step
            if (i_ep + 1) % 1 == 0:

                rewards.append(eps_reward)
                if len(rewards) > self.SAC_cfg.ma_window:
                    # mean_reward = np.mean(rewards[-self.SAC_cfg.ma_window])
                    # ma_rewards.append(mean_reward)
                    mean_reward = eps_reward
                else:
                    mean_reward = eps_reward
                print(f"Episode:{i_ep + 1}/{self.SAC_cfg.train_eps}, Reward:{mean_reward:.3f}")
                print(f'总步数：{total_nums}')
                self.writer.add_scalar("Reward", mean_reward, total_nums)

            if total_nums >= self.SAC_cfg.eval_total_timesteps:
                break
        print('Complete evaling！')
        self.rewards = rewards
        self.ma_rewards = ma_rewards

    def save(self):

        base_path = mkdir(self.SAC_cfg.base_path)
        self.model_path = base_path + '/models/'
        self.result_path = base_path + '/results/'

        make_dir(self.result_path, self.model_path)
        self.agent.save(path=self.model_path)

        save_results(self.rewards, self.ma_rewards, tag='train', path=self.result_path)
        # plot_rewards(self.rewards, self.ma_rewards, tag="train",
        #              algo=self.SAC_cfg.algo, path=self.SAC_cfg.result_path)
