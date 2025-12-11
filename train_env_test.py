#!/usr/bin/env python3
"""
DUNE环境测试脚本 - 测试随机化的障碍物环境
支持随机动作和训练好的策略两种模式
"""

import torch
import numpy as np
import os
from train_td3 import Actor
from env.env_dune import DiffRobotEnv, V_MAX, V_MIN, W_MAX, W_MIN
from dune.dune import DUNE
import yaml
import argparse


def load_config_and_dune(config_file="config/dune_train.yaml"):
    """加载配置文件并初始化DUNE"""
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    corner_points = config["robot"]["corner_points"]
    vertices = np.array([
        [corner_points["rear_left"][0], corner_points["front_left"][0],
         corner_points["front_right"][0], corner_points["rear_right"][0]],
        [corner_points["rear_left"][1], corner_points["front_left"][1],
         corner_points["front_right"][1], corner_points["rear_right"][1]]
    ])

    checkpoint_dir = config["train"]["checkpoint_dir"]
    dune = DUNE(robot_vertices=vertices,
                checkpoint=checkpoint_dir + "/model_500.pth")
    dune.switch_mode(0)  # 使用DUNE网络模式

    return dune, config


def load_policy(model_path, obs_dim):
    """加载训练好的策略"""
    action_low = np.array([V_MIN, W_MIN])
    action_high = np.array([V_MAX, W_MAX])

    policy = Actor(obs_dim, 2, action_low, action_high)
    checkpoint = torch.load(model_path, map_location='cpu')
    policy.load_state_dict(checkpoint['actor'])
    policy.eval()
    print(f"✓ 成功加载模型: {model_path}")

    return policy


def main():
    dune, config = load_config_and_dune()
    env = DiffRobotEnv(irsim_yaml="config/irsim_env_dune_train.yaml", display=True, dune=dune)
    obs_dim = env.observation_space.shape[0]
    policy = load_policy("models/td3_step_100000.pth", obs_dim)

    for episode in range(10):
        obs, info = env.reset(episode + 10)
        done = False
        step = 0

        while not done and step < 200:
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                action = policy(obs_tensor).cpu().numpy().flatten()

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            env.render()
            step += 1


if __name__ == "__main__":
    main()