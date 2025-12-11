from ENV.env_dune import DiffRobotEnv
from train_bc import BCPolicy
import torch
import numpy as np

env = DiffRobotEnv("ENV/irsim_env.yaml", display=True)
state, _ = env.reset()
policy = BCPolicy(env.observation_space.shape[0], env.action_space.shape[0], env.action_space.low, env.action_space.high)
policy.load_state_dict(torch.load("models/bc_policy_best.pth"))
for episode in range(10):
    state, _ = env.reset(np.random.randint(0, 1000))
    for i in range(100):
        # 修改: 将numpy数组转换为torch张量
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = policy(state_tensor).float().cpu().numpy().flatten()
        state, reward, terminated, truncated, info = env.step(action)

        env.render()
        if terminated or truncated:
            break