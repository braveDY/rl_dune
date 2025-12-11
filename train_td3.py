import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from pathlib import Path
from env.env_dune import DiffRobotEnv
import yaml
from dune.dune import DUNE
from env.env_dune import V_MIN, V_MAX, W_MIN, W_MAX

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size=1000000):
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor网络 - TD3使用确定性策略"""
    def __init__(self, obs_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出[-1, 1]
        )
        # 动作空间缩放参数
        self.register_buffer('action_low', torch.FloatTensor(action_low))
        self.register_buffer('action_high', torch.FloatTensor(action_high))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2.0)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2.0)
    
    def forward(self, obs):
        # 输出[-1, 1]，然后缩放到实际动作空间 [action_low, action_high]
        action = self.net(obs)
        return action * self.action_scale + self.action_bias


class Critic(nn.Module):
    """Critic网络（双Q网络）"""
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.q1(x), self.q2(x)


class TD3Agent:
    """TD3智能体 - 使用确定性策略"""
    def __init__(self, obs_dim, action_dim, action_low, action_high, device, bc_policy=None):
        self.device = device
        self.action_low = action_low
        self.action_high = action_high
        
        # Actor网络
        self.actor = Actor(obs_dim, action_dim, action_low, action_high).to(device)
        self.actor_target = Actor(obs_dim, action_dim, action_low, action_high).to(device)
        
        # 从之前的BC策略初始化Actor， bc_policy为之前的Actor模型
        if bc_policy is not None:
            self.actor.load_state_dict(bc_policy.state_dict())
            print("✓ Actor网络已从BC策略初始化")
        
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Critic网络
        self.critic = Critic(obs_dim, action_dim).to(device)
        self.critic_target = Critic(obs_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.action_dim = action_dim
    
    def select_action(self, obs, noise=0.1):
        """选择动作（带探索噪声）- TD3确定性策略"""
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(obs).cpu().numpy()[0]
        
        # 添加高斯噪声进行探索
        if noise > 0:
            noise_scale = (self.action_high - self.action_low) * noise
            action += np.random.normal(0, noise_scale, size=len(action))
            action = np.clip(action, self.action_low, self.action_high)
        
        return action
    
    def update(self, replay_buffer, batch_size=256, gamma=0.99, tau=0.005, 
               policy_noise=0.2, noise_clip=0.5, policy_freq=2, update_step=0):
        """更新网络"""
        if len(replay_buffer) < batch_size:
            return 0, 0
        
        # 采样batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            # 目标动作加噪声（缩放到实际动作空间）
            action_scale = torch.FloatTensor((self.action_high - self.action_low) / 2.0).to(self.device)
            noise = (torch.randn_like(actions) * policy_noise * action_scale).clamp(
                -noise_clip * action_scale, noise_clip * action_scale
            )
            next_actions = (self.actor_target(next_states) + noise).clamp(
                torch.FloatTensor(self.action_low).to(self.device),
                torch.FloatTensor(self.action_high).to(self.device)
            )
            
            # 计算目标Q值
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * gamma * torch.min(target_q1, target_q2)
        
        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q1, target_q) + nn.MSELoss()(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟更新Actor
        actor_loss_value = 0.0
        if update_step % policy_freq == 0:
            # 计算actor loss：使用Q1网络评估actor生成的动作
            actions_pred = self.actor(states)
            q1_input = torch.cat([states, actions_pred], dim=1)
            actor_loss = -self.critic.q1(q1_input).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
            actor_loss_value = actor_loss.item()
        
        return critic_loss.item(), actor_loss_value


def train_td3(env, agent, replay_buffer, total_steps=1000000, 
              start_steps=1000, batch_size=256, eval_freq=5000):
    """训练TD3"""
    print(f"\n开始TD3训练...")
    print(f"总步数: {total_steps}, 预热步数: {start_steps}")
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_steps = 0
    episode_num = 0
    

    for step in range(total_steps):
        # 选择动作
        if step < start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, noise=0.1)
        
        # 执行动作
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # 存储经验
        replay_buffer.add(obs, action, reward, next_obs, done)
        
        obs = next_obs
        episode_reward += reward
        episode_steps += 1
        
        # 更新网络
        if step >= start_steps:
            critic_loss, actor_loss = agent.update(replay_buffer, batch_size, update_step=step)
        
        # Episode结束
        if done:
            print(f"Step {step+1}, Episode {episode_num+1}, "
                  f"Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                  f"Success: {info.get('is_reach_target', False)}")
            
            obs, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode_num += 1
        
        # 定期评估和保存
        if (step + 1) % eval_freq == 0:
            save_path = f"models/td3_step_{step+1}.pth"
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'step': step + 1
            }, save_path)
            print(f"✓ 模型已保存: {save_path}")

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


if __name__ == "__main__":
    # 初始化DUNE
    dune, config = load_config_and_dune()
    
    # 初始化环境（使用正确的配置文件路径）
    env = DiffRobotEnv(
        irsim_yaml="config/irsim_env_dune_train.yaml",
        display=False,
        dune=dune
    )
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_low = np.array([V_MIN, W_MIN])
    action_high = np.array([V_MAX, W_MAX])
    
    print(f"观察空间维度: {obs_dim}")
    print(f"动作空间: [{action_low[0]:.2f}, {action_low[1]:.2f}] 到 "
          f"[{action_high[0]:.2f}, {action_high[1]:.2f}]")
    
    # 初始化TD3智能体
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 可选：从已有模型初始化（继续训练）
    pretrained_model_file = None
    pretrained_model_file = "models/td3_step_100000.pth"
    if Path(pretrained_model_file).exists():
        pretrained_model = load_policy(pretrained_model_file, obs_dim)
    
    agent = TD3Agent(
        obs_dim,
        action_dim,
        action_low,
        action_high,
        device,
        bc_policy=pretrained_model
    )
    
    # 初始化经验回放
    replay_buffer = ReplayBuffer(max_size=1000000)
    
    # 开始训练
    train_td3(
        env,
        agent,
        replay_buffer,
        total_steps=100000,
        start_steps=0,
        batch_size=256,
        eval_freq=5000
    )
    
    env.close()

    