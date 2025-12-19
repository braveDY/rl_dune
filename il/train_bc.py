"""
行为克隆（Behavior Cloning）模仿学习
使用专家数据训练策略网络
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm


class ExpertDataset(Dataset):
    """专家数据集"""
    def __init__(self, data_path):
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.observations = torch.FloatTensor(data['observations'])
        self.actions = torch.FloatTensor(data['actions'])
        print(f"加载数据: {len(self.observations)} 条transitions")
    
    def __len__(self):
        return len(self.observations)
    
    def __getitem__(self, idx):
        return self.observations[idx], self.actions[idx]


class BCPolicy(nn.Module):
    """行为克隆策略网络"""
    def __init__(self, obs_dim, action_dim, action_low, action_high, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出[-1, 1]范围
        )
        # 动作空间缩放参数
        self.register_buffer('action_low', torch.FloatTensor(action_low))
        self.register_buffer('action_high', torch.FloatTensor(action_high))
        self.register_buffer('action_scale', (self.action_high - self.action_low) / 2.0)
        self.register_buffer('action_bias', (self.action_high + self.action_low) / 2.0)
    
    def forward(self, obs):
        # 输出[-1, 1]，然后缩放到实际动作空间
        action = self.net(obs)
        return action * self.action_scale + self.action_bias


def train_bc(data_path, save_path, action_low, action_high, epochs=10000, batch_size=256, lr=1e-3):
    """训练BC策略"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    dataset = ExpertDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 数据诊断
    print(f"\n📊 数据诊断:")
    print(f"观测形状: {dataset.observations.shape}")
    print(f"动作形状: {dataset.actions.shape}")
    print(f"观测范围: [{dataset.observations.min():.2f}, {dataset.observations.max():.2f}]")
    print(f"动作范围: [{dataset.actions.min():.2f}, {dataset.actions.max():.2f}]")
    print(f"动作均值: {dataset.actions.mean(dim=0)}")
    print(f"动作标准差: {dataset.actions.std(dim=0)}")
    print(f"期望动作范围: [{action_low[0]:.2f}, {action_low[1]:.2f}] 到 [{action_high[0]:.2f}, {action_high[1]:.2f}]")
    
    # 创建模型
    obs_dim = dataset.observations.shape[1]
    action_dim = dataset.actions.shape[1]
    policy = BCPolicy(obs_dim, action_dim, action_low, action_high).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\n开始训练BC策略...")
    print(f"观测维度: {obs_dim}, 动作维度: {action_dim}")
    print(f"设备: {device}")
    
    # 训练循环
    best_loss = float('inf')
    pbar = tqdm(range(epochs), desc="训练BC策略", ncols=80)
    for epoch in pbar:
        total_loss = 0
        num_batches = 0
        
        for obs, actions in dataloader:
            obs, actions = obs.to(device), actions.to(device)
            
            # 前向传播
            pred_actions = policy(obs)
            loss = criterion(pred_actions, actions)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
                
        pbar.set_description(f"Loss: {total_loss:.3f}")
            
        # 显示预测样本
        if (epoch + 1) % 100 == 0:
            with torch.no_grad():
                sample_obs = dataset.observations[:5].to(device)
                sample_actions = dataset.actions[:5]
                pred = policy(sample_obs).cpu()
                print(f"  真实动作: {sample_actions[0].numpy()}")
                print(f"  预测动作: {pred[0].numpy()}")
        
        # 保存最佳模型
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(policy.state_dict(), save_path.replace('.pth', '_best.pth'))
        
    
    # 保存模型
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), save_path)
    print(f"\n✓ BC策略已保存到: {save_path}")
    
    return policy


if __name__ == "__main__":
    from environment.env_dune import V_MIN, V_MAX, W_MIN, W_MAX
    
    train_bc(
        data_path="imitation_data/expert_demonstrations.pkl",
        save_path="models/bc_policy.pth",
        action_low=[V_MIN, W_MIN],
        action_high=[V_MAX, W_MAX],
        epochs=50000,  # 先用少量epoch测试
        batch_size=256,
        lr=1e-3
    )