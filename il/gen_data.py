"""专家数据收集脚本 - 用于收集NeuPAN专家策略的演示数据，供后续模仿学习使用"""
import numpy as np
import pickle
from pathlib import Path
from environment.env_dune import DiffRobotEnv, V_MIN, V_MAX, W_MIN, W_MAX
from neupan import neupan
import warnings
import tqdm

warnings.filterwarnings("ignore")


class ExpertPolicy:
    """NeuPAN专家策略包装器"""
    
    def __init__(self, config_path: str):
        self.planner = neupan.init_from_yaml(config_path)
    
    def get_action(self, robot_state, lidar_scan) -> np.ndarray:
        points = self.planner.scan_to_point(robot_state, lidar_scan)
        action, _ = self.planner(robot_state, points)
        action = np.asarray(action).flatten()
        
        v = action[0] if len(action) >= 2 else 0.0
        w = action[1] if len(action) >= 2 else 0.0
        
        v = np.clip(v, V_MIN, V_MAX)
        w = np.clip(w, W_MIN, W_MAX)
        return np.array([v, w], dtype=np.float32)
    
    def reset(self, start, goal):
        self.planner.update_initial_path_from_goal(start, goal)
        self.planner.reset()


def collect_expert_data(env, expert, num_episodes=100, max_steps=500):
    """收集专家演示数据"""
    observations, actions, rewards, next_observations, dones = [], [], [], [], []
    success_count = 0
    
    print(f"开始收集 {num_episodes} 个episodes的专家数据...")
    pbar = tqdm.tqdm(range(num_episodes), ncols=80)
    
    for ep in pbar:
        obs, info = env.reset()
        expert.reset(info['start'], info['goal'])
        
        for step in range(max_steps):
            robot_state = env._env.get_robot_state()
            lidar_scan = env._env.get_lidar_scan()
            action = expert.get_action(robot_state, lidar_scan)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_obs)
            dones.append(done)
            
            obs = next_obs
            if done:
                if info['is_reach_target']:
                    success_count += 1
                break
        
        pbar.set_description(f"success {success_count/(ep+1)*100:.1f}%")
    
    actions_array = np.array(actions)
    print(f"\n收集的动作统计:")
    print(f"  v范围: [{actions_array[:, 0].min():.3f}, {actions_array[:, 0].max():.3f}]")
    print(f"  w范围: [{actions_array[:, 1].min():.3f}, {actions_array[:, 1].max():.3f}]")
    
    dataset = {
        'observations': np.array(observations, dtype=np.float32),
        'actions': np.array(actions, dtype=np.float32),
        'rewards': np.array(rewards, dtype=np.float32),
        'next_observations': np.array(next_observations, dtype=np.float32),
        'dones': np.array(dones, dtype=bool),
        'metadata': {
            'num_episodes': num_episodes,
            'success_count': success_count,
            'success_rate': success_count / num_episodes,
            'total_transitions': len(observations)
        }
    }
    
    print(f"\n收集完成！")
    print(f"成功episodes: {success_count}/{num_episodes} ({success_count/num_episodes*100:.1f}%)")
    print(f"总transitions: {len(observations)}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='收集专家演示数据')
    parser.add_argument('--num_episodes', type=int, default=128, help='收集的episode数量')
    parser.add_argument('--max_steps', type=int, default=150, help='每个episode的最大步数')
    parser.add_argument('--output', type=str, default='imitation_data/expert_demonstrations.pkl',
                        help='输出文件路径')
    args = parser.parse_args()
    
    env_config = "config/irsim_env_dune_train.yaml"
    expert_config = "TEST/convex_obs/planner.yaml"
    
    env = DiffRobotEnv(env_config, display=False)
    expert = ExpertPolicy(expert_config)
    
    dataset = collect_expert_data(env, expert,
                                  num_episodes=args.num_episodes,
                                  max_steps=args.max_steps)
    env.close()
    
    save_path = Path(args.output)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\n数据已保存到: {save_path}")
    print(f"文件大小: {save_path.stat().st_size / 1024 / 1024:.2f} MB")
