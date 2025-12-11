import torch
import numpy as np
import math
import glob
from train_td3 import Actor
from irsim import EnvBase


class TD3PolicyTester:
    """TD3策略测试器"""
    
    def __init__(self, model_path, env_yaml, action_low, action_high):
        self.env = EnvBase(env_yaml, display=True)
        self.robot = self.env.robot_list[0]
        self.lidar_beams = int(self.robot.lidar.number)
        
        obs_dim = 4 + self.lidar_beams
        self.policy = Actor(obs_dim, 2, action_low, action_high)
        checkpoint = torch.load(model_path, map_location='cpu')
        self.policy.load_state_dict(checkpoint['actor'])
        self.policy.eval()
        print(f"✓ 成功加载模型: {model_path}")
        
        self.pre_action = np.zeros(2)

    def _get_observation(self, robot_state, goal, lidar_ranges):
        x, y, theta = robot_state[0], robot_state[1], robot_state[2]
        dx, dy = goal[0] - x, goal[1] - y
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        dx_robot = dx * cos_t + dy * sin_t
        dy_robot = -dx * sin_t + dy * cos_t
        
        if isinstance(lidar_ranges, dict):
            lidar_ranges = lidar_ranges['ranges']
            
        return np.array([math.hypot(dx_robot, dy_robot), math.atan2(dy_robot, dx_robot),
                         self.pre_action[0], self.pre_action[1]] + list(lidar_ranges), dtype=np.float32)

    def get_action(self, robot_state, goal, lidar_scan):
        obs = torch.FloatTensor(self._get_observation(robot_state, goal, lidar_scan)).unsqueeze(0)
        with torch.no_grad():
            self.pre_action = self.policy(obs).cpu().numpy().flatten()
        return self.pre_action

    def test_episode(self, start, goal, max_steps=200, reach_threshold=0.5):
        self.pre_action = np.zeros(2)
        self.env.reset()
        self.robot.set_state(np.array([[start[0]], [start[1]], [start[2]]]))
        self.robot.set_goal(np.array([[goal[0]], [goal[1]], [goal[2]]]))
        
        for step in range(max_steps):
            robot_state = np.squeeze(self.env.get_robot_state())
            lidar_scan = self.env.get_lidar_scan()
            distance = math.hypot(goal[0] - robot_state[0], goal[1] - robot_state[1])
            
            if distance < reach_threshold:
                print(f"  ✓ 成功到达! 步数: {step+1}, 距离: {distance:.3f}m")
                return True, step + 1, distance
            
            if self.robot.collision:
                print(f"  ✗ 碰撞! 步数: {step+1}, 距离: {distance:.3f}m")
                return False, step + 1, distance
            
            action = self.get_action(robot_state, goal, lidar_scan)
            self.env.step(action)
            self.env.render()
        
        robot_state = np.squeeze(self.env.get_robot_state())
        final_dist = math.hypot(goal[0] - robot_state[0], goal[1] - robot_state[1])
        print(f"  ✗ 超时! 步数: {max_steps}, 距离: {final_dist:.3f}m")
        return False, max_steps, final_dist
    
    def close(self):
        self.env.end()


def test_all_envs(model_path, action_low, action_high, start, goal, max_steps=150, num_trials=3):
    """测试example下所有配置文件"""
    env_files = sorted(glob.glob("test_env_cfg/*.yaml"))
    results = {}
    
    print(f"\n{'='*60}")
    print(f"开始测试 {len(env_files)} 个环境配置，每个测试 {num_trials} 次")
    print(f"{'='*60}\n")
    tester = TD3PolicyTester(model_path, env_files[0], action_low, action_high)    
    
    for env_yaml in env_files:
        env_name = env_yaml.split('/')[-1].replace('.yaml', '')
        print(f"\n[测试环境: {env_name}]")
        
        tester.env = EnvBase(env_yaml, display=True)
        tester.robot = tester.env.robot_list[0]
        successes = 0
        
        for trial in range(num_trials):
            print(f"  Trial {trial+1}/{num_trials}:")
            success, steps, dist = tester.test_episode(start, goal, max_steps)
            if success:
                successes += 1
        
        tester.close()
        success_rate = successes / num_trials * 100
        results[env_name] = success_rate
        print(f"  成功率: {successes}/{num_trials} ({success_rate:.1f}%)")
    
    # 打印汇总
    print(f"\n{'='*60}")
    print("测试结果汇总:")
    print(f"{'='*60}")
    for env_name, rate in results.items():
        status = "✓" if rate > 50 else "✗"
        print(f"  {status} {env_name}: {rate:.1f}%")
    
    return results


if __name__ == "__main__":
    MODEL_PATH = "models/td3_step_100000.pth"
    
    action_low = np.array([0.0, -1.5])
    action_high = np.array([2.0, 1.5])
    start = np.array([0.0, 10.0, 0.0])
    goal = np.array([20.0, 10.0, 0.0])
    
    test_all_envs(MODEL_PATH, action_low, action_high, start, goal, max_steps=150, num_trials=3)
