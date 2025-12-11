import numpy as np
import torch
import torch.nn as nn
from typing import Optional, Tuple
import os
import sys
from shapely.geometry import Polygon, Point
try:
    from .obs_point_net import ObsPointNet
    from .utils import convex_vertex_to_ineq
except ImportError:
    from obs_point_net import ObsPointNet
    from utils import convex_vertex_to_ineq

class DUNE(nn.Module):
    def __init__(self, 
                 robot_vertices: Optional[np.ndarray] = None,
                 checkpoint: Optional[str] = None,
                 device: str = 'cpu'):
        super().__init__()
        self.device = torch.device(device)
        
        self.robot_vertices = self._init_robot_vertices(robot_vertices)
        self.G, self.h = self._calc_robot_constraints()
        self.robot_hull = Polygon(self.robot_vertices.T)
        
        self.n_constraints = self.G.shape[0]
        self.distance_net = ObsPointNet(input_dim=2, output_dim=self.n_constraints).to(self.device)
        
        self._initialize_model(checkpoint)
        
        self.mode = 0

    def _init_robot_vertices(self, robot_vertices: Optional[np.ndarray]) -> np.ndarray:
        if robot_vertices is not None:
            if robot_vertices.shape[0] != 2 or robot_vertices.shape[1] < 3:
                raise ValueError(f"Invalid robot vertices shape! Expected (2, N) with N≥3, got {robot_vertices.shape}")
            return robot_vertices.copy()
        
        length, width = 1.6, 2.0
        return np.array([
            [-length/2, length/2, length/2, -length/2],
            [-width/2, -width/2, width/2, width/2]
        ])

    def _calc_robot_constraints(self) -> Tuple[torch.Tensor, torch.Tensor]:
        G_np, h_np = convex_vertex_to_ineq(self.robot_vertices)
        if G_np is None or h_np is None:
            raise RuntimeError("Failed to generate valid convex hull constraints! Check if vertices form a convex polygon")
        
        G = torch.from_numpy(G_np).float().to(self.device)
        h = torch.from_numpy(h_np).float().to(self.device)
        return G, h

    def _initialize_model(self, checkpoint: Optional[str]) -> None:
        if checkpoint and os.path.exists(checkpoint):
            try:
                state_dict = torch.load(checkpoint, map_location=self.device)
                self.distance_net.load_state_dict(state_dict, strict=False)
                self.distance_net.eval()
                print(f"✓ Successfully loaded DUNE model: {checkpoint}")
                return
            except Exception as e:
                print(f"⚠ Failed to load model: {str(e)}")
                sys.exit(1)
        else:
            print(f"⚠ Model file not found: {checkpoint}")
            sys.exit(1)
        
    def compute_distances(self, points: np.ndarray) -> np.ndarray:
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError(f"Invalid points shape! Expected (N, 2), got {points.shape}")
        if len(points) == 0:
            return np.array([])

        if self.mode == 0:
            points_tensor = torch.from_numpy(points).float().to(self.device)
            with torch.no_grad():
                mu = self.distance_net(points_tensor)
                Gp = torch.matmul(points_tensor, self.G.T)
                Gp_minus_h = Gp - self.h.T
                distances = torch.sum(mu * Gp_minus_h, dim=1)
            return distances.cpu().numpy()
        else:
            return np.array([self.robot_hull.distance(Point(p)) for p in points])

    def polar_to_cartesian(self, ranges: np.ndarray, angles: np.ndarray) -> np.ndarray:
        if len(ranges) != len(angles):
            raise ValueError(f"Mismatched lengths! ranges({len(ranges)}) vs angles({len(angles)})")
        
        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)
        return np.stack([x, y], axis=1)

    def forward(self, ranges: np.ndarray, angles: np.ndarray) -> Tuple[np.ndarray, float]:
        points = self.polar_to_cartesian(ranges, angles)
        distances = self.compute_distances(points)
        min_distance = np.min(distances) if len(distances) > 0 else np.inf
        return distances, min_distance

    def switch_mode(self, mode: int) -> None:
        if mode not in [0, 1]:
            raise ValueError(f"Invalid mode! Only 0(DUNE) or 1(Shapely) allowed, got {mode}")
        self.mode = mode
        mode_name = "DUNE Network" if mode == 0 else "Shapely Traditional Geometry"
        print(f"✓ Switched distance calculation mode to: {mode_name}")





if __name__ == "__main__":
    test_points = np.array([
        [1.0, 0.0],  
        [0.0, 1.0],  
        [-1.0, 0.0], 
        [0.0, -1.0], 
        [1.0, 1.0],  
        [1.0, -1.0], 
        [-1.0, 1.0], 
        [-1.0, -1.0] 
    ]) * 3.0
    
    import yaml
    import argparse

    parser = argparse.ArgumentParser(description='Train DUNE model')
    parser.add_argument('--config', type=str, default="config/dune_train.yaml", help='Path to the training configuration YAML file')
    config_file = parser.parse_args().config
    
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
    dune = DUNE(robot_vertices=vertices, checkpoint=checkpoint_dir + "/model_500.pth")
    
    # ---------------------- Test Shapely Algorithm ----------------------
    print("="*50)
    print("Testing Shapely convex hull distance algorithm:")
    print("="*50)
    dune.switch_mode(1)
    shapely_distances = dune.compute_distances(test_points)
    for idx, (point, dist) in enumerate(zip(test_points, shapely_distances)):
        print(f"Point {idx+1}: {point} → Min distance: {dist:.4f}")

    # ---------------------- Test DUNE Algorithm ----------------------
    print("\n" + "="*50)
    print("Testing DUNE algorithm (using compute_distances method):")
    print("="*50)
    dune.switch_mode(0)
    dune_distances = dune.compute_distances(test_points)
    for idx, (point, dist) in enumerate(zip(test_points, dune_distances)):
        print(f"Point {idx+1}: {point} → Min distance: {dist:.4f}")

    # ---------------------- Results Comparison ----------------------
    print("\n" + "="*50)
    print("Algorithm comparison (difference = DUNE result - Shapely result):")
    print("="*50)
    for idx in range(len(test_points)):
        point = test_points[idx]
        diff = dune_distances[idx] - shapely_distances[idx]
        print(f"Point {idx+1}: {point} → Shapely: {shapely_distances[idx]:.4f}, DUNE: {dune_distances[idx]:.4f}, Diff: {diff:.4f}")



    