import torch
from torch.utils.data import Dataset, DataLoader
import cvxpy as cp
from torch.optim import Adam
import numpy as np
import os


class PointDataset(Dataset):
    def __init__(self, input_data, label_data, distance_data):
        """Dataset for DUNE training
        
        Args:
            input_data: point p, [2, 1]
            label_data: mu, [G.shape[0], 1] 
            distance_data: distance, scalar
        """
        self.input_data = input_data
        self.label_data = label_data
        self.distance_data = distance_data

    def __len__(self) -> int:
        return len(self.input_data)

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        label_sample = self.label_data[idx]
        distance_sample = self.distance_data[idx]
        return input_sample, label_sample, distance_sample


class DUNETrain:
    def __init__(self, model, robot_G, robot_h, checkpoint_path) -> None:
        self.G = robot_G
        self.h = robot_h
        self.model = model
        self.checkpoint_path = checkpoint_path
        
        self._construct_problem()
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

    def _construct_problem(self) -> None:
        """Construct optimization problem (10):
        
        max mu^T * (G * p - h)
        s.t. ||G^T * mu|| <= 1
             mu >= 0
        """
        self.mu = cp.Variable((self.G.shape[0], 1), nonneg=True)
        self.p = cp.Parameter((2, 1))
        
        cost = self.mu.T @ (self.G.cpu() @ self.p - self.h.cpu())
        constraints = [cp.norm(self.G.cpu().T @ self.mu) <= 1]
        
        self.prob = cp.Problem(cp.Maximize(cost), constraints)

    def _process_data(self, rand_p):
        distance_value, mu_value = self._prob_solve(rand_p)
        return (
            torch.tensor(rand_p, dtype=torch.float32),
            torch.tensor(mu_value, dtype=torch.float32),
            torch.tensor(distance_value, dtype=torch.float32),
        )

    def generate_data_set(self, data_size: int = 10000, data_range: list = [-50, -50, 50, 50]) -> PointDataset:
        """Generate dataset for training
        
        Args:
            data_size: Number of samples to generate
            data_range: [low_x, low_y, high_x, high_y]
            
        Returns:
            PointDataset: Generated dataset
        """
        input_data = []
        label_data = []
        distance_data = []
        
        rand_p = np.random.uniform(
            low=data_range[:2], high=data_range[2:], size=(data_size, 2)
        )
        rand_p_list = [rand_p[i].reshape(2, 1) for i in range(data_size)]
        
        for p in rand_p_list:
            results = self._process_data(p)
            input_data.append(results[0])
            label_data.append(results[1])
            distance_data.append(results[2])
        
        return PointDataset(input_data, label_data, distance_data)

    def _prob_solve(self, p_value):
        self.p.value = p_value
        self.prob.solve()
        return self.prob.value, self.mu.value

    def start(self, data_size: int = 100000, data_range: list = [-25, -25, 25, 25], 
              batch_size: int = 256, epoch: int = 5000, lr: float = 5e-5, 
              save_freq: int = 500, **kwargs) -> str:
        """Start training process
        
        Args:
            data_size: Number of training samples
            data_range: Training data range [low_x, low_y, high_x, high_y]
            batch_size: Training batch size
            epoch: Number of training epochs
            lr: Learning rate
            save_freq: Model save frequency
            
        Returns:
            str: Path to final saved model
        """
        self.optimizer.param_groups[0]["lr"] = float(lr)
        
        print(f"Generating dataset: {data_size} samples...")
        dataset = self.generate_data_set(data_size, data_range)
        train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Starting training: {epoch} epochs...")
        
        for i in range(epoch + 1):
            self.model.train()
            total_loss = self._train_one_epoch(train_dataloader)
            
            if i % 100 == 0:
                print(f"Epoch {i}/{epoch}, Loss: {total_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            if i % save_freq == 0 and i > 0:
                model_path = os.path.join(self.checkpoint_path, f"model_{i}.pth")
                torch.save(self.model.state_dict(), model_path)
                print(f"Saving model: {model_path}")
        
        final_model_path = os.path.join(self.checkpoint_path, f"model_final.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"✓ Training completed! Final model: {final_model_path}")
        
        return final_model_path

    def _train_one_epoch(self, train_dataloader) -> float:
        """Train one epoch
        
        Args:
            train_dataloader: Training data loader
            
        Returns:
            float: Average loss for the epoch
        """
        total_loss = 0
        device = self.G.device
        
        for input_point, label_mu, label_distance in train_dataloader:
            self.optimizer.zero_grad()
            
            input_point = input_point.to(device)
            label_mu = label_mu.to(device)
            label_distance = label_distance.to(device)
            
            input_point = torch.squeeze(input_point)
            output_mu = self.model(input_point)
            output_mu = torch.unsqueeze(output_mu, 2)
            
            distance = self._cal_distance(output_mu, input_point)
            mse_mu = self.loss_fn(output_mu, label_mu)
            mse_distance = self.loss_fn(distance, label_distance)
            mse_fa, mse_fb = self._cal_loss_fab(output_mu, label_mu, input_point)
            
            loss = mse_mu + mse_distance + mse_fa + mse_fb
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_dataloader)

    def _cal_loss_fab(self, output_mu, label_mu, input_point):
        """Calculate the loss of fa and fb
        
        fa: -mu^T * G * R^T  ==> lam^T
        fb: mu^T * G * R^T * p - mu^T * h  ==> lam^T * p + mu^T * h
        """
        mu1 = output_mu
        mu2 = label_mu
        ip = torch.unsqueeze(input_point, 2)
        mu1T = torch.transpose(mu1, 1, 2)
        mu2T = torch.transpose(mu2, 1, 2)
        
        theta = np.random.uniform(0, 2 * np.pi)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        R = torch.tensor(R, dtype=torch.float32, device=self.G.device)
        
        fa = torch.transpose(-R @ self.G.T @ mu1, 1, 2)
        fa_label = torch.transpose(-R @ self.G.T @ mu2, 1, 2)
        
        fb = fa @ ip + mu1T @ self.h
        fb_label = fa_label @ ip + mu2T @ self.h
        
        mse_lamt = self.loss_fn(fa, fa_label)
        mse_lamtb = self.loss_fn(fb, fb_label)
        
        return mse_lamt, mse_lamtb

    def _cal_distance(self, mu, input_point):
        """Calculate distance using mu and input point"""
        input_point = torch.unsqueeze(input_point, 2)
        temp = self.G @ input_point - self.h
        muT = torch.transpose(mu, 1, 2)
        distance = torch.squeeze(torch.bmm(muT, temp))
        return distance



if __name__ == "__main__":
    import yaml
    import argparse
    from obs_point_net import ObsPointNet
    from utils import convex_vertex_to_ineq
    
    parser = argparse.ArgumentParser(description='Train DUNE model')
    parser.add_argument('--config', type=str, default="config/dune_train.yaml", help='Path to the training configuration YAML file')
    config_file = parser.parse_args().config
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    print("Begin training DUNE model...")
    
    checkpoint_dir = config["train"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    corner_points = config["robot"]["corner_points"]
    vertices = np.array([
        [corner_points["rear_left"][0], corner_points["front_left"][0], 
         corner_points["front_right"][0], corner_points["rear_right"][0]],
        [corner_points["rear_left"][1], corner_points["front_left"][1], 
         corner_points["front_right"][1], corner_points["rear_right"][1]]
    ])
    G, h = convex_vertex_to_ineq(vertices)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = DUNETrain(
        model=ObsPointNet(input_dim=2, output_dim=G.shape[0]).to(device),
        robot_G=torch.from_numpy(G).float().to(device),
        robot_h=torch.from_numpy(h).float().to(device),
        checkpoint_path=checkpoint_dir
    )
    
    trainer.start(
        data_size=config["train"]["data_size"],
        data_range=config["train"]["data_range"],
        batch_size=config["train"]["batch_size"],
        epoch=config["train"]["epoch"],
        lr=config["train"]["lr"],
        save_freq=config["train"]["save_freq"]
    )

