# cad_recode/dataset.py

import os, random
import numpy as np
import torch
from torch.utils.data import Dataset
import cadquery as cq
from cad_recode.utils import sample_points_on_shape, farthest_point_sample

class CadRecodeDataset(Dataset):
    def __init__(self, root_dir, split='train', n_points=256, noise_std=0.01, noise_prob=0.5):
        """
        Args:
            root_dir (str): Path to the root dataset folder (contains subfolders train/val/test).
            split (str): One of 'train', 'val', or 'test'.
            n_points (int): Number of points to sample via FPS.
            noise_std (float): Standard deviation of Gaussian noise.
            noise_prob (float): Probability of adding noise (only in training).
        """
        self.split = split
        self.n_points = n_points
        self.noise_std = noise_std
        self.noise_prob = noise_prob
        # Collect all .py files under e.g. root_dir/train/*/*.py
        split_dir = os.path.join(root_dir, split)
        self.files = []
        if os.path.isdir(split_dir):
            for batch_dir in os.listdir(split_dir):
                batch_path = os.path.join(split_dir, batch_dir)
                if os.path.isdir(batch_path):
                    for fname in os.listdir(batch_path):
                        if fname.endswith('.py'):
                            self.files.append(os.path.join(batch_path, fname))
        self.files.sort()  # for reproducibility

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        # Read the CadQuery script
        with open(file_path, 'r') as f:
            code = f.read()

        # Execute the script to obtain the CAD solid
        local_vars = {}
        try:
            exec(code, {"cq": cq}, local_vars)
        except Exception as e:
            raise RuntimeError(f"Failed to execute CAD script {file_path}: {e}")

        # Extract the shape: common variable names or last workplane
        shape = None
        if "result" in local_vars:
            shape = local_vars["result"]
        elif "r" in local_vars:
            shape = local_vars["r"]
        elif "shape" in local_vars:
            shape = local_vars["shape"]

        # If we got a CadQuery Workplane, retrieve its solid
        if isinstance(shape, cq.Workplane):
            try:
                shape = shape.val()
            except Exception:
                shape = shape.objects[0]

        if shape is None:
            raise RuntimeError(f"No shape found in script {file_path}")

        # Sample surface points (via mesh tessellation) and downsample
        pts = sample_points_on_shape(shape, n_samples=1024)  # returns (M,3)
        if pts.shape[0] > self.n_points:
            pts = farthest_point_sample(pts, self.n_points)   # FPS downsampling:contentReference[oaicite:6]{index=6}

        # Normalize point cloud (centroid=0, max radius=1)
        centroid = pts.mean(axis=0)
        pts = pts - centroid
        max_dist = np.linalg.norm(pts, axis=1).max()
        if max_dist > 1e-6:
            pts = pts / max_dist

        # Add Gaussian noise on-the-fly during training
        if self.split == 'train' and random.random() < self.noise_prob:
            noise = np.random.normal(0.0, self.noise_std, size=pts.shape).astype(np.float32)
            pts = pts + noise

        # Convert to torch tensor
        points_tensor = torch.from_numpy(pts.astype(np.float32))  # shape (n_points, 3)
        code_str = code  # the original CadQuery code as string
        return points_tensor, code_str
