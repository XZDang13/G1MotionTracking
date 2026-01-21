from enum import Enum
import math
import numpy as np
import torch
from dataclasses import dataclass

from utilis import interpolate, slerp, compute_frame_blend, IndexLike

class MotionLib:
    def __init__(self, motion_file: str, device: torch.device=torch.device("cpu")) -> None:
        
        motion_data = np.load(motion_file)
        self.device = device

        self.fps = motion_data["fps"]

        self.joint_names = motion_data["joint_names"].tolist()
        self.body_names = motion_data["body_names"].tolist()

        self.joint_pos = torch.as_tensor(motion_data["joint_pos"], dtype=torch.float32, device=self.device)
        self.joint_vel = torch.as_tensor(motion_data["joint_vel"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(motion_data["body_pos_w"], dtype=torch.float32, device=self.device)
        self.body_quaternions = torch.tensor(motion_data["body_quat_w"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            motion_data["body_lin_vel_w"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            motion_data["body_ang_vel_w"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / self.fps
        self.num_frames = self.joint_pos.shape[0]
        self.duration = self.dt * self.num_frames
        print(f"motion data loaded: {self.duration} s")

    def sample_motion(
        self,
        times: torch.Tensor,
        position_offsets: torch.Tensor|None=None
    ) -> dict[str: torch.Tensor]:
        
        index_0, index_1, blend = compute_frame_blend(
            times, self.duration, self.num_frames, self.dt
        )

        index_0 = index_0.to(device=self.device)
        index_1 = index_1.to(device=self.device)
        blend = blend.to(device=self.device, dtype=torch.float32)

        joint_pos = interpolate(
            self.joint_pos, b=self.joint_pos, blend=blend, start=index_0, end=index_1
        )

        joint_vel = interpolate(
            self.joint_vel, b=self.joint_vel, blend=blend, start=index_0, end=index_1
        )

        body_positions = interpolate(
            self.body_positions, b=self.body_positions, blend=blend, start=index_0, end=index_1
        )

        if position_offsets is not None:
            if position_offsets.ndim == 2:
                position_offsets = position_offsets.unsqueeze(1)
            body_positions += position_offsets

        body_quaternions = slerp(
            self.body_quaternions, q1=self.body_quaternions, blend=blend, start=index_0, end=index_1
        )
        body_linear_velocities = interpolate(
            self.body_linear_velocities, b=self.body_linear_velocities, blend=blend, start=index_0, end=index_1
        )
        body_angular_velocities = interpolate(
            self.body_angular_velocities, b=self.body_angular_velocities, blend=blend, start=index_0, end=index_1
        )

        motions = {
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_positions": body_positions,
            "body_quaternions": body_quaternions,
            "body_linear_velocities": body_linear_velocities,
            "body_angular_velocities": body_angular_velocities,

        }
        
        return motions