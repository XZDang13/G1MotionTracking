import torch
from dataclasses import dataclass, field

@dataclass
class CMD:
    pos: float = 0.0
    kp: float = 0.0
    kd: float = 0.0

@dataclass
class G1Config:
    # ---- required (no defaults) must come first ----
    policy_joints_order: list[str]
    stiffness: torch.Tensor
    damping: torch.Tensor
    action_offset: torch.Tensor
    action_scale: torch.Tensor

    # ---- optional (defaults) after ----
    control_dt: float = 1 / 50

    joints_settings: dict[str, tuple] = field(default_factory=lambda: {
        "left_hip_pitch_joint": (),
        "left_hip_roll_joint": (),
        "left_hip_yaw_joint": (),
        "left_knee_joint": (),
        "left_ankle_pitch_joint": (),
        "left_ankle_roll_joint": (),
        "right_hip_pitch_joint": (),
        "right_hip_roll_joint": (),
        "right_hip_yaw_joint": (),
        "right_knee_joint": (),
        "right_ankle_pitch_joint": (),
        "right_ankle_roll_joint": (),
        "waist_yaw_joint": (),
        "waist_roll_joint": (),
        "waist_pitch_joint": (),
        "left_shoulder_pitch_joint": (),
        "left_shoulder_roll_joint": (),
        "left_shoulder_yaw_joint": (),
        "left_elbow_joint": (),
        "left_wrist_roll_joint": (),
        "left_wrist_pitch_joint": (),
        "left_wrist_yaw_joint": (),
        "right_shoulder_pitch_joint": (),
        "right_shoulder_roll_joint": (),
        "right_shoulder_yaw_joint": (),
        "right_elbow_joint": (),
        "right_wrist_roll_joint": (),
        "right_wrist_pitch_joint": (),
        "right_wrist_yaw_joint": (),
    })

    gravity_vector: torch.Tensor = field(
        default_factory=lambda: torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
    )

    # derived fields (not passed to __init__)
    pd_params: dict[str, tuple[float, float]] = field(init=False)
    joint_num: int = field(init=False)

    def __post_init__(self):
        # basic sanity checks (optional but helpful)
        assert len(self.policy_joints_order) == self.stiffness.numel() == self.damping.numel()
        self.pd_params = {}
        
        for name in self.joints_settings:
             self.pd_params[name] = (0, 0)

        
        for idx, name in enumerate(self.policy_joints_order):
            self.pd_params[name] = (self.stiffness[idx].item(), self.damping[idx].item())
        
        
        self.joint_num = len(self.policy_joints_order)
