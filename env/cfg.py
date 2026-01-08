import torch
import numpy as np
from isaaclab.utils import configclass
from Ref2Act.config.env_cfg import G1MotionTrackingEnvCfg, EventCfg
from Ref2Act.motion_lib import SamplerMod

@configclass
class G1JabEnv(G1MotionTrackingEnvCfg):
    expert_motion_file = "env/assests/jab.npz"
    sampler_mod = SamplerMod.Clamp
    episode_length_s = 3.0

@configclass
class G1JabTrainingEnv(G1JabEnv):
    events = EventCfg()