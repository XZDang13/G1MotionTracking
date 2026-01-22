import torch
from Ref2Act.sim2sim import MujocoEnv

from RLAlg.normalizer import Normalizer
from RLAlg.nn.steps import StochasticContinuousPolicyStep
from model import Actor

weight = torch.load("punch.pth")

normalizer_weight = weight["actor_norm"]
actor_weight = weight["actor"]
joint_effort_limits = weight["joint_effort_limits"].cpu()
joint_pos_limits = weight["joint_pos_limits"].cpu()
joint_stiffness = weight["joint_stiffness"].cpu()
joint_damping = weight["joint_damping"].cpu()
action_offset = weight["action_offset"].cpu()
action_scale = weight["action_scale"].cpu()

device = torch.device("cuda:0")

obs_normalizer = Normalizer((124,)).to(device)
actor = Actor(124, 23).to(device)

obs_normalizer.load_state_dict(normalizer_weight)
actor.load_state_dict(actor_weight)
obs_normalizer.eval()
actor.eval()

@torch.no_grad()
def get_action(obs_batch:torch.Tensor, determine:bool=False):
    obs_batch = obs_normalizer(obs_batch)
    actor_step:StochasticContinuousPolicyStep = actor(obs_batch)
    action = actor_step.action
    if determine:
        action = actor_step.mean
    
    return action.cpu()

env = MujocoEnv(1/200, 4, kp=joint_stiffness,
                kd=joint_damping, effort_limits=joint_effort_limits, joint_pos_limits = joint_pos_limits,
                action_offset=action_offset,
                action_scale=action_scale, expert_motion_file="env/assests/punch.npz", render=True)

obs = env.reset()

def wait_for_next_step():
    while True:
        key = input("Press n for next step (q to quit): ").strip().lower()
        if key == "n":
            return True
        if key in {"q", "quit"}:
            return False

for _ in range(1000):
    #if not wait_for_next_step():
    #    break
    #print(obs)
    action = get_action(obs.to(device), True)
    #print("Action:")
    #print(action)
    #print(action)
    obs = env.step(action)

env.close()
