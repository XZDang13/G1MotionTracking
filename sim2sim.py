import torch
from Ref2Act.sim2sim import MujocoEnv

from RLAlg.normalizer import Normalizer
from RLAlg.nn.steps import StochasticContinuousPolicyStep
from model import Actor

weight = torch.load("weight.pth")

normalizer_weight = weight["actor_norm"]
actor_weight = weight["actor"]
joint_stiffness = weight["joint_stiffness"].cpu()
joint_damping = weight["joint_damping"].cpu()
action_offset = weight["joint_offset"].cpu()
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

env = MujocoEnv(1/1000, 20, joint_stiffness,
                joint_damping, action_offset, action_scale,
                "env/assests/jab.npz", render=True)

obs = env.reset()

for _ in range(1000):
    print(env.get_projected_gravity())
    action = get_action(obs.to(device), True)
    obs = env.step(action)

env.close()