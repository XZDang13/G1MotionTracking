import torch
from Ref2Act.sim2sim import MujocoEnv

from RLAlg.normalizer import Normalizer
from RLAlg.nn.steps import StochasticContinuousPolicyStep
from model import Actor

def wait_for_next_step():
            while True:
                key = input("Press n for next step (q to quit): ").strip().lower()
                if key == "n":
                    return True
                if key in {"q", "quit"}:
                    return False

class Sim2Sim:
    def __init__(self, weight_path, motion_path):

        weight = torch.load(weight_path)

        normalizer_weight = weight["actor_norm"]
        actor_weight = weight["actor"]
        joint_effort_limits = weight["joint_effort_limits"].cpu()
        joint_pos_limits = weight["joint_pos_limits"].cpu()
        joint_stiffness = weight["joint_stiffness"].cpu()
        joint_damping = weight["joint_damping"].cpu()
        action_offset = weight["action_offset"].cpu()
        action_scale = weight["action_scale"].cpu()

        self.device = torch.device("cuda:0")

        self.obs_normalizer = Normalizer((124,)).to(self.device)
        self.actor = Actor(124, 23).to(self.device)

        self.obs_normalizer.load_state_dict(normalizer_weight)
        self.actor.load_state_dict(actor_weight)
        self.obs_normalizer.eval()
        self.actor.eval()

        self.env = MujocoEnv(1/200, 4, kp=joint_stiffness,
                kd=joint_damping, effort_limits=joint_effort_limits, joint_pos_limits = joint_pos_limits,
                action_offset=action_offset,
                action_scale=action_scale, expert_motion_file=motion_path, render=True)

    @torch.no_grad()
    def get_action(self, obs_batch:torch.Tensor, determine:bool=False):
        obs_batch = self.obs_normalizer(obs_batch)
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        if determine:
            action = actor_step.mean
        
        return action.cpu()

    def run(self):
        obs = self.env.reset()


        for _ in range(2000):
            action = self.get_action(obs.to(self.device), True)
            
            obs = self.env.step(action)
        self.env.close()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", choices=["handshake", "jab",
                                             "walk", "jump"],
                                            default="handshake", help="motion name")
    args = parser.parse_args()

    if args.motion == "handshake":
        weight_path = "handshake.pth"
        motion_path = "env/assests/handshake.npz"
    elif args.motion == "jab":
        weight_path = "jab.pth"
        motion_path = "env/assests/jab.npz"
    elif args.motion == "walk":
        weight_path = "walk.pth"
        motion_path = "env/assests/walk.npz"
    elif args.motion == "jump":
        weight_path = "jump.pth"
        motion_path = "env/assests/jump.npz"

    sim2sim = Sim2Sim(weight_path, motion_path)
    sim2sim.run()