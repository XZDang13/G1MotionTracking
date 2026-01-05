import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from tqdm import trange
import gymnasium
import torch
import numpy as np

from RLAlg.normalizer import Normalizer
from RLAlg.buffer.replay_buffer import ReplayBuffer, compute_gae
from RLAlg.nn.steps import StochasticContinuousPolicyStep, ValueStep
from RLAlg.alg.ppo import PPO
from RLAlg.logger import WandbLogger

from model import Actor, Critic
from env.cfg import G1JabTrainingEnv

class Trainer:
    def __init__(self):
        self.cfg = G1JabTrainingEnv()
        self.env_name = "G1MotionTracking-v0"

        self.env = gymnasium.make(self.env_name, cfg=self.cfg)

        print(self.cfg.scene.num_envs)

        #default_obs_dim = self.cfg.observation_space
        default_obs_dim = self.cfg.privilege_observation_space
        privilege_obs_dim = self.cfg.privilege_observation_space
        action_dim = self.cfg.action_space

        self.device = self.env.unwrapped.device

        self.actor = Actor(default_obs_dim, action_dim).to(self.device)
        self.critic = Critic(privilege_obs_dim).to(self.device)
        self.actor_obs_normalizer = Normalizer((default_obs_dim,)).to(self.device)
        self.critic_obs_normalizer = Normalizer((privilege_obs_dim,)).to(self.device)

        self.ac_optimizer = torch.optim.Adam(
            [
                {'params': self.actor.parameters(),
                 "name": "actor"},
                 {'params': self.critic.parameters(),
                 "name": "critic"},
            ],
            lr=1e-4
        )

        self.steps = 20

        self.rollout_buffer = ReplayBuffer(
            self.cfg.scene.num_envs,
            self.steps
        )

        self.batch_keys = ["observations",
                           "privilege_observations",
                           "actions",
                           "log_probs",
                           "rewards",
                           "values",
                           "returns",
                           "advantages"
                        ]

        self.rollout_buffer.create_storage_space("observations", (default_obs_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("privilege_observations", (privilege_obs_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("actions", (action_dim,), torch.float32)
        self.rollout_buffer.create_storage_space("log_probs", (), torch.float32)
        self.rollout_buffer.create_storage_space("rewards", (), torch.float32)
        self.rollout_buffer.create_storage_space("values", (), torch.float32)
        self.rollout_buffer.create_storage_space("dones", (), torch.float32)

        self.expert_motion_buffer = ReplayBuffer(
            500,
            400
        )

        self.ep_ret = torch.zeros(self.cfg.scene.num_envs, device=self.device)
        self.ep_len = torch.zeros(self.cfg.scene.num_envs, device=self.device)

        self.global_step = 0   
        WandbLogger.init_project("Mimic", f"G1_Jab")
        
    @torch.no_grad()
    def get_action(self, actorobs_batch:torch.Tensor, criticobs_batch:torch.Tensor, determine:bool=False):
        actor_obs_batch = self.actor_obs_normalizer(actorobs_batch)
        actor_step:StochasticContinuousPolicyStep = self.actor(actor_obs_batch)
        action = actor_step.action
        log_prob = actor_step.log_prob
        if determine:
            action = actor_step.mean
        
        critic_obs_batch = self.critic_obs_normalizer(criticobs_batch)
        critic_step:ValueStep = self.critic(critic_obs_batch)
        value = critic_step.value

        return action, log_prob, value
    
    
    def rollout(self, obs):
        self.actor.eval()
        self.critic.eval()
        for _ in range(self.steps):
            self.global_step += 1
            #default_obs = obs["default"]
            default_obs = obs["privilege"]
            privilege_obs = obs["privilege"]
            action, log_prob, value = self.get_action(default_obs, privilege_obs)
            next_obs, task_reward, terminate, timeout, info = self.env.step(action)
            
            reward = task_reward
            #reward = task_reward

            self.ep_ret += reward
            self.ep_len += 1

            done = terminate | timeout
            
            if done.any():
                log_ep_ret = self.ep_ret[done]
                log_ep_len = self.ep_len[done]

                log_ep_ret = torch.mean(log_ep_ret).item()
                log_ep_len = torch.mean(log_ep_len).item()

                step_info = {}
                step_info['step/mean_returns'] = log_ep_ret
                step_info['step/mean_length'] = log_ep_len

                self.ep_ret[done] = 0.0
                self.ep_len[done] = 0.0

                WandbLogger.log_metrics(step_info, self.global_step)

            records = {
                "observations": default_obs,
                "privilege_observations": privilege_obs,
                "actions": action,
                "log_probs": log_prob,
                "rewards": reward,
                "values": value,
                "dones": done
            }

            self.rollout_buffer.add_records(records)

            obs = next_obs


        #last_default_obs = obs["default"]
        last_default_obs = obs["privilege"]
        last_privilege_obs = obs["privilege"]
        _, _, last_value = self.get_action(last_default_obs, last_privilege_obs)
        returns, advantages = compute_gae(
            self.rollout_buffer.data["rewards"],
            self.rollout_buffer.data["values"],
            self.rollout_buffer.data["dones"],
            last_value,
            0.99,
            0.95
        )
        

        self.rollout_buffer.add_storage("returns", returns)
        self.rollout_buffer.add_storage("advantages", advantages)

        self.actor.train()
        self.critic.train()
        return obs
    
    def update(self):
        policy_loss_buffer = []
        value_loss_buffer = []
        entropy_buffer = []
        kl_divergence_buffer = []

        for i in range(5):
            for batch in self.rollout_buffer.sample_batchs(self.batch_keys, 4096*10):
                obs_batch = batch["observations"].to(self.device)
                privilege_obs_batch = batch["privilege_observations"].to(self.device)
                action_batch = batch["actions"].to(self.device)
                log_prob_batch = batch["log_probs"].to(self.device)
                value_batch = batch["values"].to(self.device)
                return_batch = batch["returns"].to(self.device)
                advantage_batch = batch["advantages"].to(self.device)

                obs_batch = self.actor_obs_normalizer(obs_batch, i==0)
                privilege_obs_batch = self.critic_obs_normalizer(privilege_obs_batch, i==0)

                policy_loss_dict = PPO.compute_policy_loss(self.actor,
                                                           log_prob_batch,
                                                           obs_batch,
                                                           action_batch,
                                                           advantage_batch,
                                                           0.2,
                                                           0.0)
                
                policy_loss = policy_loss_dict["loss"]
                entropy = policy_loss_dict["entropy"]
                kl_divergence = policy_loss_dict["kl_divergence"]

                value_loss_dict = PPO.compute_clipped_value_loss(self.critic,
                                                    privilege_obs_batch,
                                                    value_batch,
                                                    return_batch,
                                                    0.2)
                
                value_loss = value_loss_dict["loss"]

                ac_loss = policy_loss - entropy * 0.001 + value_loss * 2.5

                self.ac_optimizer.zero_grad(set_to_none=True)
                ac_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
                self.ac_optimizer.step()
                

                policy_loss_buffer.append(policy_loss.item())
                value_loss_buffer.append(value_loss.item())
                entropy_buffer.append(entropy.item())
                kl_divergence_buffer.append(kl_divergence.item())

        avg_policy_loss = np.mean(policy_loss_buffer)
        avg_value_loss = np.mean(value_loss_buffer)
        avg_entropy = np.mean(entropy_buffer)
        avg_kl_divergence = np.mean(kl_divergence_buffer)
        
        train_info = {
            "update/avg_policy_loss": avg_policy_loss,
            "update/avg_value_loss": avg_value_loss,
            "update/avg_entropy": avg_entropy,
            "update/avg_kl_divergence": avg_kl_divergence
        }

        WandbLogger.log_metrics(train_info, self.global_step)

    def train(self):
        obs, _ = self.env.reset()
        for epoch in trange(1000):
            obs = self.rollout(obs)
            self.update()
        self.env.close()

        torch.save(
            [self.actor_obs_normalizer.state_dict(), self.actor.state_dict(), self.critic.state_dict()],
            "weight.pth"
        )

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()
    simulation_app.close()