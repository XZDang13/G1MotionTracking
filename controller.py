import numpy as np
import time
import torch
import struct

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC

from motion_lib import MotionLib
from config import G1Config
from model import Actor
from RLAlg.normalizer import Normalizer
from RLAlg.nn.steps import StochasticContinuousPolicyStep

def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion.

    Args:
        q (torch.Tensor): Quaternion [w, x, y, z]
        v (torch.Tensor): Vector to rotate

    Returns:
        torch.Tensor: Rotated vector
    """
    q_w = q[0]
    q_vec = q[1:4]
    a = v * (2.0 * q_w ** 2 - 1.0)
    b = torch.cross(q_vec, v, dim=-1) * q_w * 2.0
    c = q_vec * (torch.dot(q_vec, v)) * 2.0
    return a - b + c

class MotorMode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints


def create_damping_cmd(cmd: LowCmdHG):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 8
        cmd.motor_cmd[i].tau = 0


def create_zero_cmd(cmd: LowCmdHG):
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0
        self.ly = 0
        self.rx = 0
        self.ry = 0
        self.button = [0] * 16

    def set(self, data):
        # wireless_remote
        keys = struct.unpack("H", data[2:4])[0]
        for i in range(16):
            self.button[i] = (keys & (1 << i)) >> i
        self.lx = struct.unpack("f", data[4:8])[0]
        self.rx = struct.unpack("f", data[8:12])[0]
        self.ry = struct.unpack("f", data[12:16])[0]
        self.ly = struct.unpack("f", data[20:24])[0]


def init_cmd_hg(cmd: LowCmdHG, mode_machine: int, mode_pr: int):
    cmd.mode_machine = mode_machine
    cmd.mode_pr = mode_pr
    size = len(cmd.motor_cmd)
    for i in range(size):
        cmd.motor_cmd[i].mode = 1
        cmd.motor_cmd[i].q = 0
        cmd.motor_cmd[i].qd = 0
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0

class Controller:
    def __init__(self):

        self.device = torch.device("cuda:0")

        self.obs_normalizer = Normalizer((124,)).to(self.device)
        self.actor = Actor(124, 23).to(self.device)

        weight = torch.load("final.pth")

        normalizer_weight = weight["actor_norm"]
        actor_weight = weight["actor"]
        joint_effort_limits = weight["joint_effort_limits"].cpu()
        joint_pos_limits = weight["joint_pos_limits"].cpu()
        joint_stiffness = weight["joint_stiffness"].cpu()
        joint_damping = weight["joint_damping"].cpu()
        action_offset = weight["action_offset"].cpu()
        action_scale = weight["action_scale"].cpu()
        joint_names = weight["joint_names"]

        self.config = G1Config(
            policy_joints_order=joint_names,
            stiffness=joint_stiffness,
            damping=joint_damping,
            action_offset=action_offset,
            action_scale=action_scale
        )

        self.obs_normalizer.load_state_dict(normalizer_weight)
        self.actor.load_state_dict(actor_weight)
        self.obs_normalizer.eval()
        self.actor.eval()

        self.motion_lib = MotionLib("env/assests/jab.npz")
        self.time = torch.zeros(1)

        self.remote_controller = RemoteController()

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        self.mode_pr_ = MotorMode.PR
        self.mode_machine_ = 0

        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmdHG)
        self.lowcmd_publisher_.Init()

        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateHG)
        self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        self.counter = 0
        self.last_actions = torch.zeros(self.config.joint_num, dtype=torch.float32)

        self.wait_for_low_state()
        init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)

    @torch.no_grad()
    def get_action(self, obs_batch:torch.Tensor, determine:bool=False):
        obs_batch = self.obs_normalizer(obs_batch)
        actor_step:StochasticContinuousPolicyStep = self.actor(obs_batch)
        action = actor_step.action
        if determine:
            action = actor_step.mean
        
        return action.cpu()

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: LowCmdHG):
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def move_to_default_pos(self):
        print("Moving to default pos.")
        # move time 2s
        total_time = 2
        num_step = int(total_time / self.config.control_dt)

        init_pos = {}

        target_pos = {}

        reference_motion = self.motion_lib.sample_motion(self.time)
        print(self.time)

        for idx, joint_name in enumerate(self.config.joints_settings.keys()):
            init_pos[joint_name] = self.low_state.motor_state[idx].q
            target_pos[joint_name] = 0
        
        for idx, joint_name in enumerate(self.config.policy_joints_order):
            target_pos[joint_name] = reference_motion["joint_pos"][0, idx].item()

        # move to default pos
        for i in range(num_step):
            alpha = i / num_step
            for idx, joint_name in enumerate(self.config.joints_settings.keys()):
                self.low_cmd.motor_cmd[idx].q = init_pos[joint_name] * (1 - alpha) + target_pos[joint_name] * alpha
                self.low_cmd.motor_cmd[idx].qd = 0
                self.low_cmd.motor_cmd[idx].kp = self.config.pd_params[joint_name][0]
                self.low_cmd.motor_cmd[idx].kd = self.config.pd_params[joint_name][1]
                self.low_cmd.motor_cmd[idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")
        target_pos = {}
        reference_motion = self.motion_lib.sample_motion(self.time)
        for idx, joint_name in enumerate(self.config.joints_settings.keys()):
            target_pos[joint_name] = 0

        for idx, joint_name in enumerate(self.config.policy_joints_order):
            target_pos[joint_name] = reference_motion["joint_pos"][0, idx].item()


        while self.remote_controller.button[KeyMap.A] != 1:
            for idx, joint_name in enumerate(self.config.joints_settings.keys()):
                self.low_cmd.motor_cmd[idx].q = target_pos[joint_name]
                self.low_cmd.motor_cmd[idx].qd = 0
                self.low_cmd.motor_cmd[idx].kp = self.config.pd_params[joint_name][0]
                self.low_cmd.motor_cmd[idx].kd = self.config.pd_params[joint_name][1]
                self.low_cmd.motor_cmd[idx].tau = 0
            self.send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def get_joint_state(self):
        joint_states = {}

        for idx, joint_name in enumerate(self.config.joints_settings.keys()):
            joint_states[joint_name] = (self.low_state.motor_state[idx].q, self.low_state.motor_state[idx].dq)

        joint_pos = torch.zeros(self.config.joint_num, dtype=torch.float32)
        joint_vel = torch.zeros(self.config.joint_num, dtype=torch.float32)
        for idx, joint_name in enumerate(self.config.policy_joints_order):
            joint_pos[idx] = joint_states[joint_name][0]
            joint_vel[idx] = joint_states[joint_name][1]

        return joint_pos, joint_vel
    
    def get_projected_gravity(self):
        quat = torch.as_tensor(self.low_state.imu_state.quaternion, dtype=torch.float32)

        gravity_orientation = quat_rotate_inverse(quat, self.config.gravity_vector)

        return gravity_orientation

    def get_base_ang_vel(self):
        ang_vel = torch.as_tensor(self.low_state.imu_state.gyroscope, dtype=torch.float32)

        return ang_vel

    def get_motion_command(self, times):
        reference_motion = self.motion_lib.sample_motion(times)

        joint_pos = reference_motion["joint_pos"].squeeze(0)
        joint_vel = reference_motion["joint_vel"].squeeze(0)
        body_quat = reference_motion["body_quaternions"].squeeze(0)
        root_quat = body_quat[0]

        projected_gravity = quat_rotate_inverse(root_quat, self.config.gravity_vector).float()

        return joint_pos, joint_vel, projected_gravity


    def get_target_pos(self):
        self.time += self.config.control_dt
        if self.time > self.motion_lib.duration:
            self.time = torch.zeros(1)
            self.last_actions[:] = 0.0
        
        #print(self.time)
        target_joint_pos, target_joint_vel, target_projected_gravity = self.get_motion_command(self.time)

        projected_gravity = self.get_projected_gravity()
        base_ang_vel = self.get_base_ang_vel()
        joint_pos, joint_vel = self.get_joint_state()
        
        last_action = self.last_actions

        obs = torch.cat([
            target_joint_pos,
            target_joint_vel,
            target_projected_gravity,
            projected_gravity,
            base_ang_vel,
            joint_pos,
            joint_vel,
            last_action
        ]).to(self.device)

        actions = self.get_action(obs, True).squeeze(0)
        
        self.last_actions = actions.clone()

        target_pos = (self.config.action_offset + self.config.action_scale * actions).numpy()

        cmd = {}
        #reference_motion = self.motion_lib.sample_motion(torch.zeros(1))
        #for idx, joint_name in enumerate(self.config.policy_joints_order):
        #    cmd[joint_name] = reference_motion["joint_pos"][0, idx].item()

        for idx, joint_name in enumerate(self.config.policy_joints_order):
            cmd[joint_name] = target_pos[idx]

        return cmd

    def run(self):
        cmd = self.get_target_pos()

        
        for idx, joint_name in enumerate(self.config.joints_settings.keys()):
            if joint_name in cmd:
                self.low_cmd.motor_cmd[idx].q = cmd[joint_name]
                self.low_cmd.motor_cmd[idx].qd = 0
                self.low_cmd.motor_cmd[idx].kp = self.config.pd_params[joint_name][0]
                self.low_cmd.motor_cmd[idx].kd = self.config.pd_params[joint_name][1]
                self.low_cmd.motor_cmd[idx].tau = 0

        self.send_cmd(self.low_cmd)
        

        time.sleep(self.config.control_dt)

        self.counter += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    args = parser.parse_args()


    # Initialize DDS communication
    ChannelFactoryInitialize(0, args.net)

    controller = Controller()

    controller.wait_for_low_state()
    controller.zero_torque_state()

    controller.move_to_default_pos()

    controller.default_pos_state()

    while True:
        try:
            controller.run()
            # Press the select key to exit
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
        except KeyboardInterrupt:
            break
    # Enter the damping state
    create_damping_cmd(controller.low_cmd)
    controller.send_cmd(controller.low_cmd)
    
    print("Exit")
