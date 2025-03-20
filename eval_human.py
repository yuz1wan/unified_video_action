import sys

sys.stdout = open(sys.stdout.fileno(), mode="w", buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode="w", buffering=1)
import numpy as np
import os
import pathlib
import click
import hydra
import torch
import dill
import wandb
import json
import random
from collections import defaultdict, deque
from omegaconf import open_dict
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner
from unified_video_action.utils.realsense import CameraD400
from unified_video_action.common.pytorch_util import dict_apply
import cv2


@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")

def stack_last_n_obs(all_obs, n_steps):
    assert len(all_obs) > 0
    all_obs = list(all_obs)
    result = np.zeros((n_steps,) + all_obs[-1].shape, dtype=all_obs[-1].dtype)
    start_idx = -min(n_steps, len(all_obs))
    result[start_idx:] = np.array(all_obs[start_idx:])
    if n_steps > len(all_obs):
        # pad
        result[:start_idx] = result[start_idx]
    return result

class RealWorldRunner:
    def __init__(self, policy, cfg, cam_sn=""):
        self.policy = policy
        self.n_obs_steps = 16 if cfg.task.keypoints.n_obs_steps is None else cfg.task.keypoints.n_obs_steps
        self.n_action_steps = 8 if cfg.task.keypoints.n_action_steps is None else cfg.task.keypoints.n_action_steps
        self.cfg = cfg
        if cam_sn == "":
            raise ValueError("Please provide a serial number for the camera.")
        else:
            self.camera = CameraD400(cam_sn)
        self.obs = None
        
    def get_current_img(self):
        if self.camera is None:
            raise ValueError("Camera not initialized.")
        color, depth = self.camera.get_data()
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        color = cv2.resize(color, (self.cfg.task.shape_meta.image_resolution, self.cfg.task.shape_meta.image_resolution))
        color = color.astype(np.float32)
        # convert to (C, H, W)
        color = np.transpose(color, (2, 0, 1))
        # normalize to [0, 1]
        color = color / 255.0
        return color # (96, 96, 3)
        
    def reset(self):
        obs = self.get_current_img()
        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        observation = self._get_obs(self.n_obs_steps)
        return observation
    
    def step(self):
        obs = self.get_current_img()
        self.obs.append(obs)
        observation = self._get_obs(self.n_obs_steps)
        
        np_obs_dict = dict(observation)
        obs_dict = dict_apply(
            np_obs_dict, lambda x: torch.from_numpy(x).to(device=self.policy.device)
        )
        with torch.no_grad():
            action_dict = self.policy.predict_action(obs_dict)
            
        # device_transfer
        np_action_dict = dict_apply(
            action_dict, lambda x: x.detach().to("cpu").numpy()
        )
        action = np_action_dict["action"]  # (B, 8, 2)
        print("action", action)
        
        return observation
        
    def _get_obs(self, n_obs_steps=1):
        """
        Output (n_obs_steps,) + obs_shape
        """
        if self.obs is None or len(self.obs) == 0:
            raise ValueError("No observations available. Please call reset() first.")
        return stack_last_n_obs(self.obs, n_obs_steps)

        
def main(checkpoint, output_dir, device):

    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # load checkpoint
    payload = torch.load(open(checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # set seed
    seed = cfg.training.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    with open_dict(cfg):
        cfg.output_dir = output_dir
        
    # configure workspace
    cls = hydra.utils.get_class(cfg.model._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace

    print("Loaded checkpoint from %s" % checkpoint)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    
    
    # get policy from workspace
    policy = workspace.ema_model
    policy.to(device)
    policy.eval()

    env_runners = RealWorldRunner(policy, cfg, cam_sn="D400") # replace with your camera serial number
    env_runners.reset()

    # Use keyboard to control the process
    # Press 'q' to quit
    # Press 'r' to reset
    while True:
        obs = env_runners.step()
        action = policy(obs)
        print("action", action)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            env_runners.reset()
        else:
            continue


if __name__ == "__main__":
    main()
