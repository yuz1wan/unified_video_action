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
from einops import rearrange
from omegaconf import open_dict
from unified_video_action.workspace.base_workspace import BaseWorkspace
from unified_video_action.utils.load_env import load_env_runner
from unified_video_action.utils.realsense import CameraD400
from unified_video_action.common.pytorch_util import dict_apply
import cv2

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

def plot_keypoints(video, keypoints, device, color=(255, 0, 0)):
    color = torch.tensor(color).to(device)
    _B, _C, _T, _H, _W = video.size()         
    norm_scale = torch.tensor([_W, _H]).to(device)
    
    # select frames from points to _T
    selected_index = torch.linspace(0, keypoints.size(1) - 1, _T, dtype=torch.int64)
    points = keypoints[:, selected_index, :6].to(device)
    wrist = points[:, :, 0:2] * norm_scale
    thumb = points[:, :, 2:4] * norm_scale
    index = points[:, :, 4:6] * norm_scale
    
    for i in range(_B):
        for j in range(_T):
            for dx in range(-2, 3):  # Adjust the range to control the size
                for dy in range(-2, 3):
                    x_wrist = (wrist[i, j, 0] + dx).long()
                    y_wrist = (wrist[i, j, 1] + dy).long()
                    x_thumb = (thumb[i, j, 0] + dx).long()
                    y_thumb = (thumb[i, j, 1] + dy).long()
                    x_index = (index[i, j, 0] + dx).long()
                    y_index = (index[i, j, 1] + dy).long()

                    if 0 <= x_wrist < _W and 0 <= y_wrist < _H:
                        video[i, :, j, y_wrist, x_wrist] = color.type(torch.uint8)
                    if 0 <= x_thumb < _W and 0 <= y_thumb < _H:
                        video[i, :, j, y_thumb, x_thumb] = color.type(torch.uint8)
                    if 0 <= x_index < _W and 0 <= y_index < _H:
                        video[i, :, j, y_index, x_index] = color.type(torch.uint8)
                        
    return video

class RealWorldRunner:
    def __init__(self, policy, cfg, cam_sn=""):
        self.policy = policy

        self.n_obs_steps = 16 # if cfg.task.keypoints.n_obs_steps is None else cfg.task.keypoints.n_obs_steps
        self.n_action_steps = 8 # if cfg.task.keypoints.n_action_steps is None else cfg.task.keypoints.n_action_steps
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
        color = color.astype(np.float32)
        # convert to (C, H, W)
        color = np.transpose(color, (2, 0, 1))
        # normalize to [0, 1]
        color = color / 255.0
        return color # (3, 96, 96)
        
    def reset(self):
        obs = self.get_current_img()
        self.obs = deque([obs], maxlen=self.n_obs_steps + 1)
        observation = self._get_obs(self.n_obs_steps)
        return observation
    
    def step(self):
        obs = self.get_current_img()
        self.obs.append(obs)
        obs = self._get_obs(self.n_obs_steps)
        obs = np.expand_dims(obs, axis=0)
        
        np_obs_dict ={}
        np_obs_dict['image'] = obs
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
        
        return obs, action
        
    def _get_obs(self, n_obs_steps=1):
        """
        Output (n_obs_steps,) + obs_shape
        """
        if self.obs is None or len(self.obs) == 0:
            raise ValueError("No observations available. Please call reset() first.")
        return stack_last_n_obs(self.obs, n_obs_steps)

@click.command()
@click.option("-c", "--checkpoint", required=True)
@click.option("-o", "--output_dir", required=True)
@click.option("-d", "--device", default="cuda:0")
@click.option("-s", "--cam_sn", default="233522075695")
def main(checkpoint, output_dir, device, cam_sn):

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

    env_runners = RealWorldRunner(policy, cfg, cam_sn) # replace with your camera serial number
    env_runners.reset()

    # Use keyboard to control the process
    # Press 'q' to quit
    # Press 'r' to reset
    preds = []
    while True:
        # obs: B, T, C, H, W
        # action: B, T, Da
        obs, action = env_runners.step()

        last_frame = obs[:, [-1],: , :, :] # 1, 1, C, H, W
        real = last_frame * 255
        real = rearrange(real, "b t c h w -> b c t h w")

        traj = action[:, [-1], :]
        vis_pred = plot_keypoints(torch.from_numpy(real), torch.from_numpy(traj), device)
        vis_pred = rearrange(vis_pred, "b c t h w -> b t h w c")[0, 0].cpu().numpy().astype(np.uint8)
        vis_pred = cv2.cvtColor(vis_pred, cv2.COLOR_RGB2BGR)
        cv2.imshow("Pred Action", vis_pred)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv2.waitKey(1) & 0xFF == ord('r'):
            env_runners.reset()
        else:
            continue


if __name__ == "__main__":
    main()
