from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import math
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from TeleVision import OpenTeleVision
from Preprocessor import VuerPreprocessor
from constants_vuer import tip_indices
from dex_retargeting.retargeting_config import RetargetingConfig
from pytransform3d import rotations

from pathlib import Path
import argparse
import time
import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

from env import Env

class VuerTeleop:
    def __init__(self, config_file_path):
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0]-self.crop_size_h, self.resolution[1]-2*self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0], 2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True, size=np.prod(self.img_shape) * np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3), dtype=np.uint8, buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.tv = OpenTeleVision(self.resolution_cropped, self.shm.name, image_queue, toggle_streaming)
        self.processor = VuerPreprocessor()

        RetargetingConfig.set_default_urdf_dir('../assets')
        with Path(config_file_path).open('r') as f:
            cfg = yaml.safe_load(f)
        left_retargeting_config = RetargetingConfig.from_dict(cfg['left'])
        right_retargeting_config = RetargetingConfig.from_dict(cfg['right'])
        self.left_retargeting = left_retargeting_config.build()
        self.right_retargeting = right_retargeting_config.build()

    def step(self):
        head_mat, left_wrist_mat, right_wrist_mat, left_hand_mat, right_hand_mat = self.processor.process(self.tv)

        left_wrist_mat = self.wrist_rotate(left_wrist_mat,[0,0,1,-math.pi/2])
        right_wrist_mat = self.wrist_rotate(right_wrist_mat,[0,0,1,-math.pi/2])
        right_wrist_mat = self.wrist_rotate(right_wrist_mat, [1, 0, 0, -math.pi])
        # left_wrist_mat = self.wrist_rotate(left_wrist_mat, [1, 0, 0, -math.pi / 2])

        head_rmat = head_mat[:3, :3]

        left_pose = np.concatenate([left_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                    rotations.quaternion_from_matrix(left_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        right_pose = np.concatenate([right_wrist_mat[:3, 3] + np.array([-0.6, 0, 1.6]),
                                     rotations.quaternion_from_matrix(right_wrist_mat[:3, :3])[[1, 2, 3, 0]]])
        left_qpos = self.left_retargeting.retarget(left_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]
        right_qpos = self.right_retargeting.retarget(right_hand_mat[tip_indices])[[4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3]]

        left_dist = self.cla_dist(left_hand_mat)
        right_dist = self.cla_dist(right_hand_mat)

        return head_rmat, left_pose, right_pose, left_qpos, right_qpos, left_dist, right_dist
    def wrist_rotate(self,wrist_mat,vec):
        mat = np.eye(4)
        rotate_mat = rotations.matrix_from_axis_angle(vec)
        mat[:3,:3] = rotate_mat
        return wrist_mat@mat
    def cla_dist(self,hand_mat):
        thumb_tip_pos = hand_mat[4][:3]
        index_tip_pos = hand_mat[9][:3]
        dist = np.sqrt(np.sum(np.square(thumb_tip_pos-index_tip_pos)))
        return dist

class gripperEnv(Env):
    def __init__(self,print_freq=False):
        super().__init__(print_freq)
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset_object")
    def _add_grippers(self,env,env_index):
        asset_root = "../assets"
        left_asset_path = "hu_v1/left_gripper.xml"
        # left_asset_path = "hu_v1.xml"
        right_asset_path = "hu_v1/right_gripper.xml"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(left_asset)

        self.dof_dict = {value: index
                         for index, value in enumerate(self.gym.get_asset_dof_names(left_asset))}


        # left
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, -0.2, 1.4)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        left_handle = self.gym.create_actor(env, left_asset, pose, 'left', env_index, 1)
        self.gym.set_actor_dof_states(env, left_handle, np.zeros(self.num_dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        left_idx = self.gym.get_actor_index(env, left_handle, gymapi.DOMAIN_SIM)

        # right
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.6, 0.2, 1.4)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        right_handle = self.gym.create_actor(env, right_asset, pose, 'right', env_index, 1)
        self.gym.set_actor_dof_states(env, right_handle, np.zeros(self.num_dof, gymapi.DofState.dtype),
                                      gymapi.STATE_ALL)
        right_idx = self.gym.get_actor_index(env, right_handle, gymapi.DOMAIN_SIM)

        # Retrieves buffer for Actor root states. The buffer has shape (num_actors, 13).
        # State for each actor root contains position([0:3]), rotation([3:7])
        # linear velocity([7:10]), and angular velocity([10:13]).
        self.root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(self.root_state_tensor)
        self.left_root_states = self.root_states[left_idx]
        self.right_root_states = self.root_states[right_idx]

        # props = self.gym.get_actor_dof_properties(env, left_handle)
        # props["driveMode"][:] = gymapi.DOF_MODE_POS
        # self.gym.set_actor_dof_properties(env, left_handle, props)
        #
        # props = self.gym.get_actor_dof_properties(env, right_handle)
        # props["driveMode"][:] = gymapi.DOF_MODE_POS
        # self.gym.set_actor_dof_properties(env, right_handle, props)

        return left_handle,right_handle
    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(-1, 0, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
    def _set_gripper_root_states(self,env_dix, left_pose, right_pose):
        self.left_root_states[0:7] = torch.tensor(left_pose, dtype=float)
        self.right_root_states[0:7] = torch.tensor(right_pose, dtype=float)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
    def _set_gripper_dof_states(self,env_idx,left_dist,right_dist):
        env = self.env_handles[env_idx]
        left_gripper_handle = self.left_gripper_handles[env_idx]
        right_gripper_handle = self.right_gripper_handles[env_idx]

        # 左边 gripper 的左边joint 范围为0～0.044 index 为0，右边joint 范围为-0.044～0 index 为1
        left_states = np.array([left_dist/2,-left_dist/2],dtype=gymapi.DofState.dtype)
        self.gym.set_actor_dof_states(env, left_gripper_handle, left_states, gymapi.STATE_POS)

        # 右边 gripper 的左边joint 范围为0～0.044 index 为0，右边joint 范围为-0.044～0 index 为1
        right_states = np.array([right_dist/2,-right_dist/2], dtype=gymapi.DofState.dtype)
        self.gym.set_actor_dof_states(env, right_gripper_handle, right_states, gymapi.STATE_POS)
    def _set_camera_pose(self,env_idx,head_rmat):
        env = self.env_handles[env_idx]
        left_camera_handle = self.left_cam_handles[env_idx]
        right_camera_handle = self.right_cam_handles[env_idx]

        curr_lookat_offset = self.cam_lookat_offset @ head_rmat.T
        curr_left_offset = self.left_cam_offset @ head_rmat.T
        curr_right_offset = self.right_cam_offset @ head_rmat.T

        self.gym.set_camera_location(left_camera_handle,
                                     env,
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_left_offset + curr_lookat_offset)))
        self.gym.set_camera_location(right_camera_handle,
                                     env,
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset)),
                                     gymapi.Vec3(*(self.cam_pos + curr_right_offset + curr_lookat_offset)))
    def reset_object(self,env_idx,p):
        env = self.env_handles[env_idx]
        cube_handle = self.cube_handles[env_idx]

        cube_idx = self.gym.get_actor_index(env, cube_handle, gymapi.DOMAIN_SIM)

        root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        root_states = gymtorch.wrap_tensor(root_state_tensor)
        cube_root_state = root_states[cube_idx]

        # cube_root_state[0:7] = torch.tensor(np.array([1, 0, 2,0,0,0,0]), dtype=float)
        cube_root_state[0:7] = torch.tensor(p, dtype=float)
        # Apply the new root state to the cuber
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(root_states))

    def _check_key(self):
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.action == "reset_object" and evt.value > 0:
                # print("Resetting cube position")
                self.reset_object(env_idx=0)
                # self._add_object(self.env_handles[0],env_index=0)
    def step(self, head_rmat, left_pose, right_pose,left_dist, right_dist):

        self._set_gripper_root_states(0,left_pose, right_pose)
        self._set_gripper_dof_states(env_idx=0,left_dist=left_dist,right_dist=right_dist)
        # self._check_key()
        # self.reset_object(env_idx=0,p=left_pose)
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)



        self._set_camera_pose(0,head_rmat)
        left_image,right_image = self.get_eye_cameras_image(env_idx=0)

        self.gym.draw_viewer(self.viewer, self.sim, True)
        self.gym.sync_frame_time(self.sim)

        return left_image,right_image
    def end(self):
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
if __name__ == '__main__':
    teleoperator = VuerTeleop('inspire_hand.yml')
    sim = gripperEnv()
    try:
        while True:
            head_rmat, left_pose, right_pose, left_qpos, right_qpos,left_dist, right_dist = teleoperator.step()

            left_img, right_img = sim.step(head_rmat, left_pose, right_pose, left_dist, right_dist)
            np.copyto(teleoperator.img_array, np.hstack((left_img, right_img)))
    except KeyboardInterrupt:
        sim.end()
        exit(0)
