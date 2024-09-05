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

class Env:
    def __init__(self,print_freq=False):
        self.print_freq = print_freq

        self.cam_lookat_offset = np.array([1, 0, 0])
        self.left_cam_offset = np.array([0, 0.033, 0])
        self.right_cam_offset = np.array([0, -0.033, 0])
        self.cam_pos = np.array([-0.6, 0, 1.6])

        self.gym = gymapi.acquire_gym()

        self.create_sim()
    def create_sim(self):
        sim_params = self.set_sim_parameters()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        self._create_ground_plane()
        self._create_envs()
        self._create_viewer()
    def set_sim_parameters(self):
        sim_params = gymapi.SimParams()

        sim_params.dt = 1 / 60
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)

        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        sim_params.physx.max_gpu_contact_pairs = 8388608
        sim_params.physx.contact_offset = 0.002
        sim_params.physx.friction_offset_threshold = 0.001
        sim_params.physx.friction_correlation_distance = 0.0005
        sim_params.physx.rest_offset = 0.0
        sim_params.physx.use_gpu = True
        sim_params.use_gpu_pipeline = False
        return sim_params

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.distance = 0.0
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs=1, env_spacing=1.25, num_per_row=1):
        env_lower = gymapi.Vec3(-env_spacing, -env_spacing, 0)
        env_upper = gymapi.Vec3(env_spacing, env_spacing, env_spacing)
        np.random.seed(0)

        self.env_handles = []
        self.left_gripper_handles = []
        self.right_gripper_handles = []
        self.cube_handles = []
        self.table_handles = []
        self.left_cam_handles = []
        self.right_cam_handles = []
        # create and populate the environments
        for i in range(num_envs):
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper,num_per_row)
            self.env_handles.append(env_handle)

            table_handle = self._add_table(env_handle,i)
            self.table_handles.append(table_handle)

            cube_handle = self._add_object(env_handle,i)
            self.cube_handles.append(cube_handle)

            left_handle,right_handle = self._add_grippers(env_handle,i)
            self.left_gripper_handles.append(left_handle)
            self.right_gripper_handles.append(right_handle)

            left_camera_handle,right_camera_handle = self._create_eye_cameras(env_handle)
            self.left_cam_handles.append(left_camera_handle)
            self.right_cam_handles.append(right_camera_handle)
    def _add_grippers(self,env,env_index):
        asset_root = "../assets"
        left_asset_path = "inspire_hand/inspire_hand_left.urdf"
        # left_asset_path = "hu_v1.xml"
        right_asset_path = "inspire_hand/inspire_hand_right.urdf"
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        left_asset = self.gym.load_asset(self.sim, asset_root, left_asset_path, asset_options)
        right_asset = self.gym.load_asset(self.sim, asset_root, right_asset_path, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(left_asset)

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

        return left_handle,right_handle


    def _add_table(self,env,env_index):
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, 0.8, 0.8, 0.1, table_asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0, 0, 1.2)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        table_handle = self.gym.create_actor(env, table_asset, pose, 'table', env_index,)
        color = gymapi.Vec3(0.5, 0.5, 0.5)
        self.gym.set_rigid_body_color(env, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        return table_handle
    def _add_object(self,env,env_index):
        cube_asset_options = gymapi.AssetOptions()
        cube_asset_options.density = 10
        cube_asset = self.gym.create_box(self.sim, 0.05, 0.05, 0.05, cube_asset_options)

        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(-0.3, 0, 1.25)
        pose.r = gymapi.Quat(0, 0, 0, 1)
        cube_handle = self.gym.create_actor(env, cube_asset, pose, 'cube', env_index,)
        color = gymapi.Vec3(1, 0.5, 0.5)
        self.gym.set_rigid_body_color(env, cube_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, color)
        return cube_handle

    def _create_eye_cameras(self,env):
        # create left 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        left_camera_handle = self.gym.create_camera_sensor(env, camera_props)
        self.gym.set_camera_location(left_camera_handle,
                                     env,
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.left_cam_offset + self.cam_lookat_offset)))

        # create right 1st preson viewer
        camera_props = gymapi.CameraProperties()
        camera_props.width = 1280
        camera_props.height = 720
        right_camera_handle = self.gym.create_camera_sensor(env, camera_props)
        self.gym.set_camera_location(right_camera_handle,
                                     env,
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset)),
                                     gymapi.Vec3(*(self.cam_pos + self.right_cam_offset + self.cam_lookat_offset)))

        return left_camera_handle,right_camera_handle
    def _create_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(-1, 0, 2)
        cam_target = gymapi.Vec3(0, 0, 1)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def get_eye_cameras_image(self,env_idx,show=False):
        env = self.env_handles[env_idx]
        left_camera_handle = self.left_cam_handles[env_idx]
        right_camera_handle = self.right_cam_handles[env_idx]
        left_image = self.gym.get_camera_image(self.sim,env, left_camera_handle, gymapi.IMAGE_COLOR)
        right_image = self.gym.get_camera_image(self.sim,env, right_camera_handle, gymapi.IMAGE_COLOR)
        left_image = left_image.reshape(left_image.shape[0], -1, 4)[..., :3]
        right_image = right_image.reshape(right_image.shape[0], -1, 4)[..., :3]
        if show:
            img = np.concatenate([left_image,right_image],axis=1)
            plt.imshow(img)
            plt.axis('off')
            plt.pause(0.001)
        return left_image,right_image
    def run(self):

        while not self.gym.query_viewer_has_closed(self.viewer):
            # step the physics
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)

            # update the viewer
            self.gym.step_graphics(self.sim)
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)

            self.get_eye_cameras_image(env_idx=0,show=True)

            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

if __name__ == '__main__':
    sim = Env(print_freq=100)
    sim.run()
