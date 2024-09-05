from isaacgym import gymapi

# Initialize gym
gym = gymapi.acquire_gym()

# Configure sim parameters
sim_params = gymapi.SimParams()
sim_params.dt = 1 / 60.0
sim_params.substeps = 2
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
sim_params.physx.solver_type = 1
sim_params.physx.num_position_iterations = 8
sim_params.physx.num_velocity_iterations = 1

# Create simulator
sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create a viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())
cam_pos = gymapi.Vec3(-1, 0, 2)
cam_target = gymapi.Vec3(0, 0, 1)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)
# Create an environment
env = gym.create_env(sim, gymapi.Vec3(-1.0, -1.0, 0.0), gymapi.Vec3(1.0, 1.0, 2.0), 1)

# Load URDF model
asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True  # or False depending on the model
asset = gym.load_asset(sim, "/home/shuof/work_project/TeleVision/assets/hu_v1",
                       "right_gripper.xml", asset_options)
# asset = gym.load_asset(sim, "/home/shuof/work_project/TeleVision/assets/inspire_hand",
#                        "inspire_hand_right.urdf", asset_options)

plane_params = gymapi.PlaneParams()
plane_params.distance = 0.0
plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
gym.add_ground(sim, plane_params)

# Add model to the environment
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0, 0, 1)
pose.r = gymapi.Quat.from_euler_zyx(0, 0, 0)
actor_handle = gym.create_actor(env, asset, pose, "robot", 0, 1)

# Simulation loop
while not gym.query_viewer_has_closed(viewer):
    # Step the physics
    gym.simulate(sim)
    gym.fetch_results(sim, True)

    # Update the viewer
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)

# Clean up
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)
