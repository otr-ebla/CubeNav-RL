import mujoco
import mujoco.viewer
import numpy as np
import math
import time
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
import argparse
import xml.etree.ElementTree as ET
import os
import torch
from torch.utils.tensorboard import SummaryWriter

class CubeNavigationEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}
    
    def __init__(self, num_rays = 50, render_mode=None, model_path = "xml_models/obstacles_cube_model.xml", training_mode = True):
        super().__init__()
        
        self.num_rays = num_rays
        self.training_mode = training_mode
        self.max_episode_steps = 1200  # 30 seconds (at 30[s]/0.1[s/step] = 300step)
        self.current_step = 0
        self.render_mode = render_mode
        self.model_path = model_path
        self.previous_distance = 30
        self.episode_return = 0
        self.mean_episode_return = 0
        self.episode_count = 0
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.last_episode_result = None

        self.lidar_readings = None

        self.sum_progress_reward = 0
        self.average_progress_reward = 0
        self.success_rate = 0
        self.collision_rate = 0
        self.timeout_rate = 0
        self.steps_taken = 0

        self.relative_azimuth = 0



        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        
        # Define observation space:
        # - 50 lidar readings (one for each ray)
        # - 1 value for distance to cube (x, y, z)
        # - 1 value for orientation (not used in simple version, but included for future extensions)
        self.observation_space = spaces.Box(
            low=0.0,
            high=100.0,  # or some sensible max for lidar + position/orientation values
            shape=(self.num_rays + 2,),
            dtype=np.float32
        )

        
        # Load MuJoCo model
        self.xml_model = self._load_and_modify_xml_model()
        self.model = mujoco.MjModel.from_xml_string(self.xml_model)
        self.data = mujoco.MjData(self.model)
        
        # Get cube body ID
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        
        # Get sensor IDs for the LiDAR rays
        self.lidar_sensor_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"lidar_{i}")
            for i in range(self.num_rays)
        ]
        
        # Viewer
        self.viewer = None
        if self.render_mode == "human":
            self._setup_viewer()
            
        # Reset the environment
        self.reset()
    
    def _load_and_modify_xml_model(self):
        """Load and modify the XML model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found.")
        
        tree = ET.parse(self.model_path)
        root = tree.getroot()

        cube_body = None
        for body in root.findall(".//body"):
            if body.get("name") == "cube_body":
                cube_body = body
                break
        
        if cube_body is None:
            raise ValueError("Cube body not found in the XML model.")
        
        sensor = None
        for s in root.findall(".//sensor"):
            sensor = s
            break

        if sensor is None:
            raise ValueError("Sensor element not found in the XML model.")

        # Add LiDAR sensors to the cube body
        for i in range(self.num_rays):
            angle = (2 * np.pi / self.num_rays) * i + np.pi / 2  # Start from the top
            angle = (angle + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)

            site = ET.SubElement(cube_body, "site")
            site.set("name", f"lidar_site_{i}")
            site.set("pos", f"{np.sqrt(2)*cos_angle} {np.sqrt(2)*sin_angle} -0.3")
            site.set("size", "0.05")
            site.set("rgba", "1 0 0 1")  # Red color for LiDAR sites
            site.set("zaxis", f"{cos_angle} {sin_angle} 0")

            
            rangefinder = ET.Element("rangefinder")  # Create the element
            rangefinder.set("name", f"lidar_{i}")    # Set attributes
            rangefinder.set("site", f"lidar_site_{i}")
            sensor.append(rangefinder)              # Append it to the parent element
        return ET.tostring(root, encoding="unicode")

    
    def _setup_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        # Set up camera for a top-down view
        self.viewer.cam.distance = 25.0
        self.viewer.cam.azimuth = 0.0
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.lookat[:] = [0, 0, 1]


    
    def _get_obs(self):
        # Process LiDAR readings
        self.lidar_readings = np.array([self.data.sensordata[lidar_id] for lidar_id in self.lidar_sensor_ids])

        # Get positions
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        cube_pos = self.data.xpos[cube_body_id].copy()

        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        sphere_pos = self.data.geom_xpos[sphere_geom_id].copy()

        # Get orientation of the cube (rotation matrix)
        cube_xmat = self.data.xmat[cube_body_id].reshape(3, 3)
        
        # Cube's forward vector in world frame is the first column of the rotation matrix
        cube_forward = cube_xmat[:, 0]
        
        # Heading (yaw) angle of cube in global frame (assuming 2D movement on xy-plane)
        cube_yaw = np.arctan2(cube_forward[1], cube_forward[0])

        # Vector from cube to sphere
        relative_pos = sphere_pos - cube_pos
        distance = np.linalg.norm(relative_pos[:2])
        azimuth_global = np.arctan2(relative_pos[1], relative_pos[0])

        # Relative azimuth (angle between agent's heading and the target)
        self.relative_azimuth = azimuth_global - cube_yaw
        self.relative_azimuth = (self.relative_azimuth+ np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]

        # Combine into observation
        obs = np.concatenate([self.lidar_readings, [distance, self.relative_azimuth]])
        return obs.astype(np.float32)



    
    def _get_info(self):
        # Get positions using direct geom access instead of sensors
        cube_pos = np.zeros(3)
        sphere_pos = np.zeros(3)
        
        # Get cube position from its body
        cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube_body")
        if cube_body_id >= 0:
            # Get position of the cube body in world coordinates
            cube_pos = self.data.xpos[cube_body_id].copy()
        
        # Get sphere position directly from its geom
        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_geom_id >= 0:
            # Get position of the sphere geom in world coordinates
            sphere_pos = self.data.geom_xpos[sphere_geom_id].copy()
        
        # Calculate distance to sphere
        distance_to_sphere = np.linalg.norm(sphere_pos - cube_pos)

        
        
        return {
            "distance_to_sphere": distance_to_sphere,
            "cube_position": cube_pos,
            "sphere_position": sphere_pos
        }
        
















    
    def reset(self, seed=None, options=None):
        #reset lidar to max distance
        if self.last_episode_result == "success":
            self.success_count += 1
        elif self.last_episode_result == "collision":
            self.collision_count += 1
        elif self.last_episode_result == "timeout":
            self.timeout_count += 1
            
        #self.steps_taken = 0

        if self.episode_count > 1:
            self.success_rate = self.success_count / (self.episode_count)
            self.collision_rate = self.collision_count / (self.episode_count)
            self.timeout_rate = self.timeout_count / (self.episode_count)

        if self.last_episode_result == "success" and self.training_mode == False:
            print(f"SUCCESS: Eval_episode = {self.episode_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")
        elif self.last_episode_result == "collision" and self.training_mode == False:
            print(f"COLLISION: Eval_episode = {self.episode_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")
        elif self.last_episode_result == "timeout" and self.training_mode == False:
            print(f"TIMEOUT: Eval_episode = {self.episode_count} sr={self.success_rate:.2f}, cr={self.collision_rate:.2f}, tr={self.timeout_rate:.2f}, return={self.episode_return:.2f}")

        self.episode_count += 1
        self.last_episode_result = None

        
        super().reset(seed=seed)
        
        # Reset step counter
        self.current_step = 0
        #print("Return:", self.reward)
        self.episode_return = 0
        
        # Randomize cube position (within valid area away from walls)
        if seed is not None:
            np.random.seed(seed)
            
        # Place cube randomly in the arena (not too close to walls or sphere)
        wall_buffer = 2.0
        min_x = -17.5 + wall_buffer
        max_x = 17.5 - wall_buffer
        min_y = -17.5 + wall_buffer
        max_y = 17.5 - wall_buffer

        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        yaw = np.random.uniform(-np.pi, np.pi)
        
        # Reset cube position
        self.data.qpos[:3] = [x, y, yaw]
        self.data.qvel[:] = 0
        
        
        # Reset sphere position (fixed for simplicity)
        sphere_x = np.random.uniform(min_x, max_x)
        sphere_y = np.random.uniform(min_y, max_y)
        sphere_z = 2.0

        min_separation = 7.0
        attempts = 0
        max_attempts = 100


        while np.sqrt((x-sphere_x)**2 + (y-sphere_y)**2) < min_separation:
            sphere_x = np.random.uniform(min_x, max_x)
            sphere_y = np.random.uniform(min_y, max_y)
            attempts += 1
            if attempts > max_attempts:
                # If we can't find a good position after many attempts,
                # just place the sphere far away from the cube
                angle = np.random.uniform(0, 2*np.pi)
                sphere_x = x + min_separation * np.cos(angle)
                sphere_y = y + min_separation * np.sin(angle)
                # Ensure it's still within bounds
                sphere_x = np.clip(sphere_x, min_x, max_x)
                sphere_y = np.clip(sphere_y, min_y, max_y)
                break

        sphere_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "sphere")
        if sphere_geom_id >= 0:
            self.model.geom_pos[sphere_geom_id, :] = [sphere_x, sphere_y, sphere_z]

        
        for i in range(1, 21):
            obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{i}")
            if obstacle_body_id >= 0:
                valid_position = False
                max_place_attempts = 100
                placement_attempts = 0

                while not valid_position and placement_attempts < max_place_attempts:
                    obstacle_x = np.random.uniform(min_x, max_x)
                    obstacle_y = np.random.uniform(min_y, max_y)

                    # Check if the new position is valid
                    distance_to_cube = np.sqrt((x - obstacle_x) ** 2 + (y - obstacle_y) ** 2)
                    distance_to_sphere = np.sqrt((sphere_x - obstacle_x) ** 2 + (sphere_y - obstacle_y) ** 2)

                    if distance_to_cube < min_separation or distance_to_sphere < min_separation:
                        placement_attempts += 1
                        continue

                    too_close_to_other_obstacles = False
                    for j in range(1, i):
                        other_obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"obstacle{j}")
                        if other_obstacle_body_id >= 0:
                            other_obstacle_pos = self.model.body_pos[other_obstacle_body_id].copy()
                            distance_to_other_obstacle = np.sqrt((obstacle_x - other_obstacle_pos[0]) ** 2 + (obstacle_y - other_obstacle_pos[1]) ** 2)
                            if distance_to_other_obstacle < min_separation:
                                too_close_to_other_obstacles = True
                                break
                    
                    if too_close_to_other_obstacles:
                        placement_attempts += 1
                        continue

                    valid_position = True

                if not valid_position:
                    angle = (i-1)*(2*np.pi/4)
                    radius = 14.0
                    obstacle_x = radius * np.cos(angle)
                    obstacle_y = radius * np.sin(angle)

                    obstacle_x = np.clip(obstacle_x, min_x, max_x)
                    obstacle_y = np.clip(obstacle_y, min_y, max_y)

                self.model.body_pos[obstacle_body_id, :2] = [obstacle_x, obstacle_y]

        # Update simulation
        mujoco.mj_forward(self.model, self.data)
        
        # Get observation
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    










    
    def step(self, action):
        # Movement parameters
        move_max_linspeed = 1 # m/s max speed
        max_ang_speed = 1 # rad/s max angular speed

        dt = self.model.opt.timestep # 0.1 [s/step]
        
        #translated_action0 = (action[0]+1)/2
        translated_action0 = action[0]
        # Apply action
        linear_velocity = translated_action0*move_max_linspeed
        angular_velocity = action[1]*max_ang_speed

        # if self.training_mode == False:
        #     print(f"Action: {action}, Linear Velocity: {linear_velocity}, Angular Velocity: {angular_velocity}")

        # Get current position and orientation
        x, y, theta = self.data.qpos[:3]

        if (angular_velocity != 0):
            deltax = (linear_velocity/angular_velocity)*(np.sin(theta+angular_velocity*dt) - np.sin(theta))
            deltay = (linear_velocity/angular_velocity)*(-np.cos(theta+angular_velocity*dt) + np.cos(theta))
            x += deltax
            y += deltay
            theta += angular_velocity*dt
        else:
            # Updating position
            x += linear_velocity * np.cos(theta)
            y += linear_velocity * np.sin(theta)
            # Updating orientation is not necessary, since we are not rotating
            

        # Set position and orientation
        self.data.qpos[:3] = [x, y, theta]
        
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        self.current_step += 1
        
        # Update simulation state
        mujoco.mj_forward(self.model, self.data)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        reward = 0.0
        # Check for collision with obstacles
        contact_with_obstacles = False
        contact_with_sphere = False

        cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            
            geom1_id = contact.geom1
            geom2_id = contact.geom2
            
            if geom1_id == cube_geom_id or geom2_id == cube_geom_id:
                # Check if the contact is with the walls
                other_geom_id = geom2_id if geom1_id == cube_geom_id else geom1_id

                other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom_id)

                if (other_geom_name and "wall" in other_geom_name) or (other_geom_name and "obstacle" in other_geom_name) :  
                    contact_with_obstacles = True
                    break
                elif other_geom_name and "sphere" in other_geom_name:
                    contact_with_sphere = True
                    break
        
        # Calculate reward and check termination conditions
        distance_to_sphere = info["distance_to_sphere"]

        reward = (self.previous_distance - distance_to_sphere)
        #print(f"Reward: {reward}, Distance to Sphere: {distance_to_sphere}, Previous Distance: {self.previous_distance}")
        #reward = (self.previous_distance - distance_to_sphere)

        self.episode_return += reward
        self.previous_distance = distance_to_sphere

        reward += -0.1*abs(self.relative_azimuth)

        
        # Terminal conditions
        terminated = False
        truncated = False


        too_close_to_obstacles = False
        for i in range(1, len(self.lidar_readings)):
            if self.lidar_readings[i] < 0.1:
                too_close_to_obstacles = True
                break


        
        if contact_with_obstacles or too_close_to_obstacles:
            reward = -10.0
            self.episode_return += reward
            self.last_episode_result = "collision"
            info["steps_taken"] = self.current_step
            terminated = True   

        elif distance_to_sphere < 2.0 or contact_with_sphere:  
            reward = 200
            self.episode_return += reward
            self.last_episode_result = "success"
            info["steps_taken"] = self.current_step
            terminated = True

        elif self.current_step >= self.max_episode_steps:
            #reward = -50.0
            #self.episode_return += reward
            self.last_episode_result = "timeout"
            info["steps_taken"] = self.current_step
            truncated = True

        info["episode_result"] = self.last_episode_result

        # Render if needed
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
    










    
    def render(self):
        if self.render_mode == "human" and self.viewer:
            self.viewer.sync()
            
            if not self.training_mode:  # During evaluation
                # Sleep exactly the model timestep to achieve real-time rendering
                #time.sleep(self.model.opt.timestep)
                time.sleep(0.01)  # Adjust this value for smoother rendering
                return True
            elif self.current_step % 10 == 0:  
                time.sleep(0.001)  # Faster rendering during training
            return True
        return False
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None







####################################################################################################
            















class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_results = []
        self.current_episode_rewards = []
        self.current_episode_lengths = []
        
        # Counters for statistics
        self.success_count = 0
        self.collision_count = 0
        self.timeout_count = 0
        self.total_episodes = 0

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.current_episode_rewards = [0.0] * n_envs
        self.current_episode_lengths = [0] * n_envs

    def _on_step(self) -> bool:
        dones = self.locals['dones']
        rewards = self.locals['rewards']
        infos = self.locals['infos']

        for i, done in enumerate(dones):
            self.current_episode_rewards[i] += rewards[i]
            self.current_episode_lengths[i] += 1

            if done:
                self.episode_rewards.append(self.current_episode_rewards[i])
                self.episode_lengths.append(self.current_episode_lengths[i])
                
                # Extract episode result from info
                if 'episode_result' in infos[i]:
                    result = infos[i]['episode_result']
                    self.episode_results.append(result)
                    
                    # Update counters based on episode result
                    if result == "success":
                        self.success_count += 1
                    elif result == "collision":
                        self.collision_count += 1
                    elif result == "timeout":
                        self.timeout_count += 1
                        
                    self.total_episodes += 1
                
                self.current_episode_rewards[i] = 0.0
                self.current_episode_lengths[i] = 0

        # Log statistics to TensorBoard every 1000 steps
        if self.n_calls % 1000 == 0 and self.total_episodes > 0:
            # Calculate mean reward and episode length
            if len(self.episode_rewards) > 0:
                mean_reward = sum(self.episode_rewards) / len(self.episode_rewards)
                mean_episode_length = sum(self.episode_lengths) / len(self.episode_lengths)
                self.logger.record('metrics/mean_episode_reward', mean_reward)
                self.logger.record('metrics/mean_episode_length', mean_episode_length)
            
            # Calculate and log success, collision, and timeout rates
            success_rate = self.success_count / self.total_episodes if self.total_episodes > 0 else 0
            collision_rate = self.collision_count / self.total_episodes if self.total_episodes > 0 else 0
            timeout_rate = self.timeout_count / self.total_episodes if self.total_episodes > 0 else 0
            
            self.logger.record('metrics/success_rate', success_rate)
            self.logger.record('metrics/collision_rate', collision_rate)
            self.logger.record('metrics/timeout_rate', timeout_rate)
            
            # Reset counters for the next logging interval
            self.episode_rewards = []
            self.episode_lengths = []
            self.episode_results = []

        return True




def make_env(num_rays, model_path="xml_models/obstacles_cube_model.xml", training = True):
    def _init():
        env = CubeNavigationEnv(
            num_rays=num_rays, 
            render_mode=None if training else "human",
            model_path=model_path, 
            training_mode=training
            )
        return env
    return _init



def train_td3_agent(num_rays, model_path="xml_models/obstacles_cube_model.xml", num_envs=16, num_steps=100000, run_id="training1", training=True):
    log_dir = "./td3_cube_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)

    if not training:
        env = CubeNavigationEnv(
            num_rays=num_rays, 
            render_mode="human", 
            model_path=model_path,
            training_mode=False
        )
    else:
        # Create vectorized environment
        env = SubprocVecEnv([make_env(num_rays, model_path, training=training) for _ in range(num_envs)])

    policy_kwargs = dict(
        net_arch=[128, 128],  # Two hidden layers with 128 neurons each
    )

    # Create PPO model with built-in logging
    model = TD3(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        device="cpu",
        learning_rate=0.001,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        policy_kwargs=policy_kwargs
    )

    
    # Only keep necessary callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=500000,
        save_path="./td3_cube_checkpoints/",
        name_prefix="td3_cube_model"
    )

    reward_callback = RewardCallback()
    callbacks= [checkpoint_callback, reward_callback]
    
    # Train the model
    model.learn(
        total_timesteps=num_steps,
        callback=callbacks,
        tb_log_name=run_id  # This ensures logs go to the correct subdirectory
    )
    
    # Save the final model
    model.save(f"{run_id}")
    env.close()







##################################################################################################



    
if __name__ == "__main__":
    train = True

    name = "DEFAULT"

    parser = argparse.ArgumentParser(description="Train a TD3 agent to navigate a cube to a target sphere.")
    parser.add_argument("--num_rays", type=int, default=50, help="Number of LiDAR rays around the sphere.")
    parser.add_argument("--model_path", type=str, default="xml_models/obstacles_cube_model.xml", help="Path to the MuJoCo model XML file.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--train", action="store_true", help="Train the TD3 agent.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the TD3 agent.")
    parser.add_argument("--num_steps", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--run_id", type=str, default="DEFAULT_TD3", help="Run ID for TensorBoard logging.")
    args = parser.parse_args()
    
    name = args.run_id

    train = args.train
    if args.eval:
        train = False

    if train:
        train_td3_agent(args.num_rays, args.model_path, args.num_envs, args.num_steps, args.run_id, training=True)
    
    # Load the last trained model
    model = TD3.load(f"{args.run_id}")
    
    # Create the evaluation environment
    eval_env = CubeNavigationEnv(
        num_rays=args.num_rays, 
        render_mode="human", 
        model_path=args.model_path,
        training_mode=False
    )
    
    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=300, deterministic=True)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    eval_env.close()