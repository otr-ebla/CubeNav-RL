import numpy as np
import time
import gymnasium as gym
import argparse
import os
import torch
import xml.etree.ElementTree as ET

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer
from skrl.envs.wrappers.jax import wrap_env
from skrl.utils import set_seed
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch.utils.tensorboard import SummaryWriter

from CubeNavigationEnv import CubeNavigationEnv_class


# Define custom models for the SKRL agent
class CustomActor(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions)
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, self.num_actions)
        )
        
        self.log_std_parameter = torch.nn.Parameter(torch.zeros(self.num_actions) - 2.0)  # Initialize at -2.0 like original code
        
    def compute(self, states, **kwargs):
        return self.net(states), self.log_std_parameter


class CustomCritic(Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        
        self.net = torch.nn.Sequential(
            torch.nn.Linear(self.num_observations, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 1)
        )
        
    def compute(self, states, **kwargs):
        return self.net(states)


# Helper function to create environment
def make_env(num_rays, model_path="xml_models/obstacles_cube_model.xml", training=True):
    env = CubeNavigationEnv_class(
        num_rays=num_rays, 
        render_mode=None if training else "human",
        model_path=model_path, 
        training_mode=training
    )
    return env


# Custom callback for logging rewards
class RewardLogger:
    def __init__(self, writer):
        self.writer = writer
        self.episode_rewards = []
        self.current_episode_rewards = {}  # Dictionary to track rewards by environment
        self.episodes = 0
        self.total_steps = 0
        
    def update(self, agent, rewards, terminated, truncated, env_indices=None):
        # If no env_indices provided, assume sequential indices
        if env_indices is None:
            env_indices = range(len(rewards))
            
        # Update rewards for each environment
        for i, env_idx in enumerate(env_indices):
            # Initialize if needed
            if env_idx not in self.current_episode_rewards:
                self.current_episode_rewards[env_idx] = 0.0
                
            # Add reward
            self.current_episode_rewards[env_idx] += rewards[i].item()
            
            # If episode is done
            if terminated[i] or truncated[i]:
                reward = self.current_episode_rewards[env_idx]
                self.episode_rewards.append(reward)
                
                # Log to tensorboard
                self.writer.add_scalar("charts/episode_reward", reward, self.episodes)
                self.writer.add_scalar("charts/mean_reward_last_100", np.mean(self.episode_rewards[-100:]), self.episodes)
                
                # Reset episode reward for this environment
                self.current_episode_rewards[env_idx] = 0.0
                self.episodes += 1
                
        self.total_steps += len(rewards)


def train_ppo_agent(num_rays, model_path="xml_models/obstacles_cube_model.xml", num_envs=16, num_steps=100000, run_id="training1", training=True):
    log_dir = "./ppo_cube_tensorboard/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Set seeds for reproducibility
    set_seed(42)
    
    if not training:
        # Create single environment for evaluation
        env = make_env(num_rays, model_path, training=False)
        env = wrap_env(env)
    else:
        # Create vectorized environment for training
        envs = []
        for i in range(num_envs):
            env = make_env(num_rays, model_path, training=True)
            envs.append(wrap_env(env))
        
        # Use the first environment to extract spaces
        env = envs[0]
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=f"{log_dir}/{run_id}")
    
    # Setup device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Instantiate models
    actor = CustomActor(env.observation_space, env.action_space, device)
    critic = CustomCritic(env.observation_space, env.action_space, device)
    
    # Memory
    memory = RandomMemory(memory_size=16384, num_envs=num_envs if training else 1, device=device)
    
    # Configure PPO
    ppo_config = PPO_DEFAULT_CONFIG.copy()
    ppo_config["rollouts"] = 500  # Horizon
    ppo_config["learning_epochs"] = 3  # Update epochs
    ppo_config["mini_batches"] = 8  # Number of minibatches (equivalent to batch_size=64)
    ppo_config["discount_factor"] = 0.99  # Gamma
    ppo_config["lambda"] = 0.95  # GAE lambda
    ppo_config["learning_rate"] = 3e-4  # Learning rate
    ppo_config["random_timesteps"] = 0  # Initial random exploration steps
    ppo_config["learning_starts"] = 0  # Learning starts after this many steps
    ppo_config["grad_norm_clip"] = 0.5  # Gradient clipping
    ppo_config["ratio_clip"] = 0.3  # PPO clip range
    ppo_config["value_clip"] = 0.3  # Value clip range
    ppo_config["clip_predicted_values"] = True  # Clip predicted values
    ppo_config["entropy_loss_scale"] = 0.01  # Entropy coefficient
    ppo_config["value_loss_scale"] = 0.5  # Value loss coefficient
    ppo_config["kl_threshold"] = 0.05  # KL threshold for early stopping
    
    # Set up preprocessors for observation and value normalization
    ppo_config["state_preprocessor"] = RunningStandardScaler
    ppo_config["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
    ppo_config["value_preprocessor"] = RunningStandardScaler
    ppo_config["value_preprocessor_kwargs"] = {"size": 1, "device": device}
    
    # Define agent
    agent = PPO(
        models={
            "policy": actor,
            "value": critic
        },
        memory=memory,
        cfg=ppo_config,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device
    )
    
    # Configure and create trainer
    if training:
        checkpoint_dir = "./ppo_cube_checkpoints/"
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Create reward logger
        reward_logger = RewardLogger(writer)
        
        # Monkey patch the record_transition method to include logging
        original_record_transition = agent.record_transition
        
        def record_transition_with_logging(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps):
            # Call the original method
            original_record_transition(states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps)
            
            # Update the logger
            reward_logger.update(agent, rewards, terminated, truncated)
        
        # Replace the method
        agent.record_transition = record_transition_with_logging
        
        # Create the trainer
        trainer = SequentialTrainer(
            cfg={"timesteps": num_steps, "headless": True},
            env=envs if training else [env],
            agents=agent
        )
        
        # Start training
        print(f"Training PPO agent for {num_steps} steps...")
        trainer.train()
        
        # Save the models
        torch.save(actor.state_dict(), f"{checkpoint_dir}/{run_id}_actor.pt")
        torch.save(critic.state_dict(), f"{checkpoint_dir}/{run_id}_critic.pt")
        
        # Save the state preprocessor
        if hasattr(agent, "state_preprocessor") and agent.state_preprocessor is not None:
            torch.save(agent.state_preprocessor.state_dict(), f"{checkpoint_dir}/{run_id}_state_preprocessor.pt")
        
        # Save the value preprocessor
        if hasattr(agent, "value_preprocessor") and agent.value_preprocessor is not None:
            torch.save(agent.value_preprocessor.state_dict(), f"{checkpoint_dir}/{run_id}_value_preprocessor.pt")
        
        # Close environments
        for env in envs:
            env.close()
    
    return agent


def evaluate_agent(agent, num_rays, model_path="xml_models/obstacles_cube_model.xml", num_episodes=300):
    # Create evaluation environment
    eval_env = make_env(num_rays, model_path, training=False)
    eval_env = wrap_env(eval_env)
    
    # Set evaluation mode
    agent.eval()
    
    rewards = []
    episode_lengths = []
    
    for _ in range(num_episodes):
        episode_reward = 0
        episode_length = 0
        obs, info = eval_env.reset()
        done = False
        
        while not done:
            # Get action from agent
            with torch.no_grad():
                action = agent.act(obs, explore=False)[0]  # Use deterministic actions for evaluation
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
            
            # Render if in evaluation mode
            eval_env.render()
            
        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {len(rewards)}: Reward = {episode_reward}, Length = {episode_length}")
    
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    print(f"Mean episode length: {mean_length}")
    
    eval_env.close()
    
    return mean_reward, std_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a PPO agent with SKRL to navigate a cube to a target sphere.")
    parser.add_argument("--num_rays", type=int, default=50, help="Number of LiDAR rays around the sphere.")
    parser.add_argument("--model_path", type=str, default="xml_models/obstacles_cube_model.xml", help="Path to the MuJoCo model XML file.")
    parser.add_argument("--num_envs", type=int, default=16, help="Number of parallel environments.")
    parser.add_argument("--train", action="store_true", help="Train the PPO agent.")
    parser.add_argument("--eval", action="store_true", help="Evaluate the PPO agent.")
    parser.add_argument("--num_steps", type=int, default=10000000, help="Number of training steps.")
    parser.add_argument("--run_id", type=str, default="DEFAULT", help="Run ID for TensorBoard logging.")
    args = parser.parse_args()
    
    name = args.run_id
    training = args.train
    
    if args.eval:
        training = False
    
    if training:
        agent = train_ppo_agent(args.num_rays, args.model_path, args.num_envs, args.num_steps, args.run_id, training=True)
    else:
        # Load a trained agent
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Create environment to get observation and action spaces
        env = make_env(args.num_rays, args.model_path, training=False)
        env = wrap_env(env)
        
        # Create models
        actor = CustomActor(env.observation_space, env.action_space, device)
        critic = CustomCritic(env.observation_space, env.action_space, device)
        
        # Load saved models
        checkpoint_dir = "./ppo_cube_checkpoints/"
        actor.load_state_dict(torch.load(f"{checkpoint_dir}/{name}_actor.pt"))
        critic.load_state_dict(torch.load(f"{checkpoint_dir}/{name}_critic.pt"))
        
        # Create memory
        memory = RandomMemory(memory_size=16384, num_envs=1, device=device)
        
        # Set up default configuration
        ppo_config = PPO_DEFAULT_CONFIG.copy()
        
        # Initialize preprocessors
        ppo_config["state_preprocessor"] = RunningStandardScaler
        ppo_config["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
        ppo_config["value_preprocessor"] = RunningStandardScaler
        ppo_config["value_preprocessor_kwargs"] = {"size": 1, "device": device}
        
        # Create PPO agent
        agent = PPO(
            models={
                "policy": actor,
                "value": critic
            },
            memory=memory,
            cfg=ppo_config,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device
        )
        
        # Load preprocessors if available
        try:
            if hasattr(agent, "state_preprocessor") and agent.state_preprocessor is not None:
                agent.state_preprocessor.load_state_dict(torch.load(f"{checkpoint_dir}/{name}_state_preprocessor.pt"))
            if hasattr(agent, "value_preprocessor") and agent.value_preprocessor is not None:
                agent.value_preprocessor.load_state_dict(torch.load(f"{checkpoint_dir}/{name}_value_preprocessor.pt"))
        except Exception as e:
            print(f"Warning: Error loading preprocessors: {e}. Evaluation may not work correctly.")
        
        env.close()
    
    # Evaluate the agent
    if args.eval:
        evaluate_agent(agent, args.num_rays, args.model_path, num_episodes=300)