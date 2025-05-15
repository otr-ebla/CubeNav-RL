# MuJoCo RL Cube Environment Navigation

This repository contains code for training a mobile robot agent to navigate through a  3D environment filled with randomly placed static cubes using various Deep Reinforcement Learning algorithms.

![Navigation Demo](assets/demo.gif)

## 🧠 Introduction

The goal of this project is to develop an autonomous reinforcement learning (RL) agent capable of navigating complex environments while avoiding obstacles. The environment consists of a bounded area with randomly generated static cubes representing obstacles. The agent’s objective is to reach a target location without collisions.

The agent perceives its surroundings through simulated **laser sensor data**, which serves as its observation input. These laser readings give the agent distance measurements to nearby obstacles, forming the foundation for learning navigation and avoidance behavior.

## 🚀 Features

- 🌐 Procedurally generated environments with random cube placements
- 🤖 Mobile robot agent with simulated **laser-based perception**
- 🏁 Goal-directed navigation towards a target location
- 💥 Collision detection and avoidance as part of the reward function
- 📚 Modular framework for training using different RL algorithms (e.g., PPO, TQC, SAC, TD3)
- 📊 Logging and evaluation tools for visualizing agent performance

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/otr-ebla/CubeNav-RL.git
   cd CubeNav-RL
