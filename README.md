# Intro to AI Coursework Projects

This repository contains short projects I completed as part of my Intro to AI (CS325 - Emory Univ.) coursework. Each project demonstrates  my understanding and application of various AI concepts and techniques.

## Table of Contents

- [Project 1: Adversarial Search](https://github.com/beWanja/cs325_ai_projects/tree/main/multiagent-pacman)
- [Project 2: HMMs and Bayesian Inference](https://github.com/beWanja/cs325_ai_projects/tree/main/hmm_and_bayesian_inference)
- [Project 3: MDPs and Reinforcement Learning](https://github.com/beWanja/cs325_ai_projects/tree/main/pacman-rl)
- [Project 4: Image Classification using Neural Networks](https://github.com/beWanja/cs325_ai_projects/tree/main/nns-cv)
- [Project 5: Deep RL](https://github.com/beWanja/cs325_ai_projects/tree/main/gym-env-deep-rl)

## Project 1: Adversarial Search for Pac-Man

### Description
This project covers the fundamentals of strategic decision-making in AI, including minimax, expectimax, and alpha-beta pruning algorithms and basic search algorithms. I use the PacMan AI projects developed at UC Berkley.

### Key Concepts
- Minimax
- Alpha-beta pruning
- Expectimax 
- Breadth-First Search (BFS)
- Depth-First Search (DFS)
- A* Search Algorithm

### Technologies Used
- Python 2.7

### Files
- `multiAgents.py` - Implementation of minimax, alpha-beta pruning, and expectimax for Pacman agent

## Project 2: Pac-Man Ghost Hunting 

### Description
In this project, I advance Project 1's pac-man agents by implementing ghost tracking using Bayesian Inference and Hidden Markov Models. 

### Key Concepts
- Hidden Markov Models
- Statistical Inference

### Technologies Used
- Python 2.7

### Files
- `bustersAgents.py` - Agents playing the Ghostbusters version of Pacman
- `inference.py` - Tracking ghosts over time

## Project 3: MDPs and Reinforcement Learning

### Description
This project involved implementing value iteration and Q-learning algorithms which I then applied to a Gridworld, Pac-Man, and a Crawler robot. 

### Key Concepts
- Markov Decision Process
- Reinforcement Learning
- Value Iteration
- Q-learning
- Policy Iteration

### Technologies Used
- Python 2.7

### Files
- `valueIterationAgents.py` - A value iteration agent for solving known MDPs.
- `qlearningAgents.py` - Q-learning agents for Gridworld, Crawler, and Pacman.

## Project 4: Image Classification using Neural Networks 

### Description
In this project, I build and train neural networks for image classification using the MNIST Fashion and CIFAR10 datasets.

### Key Concepts
- Neural Network Architecture
- Convolutional Networks
- Perceptron and MLP
- Model Training and Validation
- Computer Vision

### Technologies Used
- Python
- Keras
- TensorFlow
- NumPy

### Files
- `mlp_cifar_model` - Implementation of a multilayer perceptron to classify images on the CIFAR10 dataset
- `cnn_cifar_model` - Implementation of convolutional neural networks to classify images on the CIFAR10 dataset
- `CNN-cifar model.ipynb`
- `CNN-fashion model.ipynb`
- `MLP-CIFAR model.ipynb`

## Project 5: Deep Learning

### Description
In this project, I implement table-based Q-Learning and use Deep Q-Learning (DQN) and Policy Optimization (PPO) algorithms to train neural networks that operate a robot in two different environments in the [Gymnasium environment](https://github.com/Farama-Foundation/GymnasiumLinks)

### Key Concepts
- Table-based Q-learning
- Deep Q-learning (DQN)
- Policy Optimization (PPO)
 
### Technologies Used
- Python
- PyTorch
- DQN

### Files
- `car_racer.ipynb`
- `lunar_lander.ipynb`


