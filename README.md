# Reinforcement Learning Algorithms and Environments

This repository contains a comprehensive collection of Reinforcement Learning algorithms and experiments, ranging from fundamental concepts to advanced Deep Q-Networks.

## 📁 Repository Structure

The codebase is organized by algorithm families and concepts:

- **Multi-Arm-Bandits**: Implementations of various Bandit algorithms (Epsilon-greedy, UCB, Gradient Bandits).
- **Markov-Decision-Process**: Formulations and solvers for MDPs.
- **Dynamic-Programming**: Policy Evaluation, Policy Iteration, and Value Iteration.
- **Monte-Carlo**: Monte Carlo prediction and control methods.
- **Tabular-learning-and-Bellman-Equation**: Q-Learning and SARSA in tabular environments (e.g., FrozenLake).
- **Cross-entropy-method**: Implementations of the Cross-Entropy Method for basic continuous/discrete control.
- **Deep-Q-Networks (DQN)**: Advanced Reinforcement Learning using neural networks. Includes implementations for complex environments like Atari's *Riverbank* and *Pong*.
- **OpenAI Gymnasium**: Wrapper utilities and examples using the Gymnasium API.
- **torchPlayGround**: PyTorch experimental ground, including GAN implementations for Atari images.

## 🚀 Setup & Installation

It is recommended to use a virtual environment (`conda` or `venv`) to manage dependencies.

**1. Clone the repository**
```bash
git clone https://github.com/manishswami1114/Reinforcement-Learning.git
cd Reinforcement-Learning
```

**2. Install Dependencies**
You will need PyTorch and Gymnasium along with Atari dependencies:
```bash
pip install torch torchvision torchaudio
pip install gymnasium[atari,accept-rom-license]
pip install numpy matplotlib
```

## 🎮 Running an Experiment

Navigate into a specific algorithm's folder and run the python script. For example, to run the River Raid DQN simulation:

```bash
cd Deep-Q-Networks/River-Raid
python RiverRaidSimulation.py
```

## ⚙️ Notes on Data & Structure
- Pre-trained models and large checkpoint files (like `.dat`) are ignored via `.gitignore` to keep the version control system fast.
- Check the `runs/` directory (created locally during training) for TensorBoard logs if applicable.
