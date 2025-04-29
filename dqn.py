# Spring 2025, 535507 Deep Learning
# Lab5: Value-based RL
# Contributors: Wei Hung and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse

gym.register_envs(ale_py)

# Weight initializer
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

# DQN network: MLP for vector states, CNN for image states
class DQN(nn.Module):
    """
        Architecture: chooses MLP if input is 1D, else CNN for images.
    """
    def __init__(self, num_actions, input_shape):
        super(DQN, self).__init__()
        self.input_shape = input_shape
        C, *rest = input_shape
        if len(input_shape) == 1:
            self.network = nn.Sequential(
                nn.Flatten(),
                nn.Linear(input_shape[0], 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions)
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(C, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(self._get_conv_output_size(input_shape), 512),
                nn.ReLU(),
                nn.Linear(512, num_actions)
            )

    def _get_conv_output_size(self, shape):
        with torch.no_grad():
            x = torch.zeros(1, *shape)
            x = nn.Conv2d(shape[0], 32, kernel_size=8, stride=4)(x)
            x = nn.Conv2d(32, 64, kernel_size=4, stride=2)(x)
            x = nn.Conv2d(64, 64, kernel_size=3, stride=1)(x)
            return int(np.prod(x.shape[1:]))

    def forward(self, x):
        return self.network(x)

# Preprocessors
class AtariPreprocessor:
    """
        Grayscale + resize + frame stack for Atari.
    """
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        frame = self.preprocess(obs)
        self.frames.append(frame)
        return np.stack(self.frames, axis=0)

# No-op preprocessor for vector state envs
class IdentityPreprocessor:
    def reset(self, obs): return obs
    def step(self, obs): return obs

# Prioritized Experience Replay buffer
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def add(self, transition, error):
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = (abs(error) + 1e-6) ** self.alpha
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        probs = self.priorities[:len(self.buffer)]
        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        transitions = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return transitions, indices, weights

    def update_priorities(self, indices, errors):
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(err) + 1e-6) ** self.alpha

# DQN Agent with support for Double DQN, PER, multi-step returns, and snapshotting
class DQNAgent:
    def __init__(self, env_name, args):
        self.env = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", self.device)

        obs, _ = self.env.reset()
        if obs.ndim == 1:
            self.preprocessor = IdentityPreprocessor()
            init_state = obs
        else:
            self.preprocessor = AtariPreprocessor(frame_stack=args.frame_stack)
            init_state = self.preprocessor.reset(obs)
        input_shape = init_state.shape

        self.q_net = DQN(self.num_actions, input_shape).to(self.device)
        self.q_net.apply(init_weights)
        self.target_net = DQN(self.num_actions, input_shape).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # Hyperparams
        self.batch_size = args.batch_size
        self.gamma = args.discount_factor
        self.epsilon = args.epsilon_start
        self.epsilon_decay = args.epsilon_decay
        self.epsilon_min = args.epsilon_min
        self.max_episode_steps = args.max_episode_steps
        self.replay_start_size = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step = args.train_per_step

        # Task3 flags
        self.use_ddqn = args.use_double_dqn
        self.use_per = args.use_per
        self.n_step = args.n_step

        # Buffers
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(args.memory_size)
        else:
            self.memory = deque(maxlen=args.memory_size)
        if self.n_step > 1:
            self.n_step_buffer = deque(maxlen=self.n_step)

        # Counters & snapshots
        self.env_count = 0
        self.train_count = 0
        self.best_reward = -float('inf')
        os.makedirs(args.save_dir, exist_ok=True)

        # Store args for checkpoint logic
        self.args = args

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.num_actions)
        st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return int(self.q_net(st).argmax(1).item())

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return
        # ε decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # Sample
        if self.use_per:
            transitions, idxs, weights = self.memory.sample(self.batch_size)
            is_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            transitions = batch
            is_weights = None
        states, actions, rewards, next_states, dones = zip(*transitions)
        S  = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        NS = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        A  = torch.tensor(actions, dtype=torch.int64).to(self.device)
        R  = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        D  = torch.tensor(dones, dtype=torch.float32).to(self.device)

        Q  = self.q_net(S).gather(1, A.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            if self.use_ddqn:
                next_actions = self.q_net(NS).argmax(1)
                Qn = self.target_net(NS).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                Qn = self.target_net(NS).max(1)[0]
            Y  = R + (1 - D) * self.gamma * Qn

        if is_weights is not None:
            loss = (is_weights * (Q - Y).pow(2)).mean()
        else:
            loss = nn.MSELoss()(Q, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Debug log every 1000 updates
        if self.train_count % 1000 == 0:
            print(f"[Train Step {self.train_count}] Loss: {loss.item():.4f} Epsilon: {self.epsilon:.4f}")

        if self.use_per:
            errs = (Q - Y).abs().detach().cpu().numpy()
            self.memory.update_priorities(idxs, errs)

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

    def run(self, episodes):
        for ep in range(episodes):
            obs, _ = self.env.reset()
            state = (self.preprocessor.reset(obs)
                     if hasattr(self.preprocessor, 'reset') else obs)
            done = False
            total_r, steps = 0, 0
            while not done and steps < self.max_episode_steps:
                a = self.select_action(state)
                no, r, term, trunc, _ = self.env.step(a)
                done = term or trunc
                ns = self.preprocessor.step(no) if hasattr(self.preprocessor, 'step') else no
                # Multi-step Experience: accumulate n-step transitions
                self.n_step_buffer.append((state, a, r, ns, done))
                if len(self.n_step_buffer) == self.n_step:
                    # Compute n-step return
                    R, last_s, last_done = 0.0, self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
                    for idx, (_, a0, r0, _, _) in enumerate(self.n_step_buffer):
                        R += (self.gamma ** idx) * r0
                    s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                    transition = (s0, a0, R, last_s, last_done)
                    # Add to replay memory
                    if self.use_per:
                        # Use max priority for new transition
                        max_p = self.memory.priorities.max() if len(self.memory.buffer) > 0 else 1.0
                        self.memory.add(transition, max_p)
                    else:
                        self.memory.append(transition)
                for _ in range(self.train_per_step): self.train()
                state, total_r = ns, total_r + r
                self.env_count += 1; steps += 1

                # Task 3: save checkpoint if this step matches
                if hasattr(self.args, 'checkpoint_steps') and self.env_count in self.args.checkpoint_steps:
                    filename = f"LAB5_{self.args.student_id}_task3_pong{self.env_count}.pt"
                    path = os.path.join(self.args.save_dir, filename)
                    torch.save(self.q_net.state_dict(), path)
                    print(f"[Checkpoint] Saved Task3 snapshot at env step {self.env_count} to {path}")

                if self.env_count % 1000 == 0:                 
                    print(f"[Collect] Ep: {ep} Step: {steps} SC: {self.env_count} UC: {self.train_count} Eps: {self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep,
                        "Step Count": steps,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon
                    })

            # Evaluation & best-model saving
            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    task_num, suffix = (1, 'cartpole') if 'CartPole' in self.env.spec.id else (2, 'pong')
                    filename = f"LAB5_{self.args.student_id}_task{task_num}_{suffix}.pt"
                    model_path = os.path.join(self.args.save_dir, filename)
                    torch.save(self.q_net.state_dict(), model_path)
                    print(f"Saved new best model to {model_path} with reward {eval_reward:.2f}")
                print(f"[TrueEval] Ep: {ep} Eval Reward: {eval_reward:.2f} SC: {self.env_count} UC: {self.train_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward
                })
        
    def evaluate(self):
        obs, _ = self.test_env.reset()
        st = obs if hasattr(self.preprocessor,'reset')==False else self.preprocessor.reset(obs)
        done = False; tot=0
        while not done:
            # Convert state to float32 tensor
            state_tensor = torch.from_numpy(np.array(st, dtype=np.float32)).unsqueeze(0).to(self.device)
            with torch.no_grad():
                a = int(self.q_net(state_tensor).argmax(1).item())
            no, r, term, tr, _ = self.test_env.step(a)
            done = term or tr; tot += r; st = self.preprocessor.step(no) if hasattr(self.preprocessor, 'step') else no
        return tot

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--student-id', type=str, required=True)
    parser.add_argument('--frame-stack', type=int, default=4)
    parser.add_argument('--checkpoint-steps', type=int, nargs='*', default=[200000,400000,600000,800000,1000000])
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--wandb-run-name', type=str, default='dlp-lab5')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--memory-size', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--discount-factor', type=float, default=0.99)
    parser.add_argument('--epsilon-start', type=float, default=1.0)
    parser.add_argument('--epsilon-decay', type=float, default=0.99999)
    parser.add_argument('--epsilon-min', type=float, default=0.05)
    parser.add_argument('--target-update-frequency', type=int, default=1000)
    parser.add_argument('--replay-start-size', type=int, default=50000)
    parser.add_argument('--max-episode-steps', type=int, default=500)
    parser.add_argument('--train-per-step', type=int, default=1)
    parser.add_argument('--use-double-dqn', action='store_true')
    parser.add_argument('--use-per', action='store_true')
    parser.add_argument('--n-step', type=int, default=1)
    args = parser.parse_args()

    wandb.init(project='DLP-Lab5', name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(args.env, args)
    agent.run(episodes=1000)