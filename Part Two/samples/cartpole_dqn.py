## This is course material for Introduction to Modern Artificial Intelligence
## Example code: cartpole_dqn.py
## Author: Allen Y. Yang
##
## (c) Copyright 2020-2024. Intelligent Racing Inc. Not permitted for commercial use

## CartPole DQN with Rendering - Uses Gymnasium (the maintained successor to OpenAI Gym)
## Shows the game being played during training.
## The `import gymnasium as gym` alias is the migration path recommended by Gymnasium itself,
## so the rest of the script reads the same as the original Gym version.

import random
import gymnasium as gym
import os
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
# Use the legacy Adam optimizer (introduced in TF 2.11 as a stable fallback).
# Reason: the new-style Adam builds its iteration counter lazily, which makes
# Grappler's model_pruner pass log an "INVALID_ARGUMENT: Graph does not contain
# terminal node Adam/AssignAddVariableOp" message on the first interleaved
# predict()/fit() calls during DQN training. The legacy optimizer is identical
# in math, has no such interaction, and is what Apple's tensorflow-metal docs
# recommend on Apple Silicon.
from tensorflow.keras.optimizers.legacy import Adam
import time as time_module  # Rename to avoid conflict

EPISODES = 100
RENDER_EVERY = 10  # Render every N episodes to see progress

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(12, input_dim=self.state_size, activation='relu'))
        model.add(Dense(12, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state, verbose=0)[0]))
            target_f = self.model.predict(state, verbose=0)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # Gymnasium uses render_mode as a constructor kwarg; the env then renders
    # automatically during step() when render_mode='human'.
    # Fall back to a non-rendering env on headless machines (no display, no SDL).
    try:
        env = gym.make('CartPole-v1', render_mode='human')
        render_enabled = True
        print("Created environment with render_mode='human'")
    except Exception as exc:
        env = gym.make('CartPole-v1')
        render_enabled = False
        print(f"No display available ({exc.__class__.__name__}); running headless")

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    # Optional: Load pre-trained weights
    # agent.load("./save/cartpole-dqn.h5")
    
    batch_size = 32
    scores = []  # Store scores for plotting

    print("Starting DQN Training on CartPole")
    print("=" * 50)

    # Every CartPole-v1 episode is exactly one game: env.reset() at the top puts
    # the cart in a fresh randomized state, and the inner loop runs until either
    # the pole falls (terminated), the cart leaves the track (terminated), or the
    # 500-step cap is hit (truncated). Episodes alternate between two modes:
    #   * "render episodes" — pure evaluation: act greedily from the current
    #     network, no remember(), no replay(). The inner loop becomes
    #     predict -> step -> sleep, which lets the pygame window repaint at the
    #     environment's natural 50 Hz (no training stalls).
    #   * other episodes      — full training with epsilon-greedy exploration
    #     and experience replay, run as fast as the CPU allows (no rendering,
    #     no sleep). This is the same train/eval split DeepMind's original
    #     DQN paper uses to report scores.
    for e in range(EPISODES):
        # Gymnasium: reset() always returns (observation, info)
        state, _ = env.reset()
        state = np.reshape(state, [1, state_size])

        render_this_episode = (e % RENDER_EVERY == 0) or (e >= EPISODES - 5)

        if render_this_episode:
            print(f"\n🎮 Rendering Episode {e+1}/{EPISODES} (evaluation — no training)")

        for step in range(500):  # CartPole-v1 caps episodes at 500 timesteps
            if render_this_episode:
                # Pure greedy policy: pick the action with the highest Q-value.
                # No epsilon-exploration, no replay() — keeps the frame budget tiny.
                q_values = agent.model.predict(state, verbose=0)
                action = int(np.argmax(q_values[0]))
            else:
                action = agent.act(state)  # epsilon-greedy

            # Gymnasium: step() always returns (obs, reward, terminated, truncated, info)
            next_state, reward, terminated, truncated, _info = env.step(action)
            done = terminated or truncated

            # With render_mode='human' the window updates automatically inside step();
            # the sleep paces the loop at the env's physics rate (~50 Hz).
            if render_this_episode and render_enabled:
                time_module.sleep(0.02)

            # Shape the reward only when we're actually learning
            reward_shaped = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])

            if not render_this_episode:
                # Training-only: build replay buffer and update Q-network
                agent.remember(state, action, reward_shaped, next_state, done)

            state = next_state

            if done:
                scores.append(step + 1)
                label = "eval " if render_this_episode else "train"
                print(f"Episode: {e+1}/{EPISODES} ({label}), Score: {step+1}, ε: {agent.epsilon:.3f}")

                # Print progress bar
                if (e + 1) % 10 == 0:
                    avg_score = np.mean(scores[-10:])
                    print(f"📊 Last 10 episodes average: {avg_score:.1f}")

                break

            # Train the agent only on non-render episodes so render episodes
            # stay smooth (no NN training between frames).
            if not render_this_episode and len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Save model periodically
        if (e + 1) % 50 == 0:
            print(f"💾 Checkpoint at episode {e+1}")
            # Create save directory if it doesn't exist
            # os.makedirs("./save", exist_ok=True)
            # agent.save(f"./save/cartpole-dqn-ep{e+1}.h5")
    
    env.close()
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Final average score (last 10 episodes): {np.mean(scores[-10:]):.1f}")
    print(f"Best score achieved: {max(scores)}")
    print("=" * 50)
    
    # Optional: Plot learning curve
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(scores, alpha=0.6, label='Episode scores')
        
        # Calculate rolling average
        window = 10
        rolling_avg = [np.mean(scores[max(0, i-window+1):i+1]) for i in range(len(scores))]
        plt.plot(rolling_avg, linewidth=2, label=f'{window}-episode average')
        
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.title('DQN Learning Progress on CartPole-v1')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")