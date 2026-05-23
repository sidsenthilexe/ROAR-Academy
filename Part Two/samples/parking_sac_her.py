## This is course material for Introduction to Modern Artificial Intelligence
## Example code: parking_sac_her.py
## Author: ROAR Academy
##
## (c) Copyright 2020-2026. Intelligent Racing Inc. Not permitted for commercial use

"""
Train a goal-conditioned parking policy with SAC + Hindsight Experience Replay.

The environment is highway-env's stock `parking-v0`, which models a
perpendicular parking lot — the agent must drive from the middle lane, turn
roughly 90 degrees, and slot into a target spot between two parked cars. The
same SAC + HER pipeline transfers to a parallel-parking layout by subclassing
`highway_env.envs.parking_env.ParkingEnv` and overriding `_create_road()` /
`_create_vehicles()` to lay out a single row of slots aligned with the
driving direction; that's left as a stretch exercise.

Why SAC + HER?
  * SAC (Soft Actor-Critic) is an off-policy, maximum-entropy algorithm for
    continuous-action control. The entropy bonus in the objective keeps the
    policy exploring during training, which matters for the steering /
    throttle control needed here.
  * HER (Hindsight Experience Replay) re-labels the goal of each stored
    transition with the state actually achieved during the rollout. This
    turns *failed* episodes — where the agent never reaches the real
    parking spot — into useful learning signal, which is what makes
    sparse-reward goal-conditioned tasks like this learnable from scratch
    in a reasonable number of steps.

Note on libraries:
  stable-baselines3 is built on PyTorch (not TensorFlow). Importing this
  script will load torch alongside the TF runtime already used by the rest
  of Part Two; the two frameworks coexist in the same Python process.

Watching the agent learn:
  During training, a pygame window opens every EVAL_FREQ steps to show a
  greedy rollout of the current policy. The first rollout (step 0) is the
  *untrained* baseline — the cart usually thrashes randomly. Subsequent
  rollouts show the policy converging toward a real parking maneuver. The
  main training env stays headless so each checkpoint only costs a few
  seconds of wall-clock time. Toggle RENDER_DURING_TRAINING=False to skip
  the rollouts entirely (useful on a headless CI box).

Usage:
    python parking_sac_her.py            # train (with animation) then run a final eval
    python parking_sac_her.py train      # train only, save to ./parking_sac_her_model.zip
    python parking_sac_her.py eval       # load saved model, run rendered eval episodes
"""

import os
import sys

import gymnasium as gym
import highway_env  # noqa: F401  -- importing this registers parking-v0 with gymnasium
import numpy as np

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback


# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------
# Default budget tuned for a short classroom demo: ~5 minutes on a modern CPU.
# Bump to 100_000 - 200_000 for a polished policy that parks reliably.
TOTAL_TIMESTEPS = 20_000

# Animation knobs:
#   RENDER_DURING_TRAINING — every EVAL_FREQ training steps, pause and play one
#     greedy rollout in a pygame window so students can watch the policy
#     improve. The main training env stays headless for speed; only the eval
#     rollouts render. Set to False to disable (e.g., on a headless CI box).
#   EVAL_FREQ            — number of training steps between rendered rollouts.
#                          With TOTAL_TIMESTEPS=20000 and EVAL_FREQ=2000 you
#                          get one baseline rollout plus 10 progress checkpoints.
#   EVAL_EPISODES        — number of episodes to roll out at each checkpoint.
RENDER_DURING_TRAINING = True
EVAL_FREQ = 2_000
EVAL_EPISODES = 1

# Save the trained model next to this script so `python parking_sac_her.py eval`
# from any working directory still finds the checkpoint.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "parking_sac_her_model",
)


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def make_env(render: bool = False):
    """Construct the highway-env parking environment.

    parking-v0 returns a Dict observation containing `observation`,
    `achieved_goal`, and `desired_goal` — the structure HER expects. The
    Dict observation is also why the SAC policy below is `MultiInputPolicy`.
    """
    render_mode = "human" if render else None
    try:
        return gym.make("parking-v0", render_mode=render_mode)
    except Exception:
        # Fallback if 'human' rendering can't open a window (headless CI).
        return gym.make("parking-v0", render_mode="rgb_array")


# ---------------------------------------------------------------------------
# Rendered-eval callback
# ---------------------------------------------------------------------------
class RenderEvalCallback(BaseCallback):
    """Periodically pause training and roll out the current policy with rendering.

    SB3's `model.learn(callback=...)` argument hooks user code into the training
    loop. This callback opens one pygame window at training start, runs a single
    greedy (no-exploration) eval episode there at training start (so students
    see the *untrained* baseline), then again every `eval_freq` training steps,
    and closes the window at training end. The main training env stays headless
    the whole time, so this only adds a few seconds per checkpoint.
    """

    def __init__(self, eval_freq: int = 2_000, n_eval_episodes: int = 1, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_env = None
        # _on_training_start handles the untrained baseline rollout; we then
        # want the *next* rollout to fire at step >= eval_freq, so start the
        # counter at 0 (the step number of the baseline).
        self._last_eval_step = 0

    # SB3 calls this once, just before model.learn() starts iterating.
    def _on_training_start(self) -> None:
        self.eval_env = make_env(render=True)
        self._run_eval(label="step 0 (untrained baseline)")

    # SB3 calls this after every environment step taken by the training env.
    def _on_step(self) -> bool:
        steps = self.num_timesteps
        if steps - self._last_eval_step >= self.eval_freq:
            # Record the actual step we fired on so the next trigger requires
            # another full `eval_freq` steps (no double-fires at adjacent steps).
            self._last_eval_step = steps
            self._run_eval(label=f"step {steps}")
        return True  # returning False would stop training early

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
            self.eval_env = None

    def _run_eval(self, label: str) -> None:
        if self.eval_env is None:
            return
        print(f"\n--- Rendered eval at {label} ---")
        for ep in range(self.n_eval_episodes):
            obs, _ = self.eval_env.reset()
            terminated = truncated = False
            total_reward = 0.0
            steps = 0
            # Safety cap mirrors parking-v0's default episode duration so a
            # totally untrained policy can't stall the demo indefinitely.
            while not (terminated or truncated) and steps < 150:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = self.eval_env.step(action)
                total_reward += float(reward)
                steps += 1
            outcome = "parked" if terminated and not truncated else "timed out"
            print(f"  ep{ep + 1}: {steps:3d} steps, reward = {total_reward:7.2f}  ({outcome})")


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train() -> SAC:
    env = make_env(render=False)

    # Hyperparameters below mirror the SAC+HER configuration recommended by
    # the highway-env documentation for parking-v0. The big knobs that matter:
    #
    #   * MultiInputPolicy — handles the Dict observation produced by the env.
    #   * replay_buffer_class=HerReplayBuffer — turns the standard replay
    #     buffer into one that synthesizes hindsight goals at sample time.
    #   * n_sampled_goal=4 with goal_selection_strategy="future" — for each
    #     real transition, sample 4 additional hindsight goals from later
    #     states in the same trajectory. This is the recipe from the original
    #     HER paper (Andrychowicz et al., 2017).
    #   * Three 256-unit hidden layers is more than usually needed for low-
    #     dimensional control tasks, but it gives SAC enough capacity to
    #     model the multi-modal action distribution induced by HER.
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        buffer_size=int(1e5),
        learning_rate=1e-3,
        gamma=0.95,
        batch_size=256,
        tau=0.05,
        learning_starts=1_000,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    print(f"\nTraining SAC + HER on parking-v0 for {TOTAL_TIMESTEPS} timesteps...")
    print("(Bump TOTAL_TIMESTEPS to ~100_000+ for a noticeably better policy.)")
    if RENDER_DURING_TRAINING:
        print(f"A pygame window will open every {EVAL_FREQ} training steps to show "
              f"a greedy rollout of the current policy — that's the in-progress "
              f"animation of the learning process.\n")
    else:
        print("(Set RENDER_DURING_TRAINING=True for in-progress rendered rollouts.)\n")

    callback = RenderEvalCallback(
        eval_freq=EVAL_FREQ,
        n_eval_episodes=EVAL_EPISODES,
    ) if RENDER_DURING_TRAINING else None

    model.learn(total_timesteps=TOTAL_TIMESTEPS, log_interval=10, callback=callback)

    model.save(MODEL_PATH)
    print(f"\nSaved trained model to {MODEL_PATH}.zip")

    env.close()
    return model


# ---------------------------------------------------------------------------
# Evaluation (with rendering)
# ---------------------------------------------------------------------------
def evaluate(num_episodes: int = 5) -> None:
    """Roll out the trained policy with rendering so you can watch it park."""
    env = make_env(render=True)
    model = SAC.load(MODEL_PATH, env=env)

    episode_rewards = []
    for ep in range(num_episodes):
        obs, _ = env.reset()
        terminated = truncated = False
        total_reward = 0.0
        steps = 0
        while not (terminated or truncated):
            # `deterministic=True` disables the SAC entropy noise so the eval
            # rollouts reflect the learned mean policy, not its exploration noise.
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += float(reward)
            steps += 1
        episode_rewards.append(total_reward)
        outcome = "parked" if terminated and not truncated else "timed out"
        print(f"Episode {ep + 1}: {steps:3d} steps, reward = {total_reward:7.2f}  ({outcome})")

    print(f"\nMean evaluation reward over {num_episodes} episodes: "
          f"{np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    if mode == "train":
        train()
        print("\nRunning a short rendered evaluation on the freshly trained policy...\n")
        evaluate(num_episodes=3)
    elif mode == "eval":
        if not os.path.exists(MODEL_PATH + ".zip"):
            print(
                f"No saved model found at {MODEL_PATH}.zip — run\n"
                f"    python {os.path.basename(__file__)} train\n"
                "first to produce one."
            )
            sys.exit(1)
        evaluate(num_episodes=5)
    else:
        print(f"Unknown mode: {mode!r}. Use 'train' or 'eval'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
