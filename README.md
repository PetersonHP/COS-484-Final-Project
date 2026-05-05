# COS-484-Final-Project
# Judgement RL

A PPO agent trained to play Judgement, a 3-player trick-taking card game, with support for heuristic-opponent training and self-play.

## The Game

Judgement is played over 17 rounds. Round 1 deals 17 cards to each player, round 2 deals 16, down to 1 in round 17. Each round has two phases.

**Bidding:** Each player predicts how many tricks they will win that round. The dealer (seat 2) has a hard constraint — their bid cannot make the total of all bids equal the number of cards in play, so at least one player is always "off".

**Playing:** Players take turns playing cards and must follow the lead suit if possible. The highest trump card wins the trick; otherwise the highest card of the lead suit wins. Trump cycles each round: Spades → Diamonds → Clubs → Hearts → repeat.

**Scoring:** You score `10 + bid` points if tricks won exactly equals your bid. Over- or under-shooting scores zero. The theoretical maximum across all 17 rounds is 323 points.

## Files

| File | Purpose |
|---|---|
| `environment.py` | Gymnasium environment |
| `train_ppo.py` | PPO training (heuristic and self-play modes) |
| `compare_models.py` | Head-to-head evaluation and plots |

## Installation

```bash
pip install torch numpy gymnasium matplotlib pygame
```

## Usage

### Training

Two training modes are available via `--mode`.

**Heuristic mode** trains the agent against fixed rule-based opponents throughout. The heuristic bidder counts aces and high trump cards to estimate tricks; the heuristic player tries to win tricks cheaply and discards its lowest card when it cannot win.

**Self-play mode** replaces opponents with a rolling pool of past model checkpoints. Every 50 rollouts the current model is added to the pool (capped at 10 entries). When an opponent is sampled, there is a 70% chance a random past checkpoint is used and a 30% chance the current live model is used. This mixture prevents the agent from overfitting to only the latest version of itself and avoids the cyclic dynamics that pure self-play produces (A beats B beats A beats A...).

The recommended approach is to run heuristic training first to build basic game competency, then switch to self-play.

```bash
python train_ppo.py --mode heuristic   # phase 1
python train_ppo.py --mode selfplay    # phase 2
```

Checkpoints are saved to `ppo_judgement_heuristic.pt` and `ppo_judgement_selfplay.pt`. The best checkpoint by mean episode return is saved automatically during training. To evaluate a saved checkpoint:

```bash
python train_ppo.py --eval --mode heuristic
python train_ppo.py --eval --mode selfplay --games 100
```

Key hyperparameters are in the `CFG` dict at the top of `train_ppo.py`:

| Parameter | Default | Notes |
|---|---|---|
| `total_timesteps` | 8,000,000 | Total environment steps |
| `n_envs` | 16 | Parallel environments |
| `rollout_steps` | 512 | Steps collected per rollout |
| `lr` | 3e-4 | Decays linearly to 10% by end of training |
| `entropy_coef` | 0.10 | Anneals to 0.01 over training |
| `hidden` | 256 | Hidden layer size (3 layers, Tanh) |
| `pool_max_size` | 10 | Max past checkpoints kept in self-play pool |
| `pool_current_frac` | 0.3 | Probability of fighting current model vs pool |

### Comparing Models

After both training runs are complete, `compare_models.py` runs four evaluation scenarios and saves six figures to a directory:

```bash
python compare_models.py                        # defaults, 100 games each
python compare_models.py --games 200            # more games for tighter estimates
python compare_models.py --out my_figures/      # custom output directory
python compare_models.py --no_plot              # print table only
```

The four scenarios are:

| Scenario | P1 (agent) | P2 + Dealer |
|---|---|---|
| Solo — heuristic model | Heuristic-trained | Heuristics |
| Solo — self-play model | Self-play-trained | Heuristics |
| H2H — heuristic as P1 | Heuristic-trained | Self-play-trained |
| H2H — self-play as P1 | Self-play-trained | Heuristic-trained |

In every scenario P2 and Dealer are always the same type. The six output figures cover training curves, score distributions, mean return summaries, both head-to-head histograms, and top-scorer win rate.

### Running the Environment Directly

```bash
python environment.py              # headless smoke test, 5 games
python environment.py --render     # opens pygame window
python environment.py --render --games 1
```

## Architecture

The agent is a shared-trunk actor-critic network: three hidden layers of 256 units with Tanh activations and orthogonal initialisation, with separate linear heads for the policy and value function. The policy output is masked to illegal actions before sampling.

The observation is 129-dimensional and encodes the agent's hand, cards played in completed tricks, the current trick in progress, bids and tricks won for all players, trump and lead suit, game phase, round progress, and cumulative scores. All per-player slots are rotated so index 0 always means "self", which allows the same weights to be used from any seat — a requirement for self-play where the trained model controls P2 and Dealer as well as P1.
