"""
PPO training for JudgementEnv (Judgement card game).

Supports two training modes:
  --mode heuristic   (default) train against fixed heuristic opponents
  --mode selfplay              train against a rolling pool of past checkpoints

Dependencies:
    pip install torch numpy gymnasium matplotlib

Usage:
    python train_ppo.py                        # heuristic mode
    python train_ppo.py --mode selfplay        # self-play mode
    python train_ppo.py --eval                 # eval a saved checkpoint
"""

import argparse
import os
import time
import random
from copy import deepcopy
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

from environment import JudgementEnv, OBS_DIM, _MAX_SCORE

CFG = dict(
    # Environment
    max_cards       = 17,

    # Rollout
    n_envs          = 16,
    rollout_steps   = 512,

    # Training
    total_timesteps = 8_000_000,
    lr              = 3e-4,
    gamma           = 0.99,
    gae_lambda      = 0.95,
    clip_eps        = 0.2,
    entropy_coef    = 0.10,
    entropy_min     = 0.01,
    value_coef      = 0.5,
    max_grad_norm   = 0.5,
    n_epochs        = 4,
    batch_size      = 256,

    # Network
    hidden          = 256,

    # Self-play pool
    pool_update_interval = 50,   # rollouts between pool snapshots
    pool_max_size        = 10,   # max frozen checkpoints kept
    pool_current_frac    = 0.3,  # prob of fighting *current* model vs pool

    # Logging / saving
    log_interval    = 10,
    save_interval   = 100,
    checkpoint_path = "ppo_judgement_{mode}.pt",
    plot_path       = "training_curve_{mode}.png",
)

def random_baseline(max_cards: int = 17, n_games: int = 200, seed: int = 42) -> float:
    env = JudgementEnv(max_cards=max_cards)
    totals = []
    for g in range(n_games):
        obs, info = env.reset(seed=seed + g)
        done, total = False, 0.0
        while not done:
            legal  = np.where(info['action_mask'])[0]
            action = int(np.random.choice(legal))
            obs, r, terminated, truncated, info = env.step(action)
            done   = terminated or truncated
            total += r
        totals.append(total)
    mean = float(np.mean(totals))
    print(f"Random baseline ({n_games} games): {mean:.2f}")
    return mean


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = 52, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
            nn.Linear(hidden,  hidden), nn.Tanh(),
        )
        self.actor  = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

        for layer in self.shared:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.actor.weight,  gain=0.01)
        nn.init.zeros_(self.actor.bias)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)
        nn.init.zeros_(self.critic.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        h      = self.shared(x)
        logits = self.actor(h).masked_fill(~mask, float('-inf'))
        value  = self.critic(h).squeeze(-1)
        return logits, value

    @torch.no_grad()
    def act(self, x: torch.Tensor, mask: torch.Tensor):
        logits, value = self(x, mask)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        log_p  = dist.log_prob(action)
        return action, log_p, value

class VecJudgement:
    # Synchronous vector of JudgementEnv instances

    def __init__(self, n_envs: int, max_cards: int = 17, seed: int = 0,
                 opponent_model=None):
        self.envs   = [JudgementEnv(max_cards=max_cards, opponent_model=opponent_model)
                       for _ in range(n_envs)]
        self.n_envs = n_envs
        self.obs    = np.zeros((n_envs, OBS_DIM), dtype=np.float32)
        self.masks  = np.zeros((n_envs, 52),      dtype=bool)

        for i, env in enumerate(self.envs):
            o, info = env.reset(seed=seed + i)
            self.obs[i]   = o
            self.masks[i] = info['action_mask']

    def set_opponent(self, model):
        # Hot-swap opponent model in all envs (takes effect from next episode).
        for env in self.envs:
            env.opponent_model = model

    def step(self, actions: np.ndarray):
        next_obs   = np.zeros_like(self.obs)
        rewards    = np.zeros(self.n_envs, dtype=np.float32)
        dones      = np.zeros(self.n_envs, dtype=bool)
        next_masks = np.zeros_like(self.masks)

        for i, (env, a) in enumerate(zip(self.envs, actions)):
            o, r, terminated, truncated, info = env.step(int(a))
            done = terminated or truncated
            if done:
                o, info = env.reset()
            next_obs[i]   = o
            rewards[i]    = r
            dones[i]      = done
            next_masks[i] = info['action_mask']

        self.obs   = next_obs
        self.masks = next_masks
        return next_obs, rewards, dones, next_masks


class PolicyPool:
    # Keeps a rolling buffer of frozen past checkpoints.
    # sample() returns a CPU-eval ActorCritic drawn uniformly from the pool,
    # or a live copy of the current model with probability `current_frac`.

    def __init__(self, hidden: int, max_size: int = 10, current_frac: float = 0.3):
        self.hidden       = hidden
        self.max_size     = max_size
        self.current_frac = current_frac
        self._pool: list[dict] = []   # list of state_dicts

    def add(self, model: ActorCritic):
        sd = deepcopy(model.state_dict())
        # Move all tensors to CPU so they don't eat GPU memory
        sd = {k: v.cpu() for k, v in sd.items()}
        self._pool.append(sd)
        if len(self._pool) > self.max_size:
            self._pool.pop(0)

    def sample(self, current_model: ActorCritic) -> ActorCritic:
        """Return a frozen opponent model (always on CPU, eval mode)."""
        use_current = (not self._pool) or (random.random() < self.current_frac)
        opp = ActorCritic(hidden=self.hidden).cpu()
        if use_current:
            opp.load_state_dict({k: v.cpu() for k, v in current_model.state_dict().items()})
        else:
            opp.load_state_dict(random.choice(self._pool))
        opp.eval()
        return opp

    def __len__(self):
        return len(self._pool)


def compute_gae(rewards, values, dones, last_value,
                gamma: float, lam: float) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape
    advantages = torch.zeros_like(rewards)
    last_gae   = torch.zeros(N, device=rewards.device)

    for t in reversed(range(T)):
        next_val  = last_value if t == T - 1 else values[t + 1]
        mask_cont = 1.0 - dones[t].float()
        delta     = rewards[t] + gamma * next_val * mask_cont - values[t]
        last_gae  = delta + gamma * lam * mask_cont * last_gae
        advantages[t] = last_gae

    return advantages, advantages + values


def train(cfg: dict, mode: str = 'heuristic'):
    # mode: 'heuristic'  — opponents always use hand-coded heuristics
    # 'selfplay'   — opponents are sampled from a rolling checkpoint pool
    
    assert mode in ('heuristic', 'selfplay'), f"Unknown mode: {mode}"

    checkpoint_path = cfg['checkpoint_path'].format(mode=mode)
    plot_path       = cfg['plot_path'].format(mode=mode)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Mode: {mode}  |  Training on {device}\n")

    rand_score = random_baseline(cfg['max_cards'])

    # In heuristic mode, opponent_model=None → env uses heuristics
    venv  = VecJudgement(cfg['n_envs'], cfg['max_cards'], opponent_model=None)
    model = ActorCritic(hidden=cfg['hidden']).to(device)
    opt   = optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-5)

    T, N       = cfg['rollout_steps'], cfg['n_envs']
    batch_size = cfg['batch_size']
    n_rollouts = cfg['total_timesteps'] // (T * N)

    total_updates = n_rollouts * cfg['n_epochs']
    scheduler = optim.lr_scheduler.LinearLR(
        opt, start_factor=1.0, end_factor=0.1, total_iters=total_updates
    )

    # Self-play state
    pool = PolicyPool(
        hidden       = cfg['hidden'],
        max_size     = cfg['pool_max_size'],
        current_frac = cfg['pool_current_frac'],
    ) if mode == 'selfplay' else None

    # Rollout buffers
    b_obs    = torch.zeros(T, N, OBS_DIM, device=device)
    b_masks  = torch.zeros(T, N, 52, device=device, dtype=torch.bool)
    b_acts   = torch.zeros(T, N, device=device, dtype=torch.long)
    b_logp   = torch.zeros(T, N, device=device)
    b_vals   = torch.zeros(T, N, device=device)
    b_rews   = torch.zeros(T, N, device=device)
    b_dones  = torch.zeros(T, N, device=device)

    ep_rewards        = deque(maxlen=100)
    ep_running_totals = np.zeros(N, dtype=np.float32)
    all_mean_rews     = []
    best_mean_rew     = -float('inf')

    total_steps = 0
    t0 = time.time()

    for rollout in range(n_rollouts):

        # Self-play: refresh opponent every N rollouts
        if mode == 'selfplay' and rollout % cfg['pool_update_interval'] == 0:
            opp = pool.sample(model)
            venv.set_opponent(opp)
            if len(pool) == 0:
                print(f"  [self-play] rollout {rollout}: initial heuristic warm-up phase")
            else:
                print(f"  [self-play] rollout {rollout}: new opponent sampled "
                      f"(pool size={len(pool)})")

        # Collect rollout
        obs_np   = venv.obs.copy()
        masks_np = venv.masks.copy()

        for t in range(T):
            obs_t  = torch.from_numpy(obs_np).to(device)
            mask_t = torch.from_numpy(masks_np).to(device)

            with torch.no_grad():
                acts, log_p, vals = model.act(obs_t, mask_t)

            next_obs, rews, dones, next_masks = venv.step(acts.cpu().numpy())

            b_obs[t]   = obs_t
            b_masks[t] = mask_t
            b_acts[t]  = acts
            b_logp[t]  = log_p
            b_vals[t]  = vals
            b_rews[t]  = torch.from_numpy(rews).to(device)
            b_dones[t] = torch.from_numpy(dones.astype(np.float32)).to(device)

            ep_running_totals += rews
            for i, done in enumerate(dones):
                if done:
                    ep_rewards.append(ep_running_totals[i])
                    ep_running_totals[i] = 0.0

            obs_np   = next_obs
            masks_np = next_masks

        total_steps += T * N

        # Self-play: add snapshot to pool after rollout
        if mode == 'selfplay' and rollout % cfg['pool_update_interval'] == 0:
            pool.add(model)

        # Bootstrap
        with torch.no_grad():
            _, last_val = model(
                torch.from_numpy(obs_np).to(device),
                torch.from_numpy(masks_np).to(device),
            )

        # GAE 
        adv, returns = compute_gae(b_rews, b_vals, b_dones, last_val,
                                   cfg['gamma'], cfg['gae_lambda'])
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        flat_obs     = b_obs.view(-1, OBS_DIM)
        flat_masks   = b_masks.view(-1, 52)
        flat_acts    = b_acts.view(-1)
        flat_old_lp  = b_logp.view(-1)
        flat_adv     = adv.view(-1)
        flat_returns = returns.view(-1)

        # PPO update
        frac         = 1.0 - (total_steps / cfg['total_timesteps'])
        entropy_coef = cfg['entropy_min'] + frac * (cfg['entropy_coef'] - cfg['entropy_min'])

        idxs = np.arange(T * N)
        for _ in range(cfg['n_epochs']):
            np.random.shuffle(idxs)
            for start in range(0, T * N, batch_size):
                mb = torch.from_numpy(idxs[start: start + batch_size]).to(device)

                logits, val = model(flat_obs[mb], flat_masks[mb])
                dist        = Categorical(logits=logits)
                new_lp      = dist.log_prob(flat_acts[mb])
                entropy     = dist.entropy()

                ratio      = (new_lp - flat_old_lp[mb]).exp()
                adv_mb     = flat_adv[mb]
                surr1      = ratio * adv_mb
                surr2      = ratio.clamp(1 - cfg['clip_eps'], 1 + cfg['clip_eps']) * adv_mb
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = nn.functional.mse_loss(val, flat_returns[mb])
                loss = actor_loss + cfg['value_coef'] * critic_loss - entropy_coef * entropy.mean()

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg['max_grad_norm'])
                opt.step()
                scheduler.step()

        # Logging
        mean_rew = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        all_mean_rews.append(mean_rew)

        if (rollout + 1) % cfg['log_interval'] == 0:
            elapsed = time.time() - t0
            fps     = total_steps / elapsed
            vs_rand = mean_rew - rand_score
            print(f"[{mode}] Rollout {rollout+1:4d}/{n_rollouts}  "
                  f"steps={total_steps:,}  "
                  f"mean_ep_ret={mean_rew:6.2f}  "
                  f"vs_random={vs_rand:+.2f}  "
                  f"entropy_coef={entropy_coef:.4f}  "
                  f"fps={fps:.0f}")

        if (rollout + 1) % cfg['save_interval'] == 0:
            if mean_rew > best_mean_rew:
                best_mean_rew = mean_rew
                torch.save({
                    'model':      model.state_dict(),
                    'optimizer':  opt.state_dict(),
                    'cfg':        cfg,
                    'mode':       mode,
                    'rollout':    rollout,
                    'rand_score': rand_score,
                }, checkpoint_path)
                print(f"  ✓ New best ({mean_rew:.2f}, +{mean_rew - rand_score:.2f} vs random) "
                      f"— saved to {checkpoint_path}")

    # Final save
    torch.save({
        'model':      model.state_dict(),
        'cfg':        cfg,
        'mode':       mode,
        'rollout':    n_rollouts,
        'rand_score': rand_score,
        'all_mean_rews': all_mean_rews,
    }, checkpoint_path)

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(all_mean_rews, linewidth=1.5,
             label=f'Mean episode return ({mode})')
    plt.axhline(rand_score, color='orange', linestyle='--',
                label=f'Random baseline ({rand_score:.1f})')
    plt.xlabel('Rollout')
    plt.ylabel('Episode Return')
    plt.title(f'PPO on Judgement — {mode} training')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"\nTraining complete. Curve saved to {plot_path}")

    return model, all_mean_rews, rand_score


# Evaluation
def evaluate(checkpoint_path: str, n_games: int = 50):
    device = torch.device("cpu")
    ckpt   = torch.load(checkpoint_path, map_location=device)
    cfg    = ckpt['cfg']
    mode   = ckpt.get('mode', 'unknown')

    model = ActorCritic(hidden=cfg['hidden']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    env    = JudgementEnv(max_cards=cfg['max_cards'])
    scores = []

    for g in range(n_games):
        obs, info = env.reset(seed=1000 + g)
        done, total = False, 0.0

        while not done:
            obs_t  = torch.from_numpy(obs).unsqueeze(0)
            mask_t = torch.from_numpy(info['action_mask']).unsqueeze(0)

            with torch.no_grad():
                logits, _ = model(obs_t, mask_t)
                action    = int(logits.argmax(dim=-1).item())

            obs, r, terminated, truncated, info = env.step(action)
            done   = terminated or truncated
            total += r

        scores.append(total)
        print(f"Game {g+1:3d}: {total:6.2f}")

    rand_score = ckpt.get('rand_score') or random_baseline(cfg['max_cards'])

    print(f"\n{'─'*40}")
    print(f"Mode           : {mode}")
    print(f"Games          : {n_games}")
    print(f"Mean return    : {np.mean(scores):.2f}")
    print(f"Std            : {np.std(scores):.2f}")
    print(f"Random baseline: {rand_score:.2f}")
    print(f"vs Random      : {np.mean(scores) - rand_score:+.2f}")
    print(f"Max possible   : {_MAX_SCORE}")
    print(f"% of max       : {100 * np.mean(scores) / _MAX_SCORE:.1f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['heuristic', 'selfplay'],
                        default='heuristic',
                        help='Training mode: heuristic opponents or self-play pool')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate a saved checkpoint instead of training')
    parser.add_argument('--checkpoint', default=None,
                        help='Path to checkpoint (for --eval). '
                             'Defaults to ppo_judgement_{mode}.pt')
    parser.add_argument('--games', type=int, default=50)
    args = parser.parse_args()

    if args.eval:
        ckpt_path = args.checkpoint or CFG['checkpoint_path'].format(mode=args.mode)
        evaluate(ckpt_path, args.games)
    else:
        train(CFG, mode=args.mode)
