"""
compare_models.py — head-to-head comparison of heuristic-trained vs self-play-trained agents.

What this does:
  1. Loads both checkpoints
  2. Runs each agent solo against the heuristic opponents (apples-to-apples baseline)
  3. Runs them HEAD-TO-HEAD: heuristic-model as P1, selfplay-model as P2+Dealer, and vice versa
  4. Plots training curves side-by-side (if both checkpoints saved curve data)
  5. Prints a clean summary table

Usage:
    python compare_models.py
    python compare_models.py --heuristic_ckpt ppo_judgement_heuristic.pt \
                             --selfplay_ckpt  ppo_judgement_selfplay.pt  \
                             --games 100
"""

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import defaultdict

from environment import JudgementEnv, OBS_DIM, _MAX_SCORE
from train_ppo import ActorCritic, random_baseline


# helpers

def load_model(path: str, device='cpu') -> tuple[ActorCritic, dict]:
    ckpt  = torch.load(path, map_location=device)
    cfg   = ckpt['cfg']
    model = ActorCritic(hidden=cfg['hidden']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, ckpt


def greedy_action(model: ActorCritic, obs: np.ndarray, mask: np.ndarray) -> int:
    obs_t  = torch.from_numpy(obs).unsqueeze(0)
    mask_t = torch.from_numpy(mask).unsqueeze(0)
    with torch.no_grad():
        logits, _ = model(obs_t, mask_t)
    return int(logits.argmax(dim=-1).item())


# evaulation modes

def eval_vs_heuristics(model: ActorCritic, n_games: int = 100,
                       max_cards: int = 17, seed_offset: int = 0) -> list[float]:
    #  Standard eval: agent plays P1 against heuristic P2+Dealer.
    # Returns list of episode returns.
    env    = JudgementEnv(max_cards=max_cards, opponent_model=None)
    scores = []

    for g in range(n_games):
        obs, info = env.reset(seed=seed_offset + g)
        done, total = False, 0.0
        while not done:
            action = greedy_action(model, obs, info['action_mask'])
            obs, r, terminated, truncated, info = env.step(action)
            done   = terminated or truncated
            total += r
        scores.append(total)

    return scores


def eval_head_to_head(model_p1: ActorCritic, model_opp: ActorCritic,
                      n_games: int = 100, max_cards: int = 17,
                      seed_offset: int = 0) -> dict:
    # model_p1  plays as the agent (P1, seat 0).
    # model_opp plays as P2 + Dealer via opponent_model injection.

   # Returns dict with scores for all three seats.
    
    env = JudgementEnv(max_cards=max_cards, opponent_model=model_opp)

    p1_scores      = []
    opp_p2_scores  = []
    opp_d_scores   = []

    for g in range(n_games):
        obs, info = env.reset(seed=seed_offset + g)
        done, total = False, 0.0
        while not done:
            action = greedy_action(model_p1, obs, info['action_mask'])
            obs, r, terminated, truncated, info = env.step(action)
            done   = terminated or truncated
            total += r

        finals = info['cumulative_scores']
        p1_scores.append(finals[0])
        opp_p2_scores.append(finals[1])
        opp_d_scores.append(finals[2])

    return {
        'p1':      np.array(p1_scores),
        'opp_p2':  np.array(opp_p2_scores),
        'opp_d':   np.array(opp_d_scores),
    }


def plot_comparison(
    h_scores_vs_heuristic: list[float],
    sp_scores_vs_heuristic: list[float],
    h_as_p1_vs_sp: dict,
    sp_as_p1_vs_h: dict,
    rand_score: float,
    h_curve: list | None,
    sp_curve: list | None,
    save_dir: str = ".",
) -> list[str]:
    # Saves each panel as its own file. Returns list of saved paths.
    import os
    os.makedirs(save_dir, exist_ok=True)

    BLUE   = '#3b82f6'
    ORANGE = '#f97316'
    GREEN  = '#22c55e'
    GREY   = '#6b7280'

    saved = []

    def savefig(fig, name):
        path = os.path.join(save_dir, name)
        fig.savefig(path, dpi=130, bbox_inches='tight')
        plt.close(fig)
        saved.append(path)
        print(f"  Saved: {path}")

    # Training curves
    fig, ax = plt.subplots(figsize=(7, 4))
    if h_curve and sp_curve:
        ax.plot(h_curve,  color=BLUE,   linewidth=1.4, label='Heuristic')
        ax.plot(sp_curve, color=ORANGE, linewidth=1.4, label='Self-play')
        ax.axhline(rand_score, color=GREY, linestyle='--', linewidth=1,
                   label=f'Random ({rand_score:.0f})')
        ax.legend(fontsize=8)
    elif h_curve:
        ax.plot(h_curve, color=BLUE, linewidth=1.4, label='Heuristic')
        ax.axhline(rand_score, color=GREY, linestyle='--', linewidth=1)
    elif sp_curve:
        ax.plot(sp_curve, color=ORANGE, linewidth=1.4, label='Self-play')
        ax.axhline(rand_score, color=GREY, linestyle='--', linewidth=1)
    else:
        ax.text(0.5, 0.5, 'No curve data\nin checkpoints',
                ha='center', va='center', transform=ax.transAxes, color=GREY)
    ax.set_xlabel('Rollout', fontsize=9)
    ax.set_ylabel('Mean episode return', fontsize=9)
    ax.set_title('Training curves', fontsize=10, fontweight='bold')
    savefig(fig, 'fig1_training_curves.png')

    # Score distributions vs heuristics
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(h_scores_vs_heuristic,  bins=20, alpha=0.65, color=BLUE,
            label=f'Heuristic  μ={np.mean(h_scores_vs_heuristic):.1f}')
    ax.hist(sp_scores_vs_heuristic, bins=20, alpha=0.65, color=ORANGE,
            label=f'Self-play  μ={np.mean(sp_scores_vs_heuristic):.1f}')
    ax.axvline(rand_score, color=GREY, linestyle='--', linewidth=1.2,
               label=f'Random ({rand_score:.0f})')
    ax.set_xlabel('Episode return', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title('Solo score vs heuristic opponents', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    savefig(fig, 'fig2_solo_score_distributions.png')

    # 3. Mean returns bar chart
    fig, ax = plt.subplots(figsize=(7, 4))
    labels = ['Heuristic\nvs heuristic', 'Self-play\nvs heuristic',
              'Heuristic\nvs self-play', 'Self-play\nvs heuristic\n(as P1)']
    means  = [
        np.mean(h_scores_vs_heuristic),
        np.mean(sp_scores_vs_heuristic),
        np.mean(h_as_p1_vs_sp['p1']),
        np.mean(sp_as_p1_vs_h['p1']),
    ]
    stds   = [
        np.std(h_scores_vs_heuristic),
        np.std(sp_scores_vs_heuristic),
        np.std(h_as_p1_vs_sp['p1']),
        np.std(sp_as_p1_vs_h['p1']),
    ]
    colors = [BLUE, ORANGE, BLUE, ORANGE]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors,
                  alpha=0.8, width=0.6, error_kw=dict(elinewidth=1.2))
    ax.axhline(rand_score, color=GREY, linestyle='--', linewidth=1.2,
               label=f'Random ({rand_score:.0f})')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel('Mean episode return', fontsize=9)
    ax.set_title('Mean returns (± std)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=8)
    savefig(fig, 'fig3_mean_returns_summary.png')

    # 4. H2H: Heuristic as P1 vs Self-play
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(h_as_p1_vs_sp['p1'],     bins=20, alpha=0.65, color=BLUE,
            label=f'Heuristic (P1)  μ={np.mean(h_as_p1_vs_sp["p1"]):.1f}')
    ax.hist(h_as_p1_vs_sp['opp_p2'], bins=20, alpha=0.55, color=ORANGE,
            label=f'Self-play (P2)  μ={np.mean(h_as_p1_vs_sp["opp_p2"]):.1f}')
    ax.hist(h_as_p1_vs_sp['opp_d'],  bins=20, alpha=0.45, color=GREEN,
            label=f'Self-play (D)   μ={np.mean(h_as_p1_vs_sp["opp_d"]):.1f}')
    ax.set_xlabel('Final game score', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title('H2H: Heuristic(P1) vs Self-play(P2+D)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    savefig(fig, 'fig4_h2h_heuristic_as_p1.png')

    # 5. H2H: Self-play as P1 vs Heuristic 
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sp_as_p1_vs_h['p1'],     bins=20, alpha=0.65, color=ORANGE,
            label=f'Self-play (P1)  μ={np.mean(sp_as_p1_vs_h["p1"]):.1f}')
    ax.hist(sp_as_p1_vs_h['opp_p2'], bins=20, alpha=0.55, color=BLUE,
            label=f'Heuristic (P2)  μ={np.mean(sp_as_p1_vs_h["opp_p2"]):.1f}')
    ax.hist(sp_as_p1_vs_h['opp_d'],  bins=20, alpha=0.45, color=GREEN,
            label=f'Heuristic (D)   μ={np.mean(sp_as_p1_vs_h["opp_d"]):.1f}')
    ax.set_xlabel('Final game score', fontsize=9)
    ax.set_ylabel('Frequency', fontsize=9)
    ax.set_title('H2H: Self-play(P1) vs Heuristic(P2+D)', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    savefig(fig, 'fig5_h2h_selfplay_as_p1.png')

    # 6. Top-score win rate
    fig, ax = plt.subplots(figsize=(7, 4))

    def first_place(scores_dict: dict) -> np.ndarray:
        p1  = scores_dict['p1']
        opp = np.maximum(scores_dict['opp_p2'], scores_dict['opp_d'])
        return p1 > opp

    h_wins  = first_place(h_as_p1_vs_sp).mean() * 100
    sp_wins = first_place(sp_as_p1_vs_h).mean() * 100
    categories = ['Heuristic\nwins as P1\nvs self-play', 'Self-play\nwins as P1\nvs heuristic']
    values     = [h_wins, sp_wins]
    ax.bar(categories, values, color=[BLUE, ORANGE], alpha=0.85, width=0.5)
    ax.set_ylim(0, 100)
    ax.axhline(33.3, color=GREY, linestyle='--', linewidth=1.2,
               label='Random chance (33%)')
    ax.set_ylabel('% games scoring highest', fontsize=9)
    ax.set_title('Top-score rate vs. opponents', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    for i, val in enumerate(values):
        ax.text(i, val + 1.5, f'{val:.1f}%', ha='center', fontsize=10, fontweight='bold')
    savefig(fig, 'fig6_topscorer_winrate.png')

    print(f"\n[compare] {len(saved)} figures saved to '{save_dir}/'")
    return saved
# Summary table

def print_summary(
    h_scores:  list[float],
    sp_scores: list[float],
    h_as_p1:   dict,
    sp_as_p1:  dict,
    rand_score: float,
):
    def win_rate(d):
        p1  = d['p1']
        opp = np.maximum(d['opp_p2'], d['opp_d'])
        return (p1 > opp).mean() * 100

    rows = [
        ("Metric", "Heuristic", "Self-play"),
        ("─" * 28, "─" * 12, "─" * 12),
        ("Solo vs heuristic — mean",
         f"{np.mean(h_scores):.1f}", f"{np.mean(sp_scores):.1f}"),
        ("Solo vs heuristic — std",
         f"{np.std(h_scores):.1f}", f"{np.std(sp_scores):.1f}"),
        ("Solo vs heuristic — % of max",
         f"{100*np.mean(h_scores)/_MAX_SCORE:.1f}%",
         f"{100*np.mean(sp_scores)/_MAX_SCORE:.1f}%"),
        ("vs Random (solo)",
         f"{np.mean(h_scores)-rand_score:+.1f}",
         f"{np.mean(sp_scores)-rand_score:+.1f}"),
        ("H2H as P1 — mean score",
         f"{np.mean(h_as_p1['p1']):.1f}",
         f"{np.mean(sp_as_p1['p1']):.1f}"),
        ("H2H win rate (top scorer)",
         f"{win_rate(h_as_p1):.1f}%",
         f"{win_rate(sp_as_p1):.1f}%"),
        ("Random baseline", f"{rand_score:.1f}", f"{rand_score:.1f}"),
        ("Max possible", f"{_MAX_SCORE}", f"{_MAX_SCORE}"),
    ]

    col_w = [32, 14, 14]
    print("\n" + "═" * sum(col_w))
    print("  MODEL COMPARISON SUMMARY")
    print("═" * sum(col_w))
    for row in rows:
        print("".join(str(cell).ljust(w) for cell, w in zip(row, col_w)))
    print("═" * sum(col_w))
    print()

    # Verdict
    h_mean  = np.mean(h_scores)
    sp_mean = np.mean(sp_scores)
    h_h2h   = np.mean(h_as_p1['p1'])
    sp_h2h  = np.mean(sp_as_p1['p1'])

    print("VERDICT:")
    if sp_mean > h_mean and sp_h2h > h_h2h:
        print("  Self-play model is stronger on both metrics.")
    elif h_mean > sp_mean and h_h2h > sp_h2h:
        print("  Heuristic model is stronger on both metrics.")
    else:
        better_solo = "Self-play" if sp_mean > h_mean else "Heuristic"
        better_h2h  = "Self-play" if sp_h2h  > h_h2h  else "Heuristic"
        print(f"  Mixed: {better_solo} scores higher solo, "
              f"{better_h2h} scores higher head-to-head.")
    print()



def main():
    parser = argparse.ArgumentParser(description="Compare heuristic vs self-play PPO checkpoints")
    parser.add_argument('--heuristic_ckpt', default='ppo_judgement_heuristic.pt')
    parser.add_argument('--selfplay_ckpt',  default='ppo_judgement_selfplay.pt')
    parser.add_argument('--games',    type=int, default=100,
                        help='Games per evaluation scenario')
    parser.add_argument('--out', default='comparison_figures',help='Output plot path')
    parser.add_argument('--no_plot',  action='store_true',
                        help='Skip plotting (print table only)')
    args = parser.parse_args()

    # Load models
    print(f"Loading heuristic checkpoint: {args.heuristic_ckpt}")
    h_model,  h_ckpt  = load_model(args.heuristic_ckpt)

    print(f"Loading self-play checkpoint: {args.selfplay_ckpt}")
    sp_model, sp_ckpt = load_model(args.selfplay_ckpt)

    max_cards  = h_ckpt['cfg']['max_cards']
    rand_score = h_ckpt.get('rand_score') or random_baseline(max_cards)

    # Scenario 1: Each model solo vs heuristic opponents
    print(f"\n[1/4] Heuristic model vs heuristic opponents ({args.games} games)...")
    h_solo = eval_vs_heuristics(h_model,  n_games=args.games, max_cards=max_cards,
                                 seed_offset=1000)

    print(f"[2/4] Self-play model vs heuristic opponents ({args.games} games)...")
    sp_solo = eval_vs_heuristics(sp_model, n_games=args.games, max_cards=max_cards,
                                  seed_offset=2000)

    # Scenario 2: Head-to-head 
    print(f"[3/4] H2H: Heuristic(P1) vs Self-play(P2+D) ({args.games} games)...")
    h_as_p1_results  = eval_head_to_head(
        h_model, sp_model, n_games=args.games, max_cards=max_cards, seed_offset=3000
    )

    print(f"[4/4] H2H: Self-play(P1) vs Heuristic(P2+D) ({args.games} games)...")
    sp_as_p1_results = eval_head_to_head(
        sp_model, h_model, n_games=args.games, max_cards=max_cards, seed_offset=4000
    )

    # Summary table 
    print_summary(h_solo, sp_solo, h_as_p1_results, sp_as_p1_results, rand_score)

    # Plot 
    if not args.no_plot:
        h_curve  = h_ckpt.get('all_mean_rews')
        sp_curve = sp_ckpt.get('all_mean_rews')
        plot_comparison(h_solo, sp_solo, h_as_p1_results, sp_as_p1_results,
        rand_score, h_curve, sp_curve, save_dir=args.out)
    print(f"Done. Figures saved to '{args.out}/'")


if __name__ == '__main__':
    main()
