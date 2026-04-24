"""
Judgement card game — full multi-round environment with pygame UI.

Game structure:
  - 17 rounds: round 1 deals 17 cards each, round 2 deals 16, ..., round 17 deals 1
  - Trump suit cycles per round: Spades → Diamonds → Clubs → Hearts → repeat
  - 3 players: agent (P1/idx 0), opponent (P2/idx 1), dealer (P3/idx 2)
  - Dealer constraint: sum of all bids cannot equal n_cards for that round
  - Must follow lead suit; highest trump wins; else highest lead-suit card wins

Episode = full 17-round game.
Reward at end of each round: (10 + bid) if tricks_won == bid, else 0.

Action space: Discrete(52)
  - Bidding : action = bid value (0 .. n_cards)
  - Playing : action = card index (0 .. 51)

Observation (129-dim float32):
  [0:52]    agent hand (binary)
  [52:104]  cards played in completed tricks this round (binary)
  [104:107] current trick card slots (card/52, -1 if empty)
  [107:110] bids per player normalized by n_cards (-1 if unplaced)
  [110:113] tricks won per player normalized by n_cards
  [113:117] trump suit one-hot
  [117:122] lead suit one-hot (index 4 = no lead set)
  [122]     phase (0=bidding, 1=playing)
  [123]     tricks_played / n_cards  (within-round progress)
  [124]     round_num / n_rounds     (across-round progress)
  [125]     n_cards / max_cards
  [126:129] cumulative scores per player, normalized by theoretical max
"""

import random

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

# ---------------------------------------------------------------------------
# Card constants
# ---------------------------------------------------------------------------

SUITS        = ['S',  'D',  'C',  'H']
SUIT_SYMBOLS = ['♠',  '♦',  '♣',  '♥']
SUIT_NAMES   = {0: 'Spades', 1: 'Diamonds', 2: 'Clubs', 3: 'Hearts'}
SUIT_COLORS  = {0: (20, 20, 20), 1: (200, 0, 0), 2: (20, 20, 20), 3: (200, 0, 0)}
RANKS        = ['2','3','4','5','6','7','8','9','10','J','Q','K','A']

TRUMP_CYCLE  = [0, 1, 2, 3]   # index into SUITS, cycles per round
MAX_CARDS    = 17              # cards dealt in round 1; rounds go 17 → 1
N_ROUNDS     = MAX_CARDS
OBS_DIM      = 129
# theoretical max score: sum of (10 + n) for n in 1..17
_MAX_SCORE   = sum(10 + n for n in range(1, MAX_CARDS + 1))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def card_display(idx: int) -> str:
    return f"{RANKS[idx % 13]}{SUIT_SYMBOLS[idx // 13]}"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class JudgementEnv(gym.Env):
    """Full 17-round Judgement. See module docstring for full spec."""

    metadata = {'render_modes': ['human']}

    def __init__(self, max_cards: int = MAX_CARDS, render_mode=None):
        super().__init__()
        self.max_cards  = max_cards
        self.n_rounds   = max_cards
        self.render_mode = render_mode
        self.n_players  = 3

        self.action_space      = spaces.Discrete(52)
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(OBS_DIM,), dtype=np.float32)

        self._screen = None
        self._clock  = None
        self._font_lg = self._font = self._font_sm = None

        # game state — populated by reset()
        self.round_num         = 0
        self.n_cards           = max_cards
        self.trump_suit        = 0
        self.cumulative_scores = [0, 0, 0]
        self.round_scores      = []        # list of per-round agent scores
        self.hands             = [[], [], []]
        self.bids              = [None, None, None]
        self.tricks_won        = [0, 0, 0]
        self.played_cards      = []
        self.current_trick     = []
        self.lead_suit         = None
        self.trick_leader      = 0
        self.tricks_played     = 0
        self.phase             = 'bidding'

    # ------------------------------------------------------------------
    # Core gym interface
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.round_num         = 0
        self.cumulative_scores = [0, 0, 0]
        self.round_scores      = []
        self._start_round()

        if self.render_mode == 'human':
            self._render_pygame()

        return self._get_obs(), {'action_mask': self._get_action_mask()}

    def step(self, action: int):
        action = int(action)
        mask = self._get_action_mask()
        if not mask[action]:
            raise ValueError(
                f"Illegal action {action} (phase={self.phase}). "
                f"Legal: {np.where(mask)[0].tolist()}"
            )

        if self.phase == 'bidding':
            self.bids[0] = action
            self._auto_bid(1)
            self._auto_bid(2)        # dealer — applies constraint
            self.phase = 'playing'
            self._advance_to_agent()
        else:
            self._do_play(0, action)
            self._advance_to_agent()

        # Round end?
        round_done   = self.tricks_played >= self.n_cards
        round_reward = 0.0
        if round_done:
            round_reward = self._player_reward(0)
            self.cumulative_scores[0] += round_reward
            self.cumulative_scores[1] += self._player_reward(1)
            self.cumulative_scores[2] += self._player_reward(2)
            self.round_scores.append(round_reward)

        # Game end?
        terminated = round_done and self.round_num >= self.n_rounds - 1

        if round_done and not terminated:
            self.round_num += 1
            self._start_round()
            self._advance_to_agent()

        obs  = self._get_obs()
        info = {
            'action_mask':        self._get_action_mask(),
            'bids':               list(self.bids),
            'tricks_won':         list(self.tricks_won),
            'round':              self.round_num + 1,
            'cumulative_scores':  list(self.cumulative_scores),
            'round_scores':       list(self.round_scores),
        }

        if self.render_mode == 'human':
            self._render_pygame()

        return obs, round_reward, terminated, False, info

    def render(self):
        if self.render_mode == 'human':
            self._render_pygame()

    def close(self):
        if self._screen is not None and _PYGAME_AVAILABLE:
            pygame.quit()
            self._screen = None

    # ------------------------------------------------------------------
    # Round management
    # ------------------------------------------------------------------

    def _start_round(self):
        self.n_cards     = self.max_cards - self.round_num   # 17, 16, …, 1
        self.trump_suit  = TRUMP_CYCLE[self.round_num % 4]
        deck             = list(range(52))
        random.shuffle(deck)
        self.hands       = [
            deck[0 * self.n_cards: 1 * self.n_cards],
            deck[1 * self.n_cards: 2 * self.n_cards],
            deck[2 * self.n_cards: 3 * self.n_cards],
        ]
        self.bids          = [None, None, None]
        self.tricks_won    = [0, 0, 0]
        self.played_cards  = []
        self.current_trick = []
        self.lead_suit     = None
        self.trick_leader  = 0    # P1 always leads the first trick of each round
        self.tricks_played = 0
        self.phase         = 'bidding'

    # ------------------------------------------------------------------
    # Observation and action mask
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        obs = np.full(OBS_DIM, -1.0, dtype=np.float32)

        for c in self.hands[0]:
            obs[c] = 1.0

        obs[52:104] = 0.0
        for c in self.played_cards:
            obs[52 + c] = 1.0

        for i in range(3):
            obs[104 + i] = (self.current_trick[i][1] / 52.0
                            if i < len(self.current_trick) else -1.0)

        for i in range(3):
            obs[107 + i] = (self.bids[i] / self.n_cards
                            if self.bids[i] is not None else -1.0)

        for i in range(3):
            obs[110 + i] = self.tricks_won[i] / self.n_cards

        obs[113:117] = 0.0
        obs[113 + self.trump_suit] = 1.0

        obs[117:122] = 0.0
        obs[117 + (self.lead_suit if self.lead_suit is not None else 4)] = 1.0

        obs[122] = 0.0 if self.phase == 'bidding' else 1.0
        obs[123] = self.tricks_played / self.n_cards
        obs[124] = self.round_num / self.n_rounds
        obs[125] = self.n_cards / self.max_cards

        for i in range(3):
            obs[126 + i] = self.cumulative_scores[i] / _MAX_SCORE

        return obs

    def _get_action_mask(self) -> np.ndarray:
        mask = np.zeros(52, dtype=bool)
        if self.phase == 'bidding':
            for b in range(self.n_cards + 1):
                mask[b] = True           # agent is never dealer, no constraint
        else:
            for c in self._legal_cards(0):
                mask[c] = True
        return mask

    # ------------------------------------------------------------------
    # Game logic
    # ------------------------------------------------------------------

    def _legal_cards(self, player: int) -> list[int]:
        hand   = self.hands[player]
        if self.lead_suit is None:
            return list(hand)
        suited = [c for c in hand if c // 13 == self.lead_suit]
        return suited if suited else list(hand)

    def _do_play(self, player: int, card: int):
        if not self.current_trick:
            self.lead_suit = card // 13
        self.hands[player].remove(card)
        self.current_trick.append((player, card))

    def _advance_to_agent(self):
        """Auto-play opponents and resolve tricks until the agent must act (or round ends)."""
        while self.tricks_played < self.n_cards:
            if len(self.current_trick) == self.n_players:
                winner = self._resolve_trick()
                self.tricks_won[winner] += 1
                self.played_cards.extend(c for _, c in self.current_trick)
                self.current_trick = []
                self.lead_suit     = None
                self.trick_leader  = winner
                self.tricks_played += 1
                continue

            play_order  = [(self.trick_leader + i) % self.n_players for i in range(self.n_players)]
            next_player = play_order[len(self.current_trick)]
            if next_player == 0:
                return   # agent's turn
            self._do_play(next_player, self._opponent_play(next_player))

    def _resolve_trick(self) -> int:
        lead = self.current_trick[0][1] // 13
        best_player, best_card = self.current_trick[0]
        for player, card in self.current_trick[1:]:
            if self._beats(card, best_card, lead):
                best_player, best_card = player, card
        return best_player

    def _beats(self, card: int, best: int, lead_suit: int) -> bool:
        cs, cr = card // 13, card % 13
        bs, br = best // 13, best % 13
        trump  = self.trump_suit
        if cs == trump and bs != trump:   return True
        if cs != trump and bs == trump:   return False
        if cs == bs:                      return cr > br
        if bs == lead_suit:               return False
        return cs == lead_suit

    def _player_reward(self, player: int) -> float:
        bid, won = self.bids[player], self.tricks_won[player]
        if bid is None:
            return 0.0
        return float(10 + bid) if won == bid else 0.0

    # ------------------------------------------------------------------
    # Opponent heuristics
    # ------------------------------------------------------------------

    def _auto_bid(self, player: int):
        hand = self.hands[player]
        bid  = sum(
            1 for c in hand
            if c % 13 == 12                                    # Ace
            or (c // 13 == self.trump_suit and c % 13 >= 9)   # J/Q/K/A of trump
        )
        bid = min(bid, self.n_cards)

        if player == 2:   # dealer constraint
            others = sum(b for b in self.bids if b is not None)
            if others + bid == self.n_cards:
                for candidate in [bid + 1, bid - 1] + list(range(self.n_cards + 1)):
                    candidate = max(0, min(candidate, self.n_cards))
                    if others + candidate != self.n_cards:
                        bid = candidate
                        break

        self.bids[player] = bid

    def _opponent_play(self, player: int) -> int:
        legal = self._legal_cards(player)
        if not self.current_trick:
            trumps = [c for c in legal if c // 13 == self.trump_suit]
            pool   = trumps if trumps else legal
            return max(pool, key=lambda c: c % 13)

        lead         = self.lead_suit
        current_best = max((c for _, c in self.current_trick),
                           key=lambda c: (c // 13 == self.trump_suit, c % 13))
        winning = [c for c in legal if self._beats(c, current_best, lead)]
        if winning:
            return min(winning, key=lambda c: (c // 13 == self.trump_suit, c % 13))
        return min(legal, key=lambda c: c % 13)

    # ------------------------------------------------------------------
    # Pygame rendering
    # ------------------------------------------------------------------

    # Layout constants
    _W,   _H   = 1150, 780
    _CW,  _CH  = 52,   76      # card width / height
    _GAP        = 4

    _COL_BG     = (34, 100, 34)
    _COL_HDR    = (15,  50, 15)
    _COL_PANEL  = (22,  75, 22)
    _COL_WHITE  = (255, 255, 255)
    _COL_TEXT   = (235, 255, 235)
    _COL_BACK   = ( 60,  90, 180)
    _COL_TRUMP  = (255, 210,   0)   # gold border for trump cards
    _COL_LEGAL  = ( 50, 230, 120)   # green border for legal plays
    _COL_SLOT   = ( 50, 130,  50)   # empty trick slot

    def _init_pygame(self):
        if not _PYGAME_AVAILABLE:
            raise RuntimeError("pygame is not installed. Run: pip install pygame")
        if self._screen is None:
            pygame.init()
            self._screen  = pygame.display.set_mode((self._W, self._H))
            pygame.display.set_caption("Judgement — RL Environment")
            self._clock   = pygame.time.Clock()
            self._font_lg = pygame.font.SysFont('Arial', 20, bold=True)
            self._font    = pygame.font.SysFont('Arial', 16)
            self._font_sm = pygame.font.SysFont('Arial', 13)

    def _draw_card(self, surf, idx: int, x: int, y: int,
                   face_down: bool = False, highlight=None):
        rect = pygame.Rect(x, y, self._CW, self._CH)
        pygame.draw.rect(surf, self._COL_WHITE, rect, border_radius=4)

        if face_down:
            inner = rect.inflate(-6, -6)
            pygame.draw.rect(surf, self._COL_BACK, inner, border_radius=3)
        else:
            suit, rank = idx // 13, idx % 13
            col = SUIT_COLORS[suit]
            # rank top-left
            r_surf = self._font_sm.render(RANKS[rank], True, col)
            surf.blit(r_surf, (x + 3, y + 2))
            # suit symbol center
            s_surf = self._font.render(SUIT_SYMBOLS[suit], True, col)
            s_rect = s_surf.get_rect(center=(x + self._CW // 2, y + self._CH // 2))
            surf.blit(s_surf, s_rect)
            # trump gold border
            border_col = self._COL_TRUMP if suit == self.trump_suit else (160, 160, 160)
            border_w   = 2 if suit == self.trump_suit else 1
            pygame.draw.rect(surf, border_col, rect, border_w, border_radius=4)

        if highlight is not None:
            pygame.draw.rect(surf, highlight, rect, 3, border_radius=4)

    def _text(self, surf, text: str, x: int, y: int, font=None, color=None):
        f = font  or self._font
        c = color or self._COL_TEXT
        surf.blit(f.render(text, True, c), (x, y))

    def _render_pygame(self):
        self._init_pygame()

        # pump events so window stays alive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._screen = None
                return

        surf = self._screen
        surf.fill(self._COL_BG)

        W, CW, CH, GAP = self._W, self._CW, self._CH, self._GAP
        trump_sym  = SUIT_SYMBOLS[self.trump_suit]
        trump_name = SUIT_NAMES[self.trump_suit]

        # ── Header ────────────────────────────────────────────────────
        pygame.draw.rect(surf, self._COL_HDR, (0, 0, W, 48))
        self._text(surf,
            f"Round {self.round_num + 1} / {self.n_rounds}   "
            f"Trump: {trump_sym} {trump_name}   "
            f"Cards this round: {self.n_cards}   "
            f"Phase: {'BIDDING' if self.phase == 'bidding' else 'PLAYING'}",
            10, 8, self._font_lg)

        # ── Score bar ─────────────────────────────────────────────────
        pygame.draw.rect(surf, self._COL_PANEL, (0, 48, W, 32))
        score_parts = [
            f"You: {self.cumulative_scores[0]}",
            f"P2: {self.cumulative_scores[1]}",
            f"Dealer: {self.cumulative_scores[2]}",
            f"Max possible: {_MAX_SCORE}",
        ]
        if self.round_scores:
            score_parts.append(f"Last round: {self.round_scores[-1]:.0f}")
        self._text(surf, "Scores —  " + "   |   ".join(score_parts), 10, 54)

        y = 90   # running y cursor

        # ── Opponents (side by side) ───────────────────────────────────
        half = W // 2 - 20

        def draw_opponent(player: int, ox: int, label: str):
            bid_str = str(self.bids[player]) if self.bids[player] is not None else '?'
            self._text(surf,
                f"{label}   bid: {bid_str}   won: {self.tricks_won[player]}",
                ox, y)
            n = len(self.hands[player])
            if n > 0:
                spacing = min(CW + GAP, half // n)
                for i in range(n):
                    self._draw_card(surf, 0, ox + i * spacing, y + 22, face_down=True)

        draw_opponent(1, 10,       "P2")
        draw_opponent(2, W // 2 + 10, "Dealer")

        y += 22 + CH + 18

        # ── Current trick ─────────────────────────────────────────────
        pygame.draw.rect(surf, self._COL_PANEL, (0, y - 4, W, CH + 50))
        self._text(surf, "Current Trick", W // 2 - 60, y)
        y += 22

        player_labels = ['You', 'P2', 'Dealer']
        trick_total_w = 3 * (CW + 40)
        tx = (W - trick_total_w) // 2

        for i in range(3):
            slot_x = tx + i * (CW + 40)
            if i < len(self.current_trick):
                player, card = self.current_trick[i]
                self._draw_card(surf, card, slot_x, y)
                self._text(surf, player_labels[player], slot_x + 4, y + CH + 2,
                           self._font_sm)
            else:
                pygame.draw.rect(surf, self._COL_SLOT,
                                 (slot_x, y, CW, CH), 2, border_radius=4)

        # trick winner label (if trick just completed)
        if self.tricks_played > 0 and not self.current_trick:
            last_winner = self.trick_leader
            self._text(surf,
                f"Trick won by: {player_labels[last_winner]}",
                tx + trick_total_w + 10, y + CH // 2 - 8,
                color=(255, 255, 100))

        y += CH + 44

        # ── Agent hand ────────────────────────────────────────────────
        bid_str = str(self.bids[0]) if self.bids[0] is not None else '?'
        self._text(surf,
            f"Your Hand   bid: {bid_str}   won: {self.tricks_won[0]}   "
            f"({len(self.hands[0])} cards)",
            10, y, self._font_lg)
        y += 26

        mask        = self._get_action_mask()
        hand_sorted = sorted(self.hands[0])
        n_agent     = len(hand_sorted)

        if n_agent > 0:
            total_w = n_agent * (CW + GAP) - GAP
            start_x = max(10, (W - total_w) // 2)
            for i, card in enumerate(hand_sorted):
                cx        = start_x + i * (CW + GAP)
                is_legal  = (self.phase == 'playing') and mask[card]
                highlight = self._COL_LEGAL if is_legal else None
                self._draw_card(surf, card, cx, y, highlight=highlight)

        # ── Status bar ────────────────────────────────────────────────
        pygame.draw.rect(surf, self._COL_HDR, (0, self._H - 36, W, 36))
        if self.phase == 'bidding':
            status = f"BIDDING — choose a bid from 0 to {self.n_cards}"
        else:
            status = "PLAYING — green-bordered cards are legal plays"
        self._text(surf, status, 10, self._H - 24)

        pygame.display.flip()
        self._clock.tick(30)


# ---------------------------------------------------------------------------
# Smoke test / demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time

    env = JudgementEnv(max_cards=17, render_mode='human')
    obs, info = env.reset(seed=0)

    print("=== Judgement — 17-round game ===")

    terminated = False
    total_reward = 0.0
    step_num = 0

    while not terminated:
        mask  = info['action_mask']
        valid = np.where(mask)[0]

        if env.phase == 'bidding':
            action = int(valid[len(valid) // 2])   # middle bid
        else:
            action = int(np.random.choice(valid))   # random legal card

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if reward > 0:
            r = info['round'] - 1  # round that just finished
            print(f"  Round {r:2d}: bid={info['bids'][0]}  won={info['tricks_won'][0]}  "
                  f"reward={reward:.0f}  cumulative={info['cumulative_scores'][0]:.0f}")

        time.sleep(0.3)   # slow down so UI is watchable
        step_num += 1

    print(f"\n=== Game over after {step_num} steps ===")
    print(f"Final scores — You: {info['cumulative_scores'][0]}  "
          f"P2: {info['cumulative_scores'][1]}  "
          f"Dealer: {info['cumulative_scores'][2]}")
    print(f"Your total reward: {total_reward:.0f} / {_MAX_SCORE} possible")

    input("\nPress Enter to close...")
    env.close()
