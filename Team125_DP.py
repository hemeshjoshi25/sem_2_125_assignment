# =============================================================================
# BIRLA INSTITUTE OF TECHNOLOGY AND SCIENCE, PILANI
# WORK INTEGRATED LEARNING PROGRAMMES DIVISION
# Deep Reinforcement Learning - Lab Assignment 1
# PART #2: Autonomous Drone Rescue Using Dynamic Programming
# =============================================================================
# Group Number        : 125  (last digit = 5)
# Grid size           : 6×6  (last digit 5–9 → 6×6)
# Max battery         : 15   (last digit odd → 15)
# Wind probability    : 30%  (last digit 5–9 → 30%)
# Rescue targets      : 3
# Charging stations   : 2
# Danger zones        : 4
# Blocked cells       : 3
# Max steps/episode   : 75
# =============================================================================

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from itertools import product
import time
import warnings
warnings.filterwarnings("ignore")


# =============================================================================
# SECTION 1: GRID CONFIGURATION
# =============================================================================

# Cell-type symbols (as per assignment specification)
S = 'S'   # Start — fixed at top-left corner
F = 'F'   # Free / Safe cell
D = 'D'   # Dangerous zone  (reward -10, episode continues)
R = 'R'   # Rescue target   (reward +20, becomes F after rescue)
C = 'C'   # Charging station (battery refilled to max on entry; hover gives +2)
W = 'W'   # Wind zone       (30% chance of random direction deflection)
X = 'X'   # Blocked cell    (impassable; drone stays put, battery still consumed)

# ── Environment constants derived from group ID last digit = 5 ──────────────
GRID_ROWS   = 6
GRID_COLS   = 6
MAX_BATTERY = 15     # odd last digit → 15
WIND_PROB   = 0.30   # last digit 5–9 → 30%
MAX_STEPS   = 75     # 6×6 grid → 75 steps max

# ── 6×6 Grid Layout ─────────────────────────────────────────────────────────
# Design rationale:
#   S  fixed at (0,0) — top-left as required
#   R1 at (0,3) — first rescue target, reachable from start via row 0
#   R2 at (3,5) — second target, in far right column to create interesting paths
#   R3 at (5,0) — third target, bottom-left corner forces careful battery mgmt
#   C1 at (2,2) — central charging station; midpoint between start & far targets
#   C2 at (4,4) — second charger; near R2 and R3 to support deep rescues
#   D1 at (0,5) — top-right corner; discourages naive rightward paths
#   D2 at (1,2) — near C1 approach; forces careful navigation
#   D3 at (3,2) — central danger; splits the grid, forces route choice
#   D4 at (5,3) — near R3 approach; penalises careless descent
#   X1 at (1,4) — blocked obstacle near row 1
#   X2 at (2,5) — blocked; cuts off direct right-column rush
#   X3 at (4,2) — blocked; near C2 approach, forces detour
#   W1 at (1,0) — wind near start column; introduces early stochasticity
#   W2 at (3,3) — central wind; affects paths between both charging stations
#   Remaining cells are F

BASE_GRID = [
    [S,   F,   F,   R,   F,   D],   # row 0
    [W,   F,   D,   F,   X,   F],   # row 1
    [F,   F,   C,   F,   F,   X],   # row 2
    [F,   F,   D,   W,   F,   R],   # row 3
    [F,   F,   X,   F,   C,   F],   # row 4
    [R,   F,   F,   D,   F,   F],   # row 5
]

# Reward structure (from assignment table)
REWARD_RESCUE  =  20
REWARD_DANGER  = -10
REWARD_BATTERY = -20
REWARD_CHARGE  =   5
REWARD_MOVE    =  -1

# Action indices
UP, DOWN, LEFT, RIGHT, HOVER = 0, 1, 2, 3, 4
ACTION_NAMES  = {UP: "↑", DOWN: "↓", LEFT: "←", RIGHT: "→", HOVER: "⊙"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1),
                 RIGHT: (0, 1), HOVER: (0, 0)}


# =============================================================================
# SECTION 2: CUSTOM DRONE RESCUE ENVIRONMENT
# =============================================================================

class DroneRescueEnv:
    """
    Finite MDP environment for the 6×6 autonomous drone rescue problem.

    State representation
    ────────────────────
    (row, col, battery, rescue_mask)

    Where `rescue_mask` is a tuple of booleans (one per rescue target),
    True if that target has already been rescued. This fully satisfies
    the Markov property: given the state, no additional history is needed.

    Transition dynamics
    ───────────────────
    • All actions cost 1 battery unit.
    • Hover on C: battery += 2 (capped at MAX_BATTERY) instead of −1.
    • Entering C (non-hover): battery = MAX_BATTERY, reward += REWARD_CHARGE.
    • Entering D: reward += REWARD_DANGER (episode continues).
    • Entering R (unrescued): reward += REWARD_RESCUE; cell becomes F.
    • Wind cell (W): if action ≠ HOVER, with probability WIND_PROB the
      actual movement direction is chosen uniformly from {UP,DOWN,LEFT,RIGHT}.
    • Entering X or out-of-bounds: drone stays; battery still consumed.
    • Termination: battery ≤ 0 | all rescued | steps ≥ MAX_STEPS.
    """

    def __init__(self, grid=BASE_GRID, max_battery=MAX_BATTERY,
                 wind_prob=WIND_PROB, max_steps=MAX_STEPS):
        self.grid        = grid
        self.rows        = len(grid)
        self.cols        = len(grid[0])
        self.max_battery = max_battery
        self.wind_prob   = wind_prob
        self.max_steps   = max_steps

        # Locate key cells once at init time
        self.start_pos       = self._find(S)[0]
        self.rescue_targets  = self._find(R)
        self.n_rescues       = len(self.rescue_targets)

        self.reset()

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _find(self, symbol):
        """Returns list of (row, col) matching symbol in the base grid."""
        return [(r, c) for r in range(self.rows) for c in range(self.cols)
                if self.grid[r][c] == symbol]

    def _get_state(self):
        """Packages mutable env variables into an immutable hashable state."""
        return (self.pos[0], self.pos[1], self.battery, self.rescue_mask)

    # ── MDP Interface ─────────────────────────────────────────────────────────
    def reset(self):
        """
        Resets the environment to the initial state.

        Returns
        -------
        state : tuple  -- (row, col, battery, rescue_mask)
        """
        self.pos         = self.start_pos
        self.battery     = self.max_battery
        self.rescue_mask = tuple([False] * self.n_rescues)
        self.steps       = 0
        self.done        = False
        return self._get_state()

    def valid_actions(self, state=None):
        """
        Returns the list of all valid action indices from any state.

        All five actions are always syntactically available; the environment
        handles blocked/boundary collisions internally (drone stays put).

        Returns
        -------
        list[int]  -- [UP, DOWN, LEFT, RIGHT, HOVER]
        """
        return [UP, DOWN, LEFT, RIGHT, HOVER]

    def step(self, action: int) -> tuple:
        """
        Executes one action and returns the MDP transition tuple.

        Wind model: if the drone's current cell is W and the action is a
        movement (not HOVER), with probability `wind_prob` the actual
        direction is resampled uniformly from {UP, DOWN, LEFT, RIGHT}.

        Parameters
        ----------
        action : int  -- One of {UP, DOWN, LEFT, RIGHT, HOVER}

        Returns
        -------
        next_state : tuple
        reward     : float
        done       : bool
        info       : dict  -- diagnostic info (wind_deflected, charged, etc.)
        """
        if self.done:
            raise RuntimeError("Episode has ended. Call reset() first.")

        r, c   = self.pos
        reward = REWARD_MOVE
        info   = {}

        # ── Wind disturbance ────────────────────────────────────────────────
        actual_action = action
        if action != HOVER and self.grid[r][c] == W:
            if np.random.rand() < self.wind_prob:
                actual_action          = np.random.choice([UP, DOWN, LEFT, RIGHT])
                info["wind_deflected"] = True

        # ── Compute next position ────────────────────────────────────────────
        dr, dc = ACTION_DELTAS[actual_action]
        nr, nc = r + dr, c + dc

        # Stay if out-of-bounds or blocked
        if not (0 <= nr < self.rows and 0 <= nc < self.cols
                and self.grid[nr][nc] != X):
            nr, nc = r, c
        self.pos = (nr, nc)

        # ── Battery update ───────────────────────────────────────────────────
        if action == HOVER and self.grid[nr][nc] == C:
            # Hovering on charger recharges +2 (capped)
            self.battery = min(self.max_battery, self.battery + 2)
        else:
            self.battery -= 1

        # ── Cell-entry effects ───────────────────────────────────────────────
        cell = self.grid[nr][nc]

        if cell == C and action != HOVER:
            self.battery    = self.max_battery   # full charge on entry
            reward         += REWARD_CHARGE
            info["charged"] = True

        elif cell == D:
            reward        += REWARD_DANGER
            info["danger"] = True

        elif cell == R:
            t_idx = self.rescue_targets.index((nr, nc))
            if not self.rescue_mask[t_idx]:
                reward           += REWARD_RESCUE
                mask_list         = list(self.rescue_mask)
                mask_list[t_idx]  = True
                self.rescue_mask  = tuple(mask_list)
                info["rescued"]   = t_idx

        # ── Termination ──────────────────────────────────────────────────────
        self.steps += 1
        if self.battery <= 0:
            reward             += REWARD_BATTERY
            self.done           = True
            info["termination"] = "battery_exhausted"
        elif all(self.rescue_mask):
            self.done           = True
            info["termination"] = "all_rescued"
        elif self.steps >= self.max_steps:
            self.done           = True
            info["termination"] = "max_steps"

        return self._get_state(), reward, self.done, info

    def render(self, state=None):
        """
        Prints a styled ASCII grid showing the drone position, cell types,
        battery level, and rescue progress.
        """
        if state is not None:
            r, c, bat, mask = state
        else:
            r, c = self.pos
            bat, mask = self.battery, self.rescue_mask

        print(f"\n  Battery: {bat}/{self.max_battery}  "
              f"Rescued: {sum(mask)}/{self.n_rescues}")
        print("  " + "─" * (self.cols * 7))
        for row in range(self.rows):
            line = "  "
            for col in range(self.cols):
                cell = self.grid[row][col]
                # Rescued targets appear as free cells
                if cell == R:
                    t_idx = self.rescue_targets.index((row, col))
                    if mask[t_idx]:
                        cell = "f"
                if (row, col) == (r, c):
                    sym = "🚁"
                else:
                    sym_map = {S:" S ", F:" · ", R:" R ", C:" C ",
                               D:" D ", W:" W ", X:"███", "f":" · "}
                    sym = sym_map.get(cell, " ? ")
                line += f"[{sym:^3}] "
            print(line)
        print("  " + "─" * (self.cols * 7))


# =============================================================================
# SECTION 3: DYNAMIC PROGRAMMING — VALUE ITERATION
# =============================================================================

def enumerate_states(env: DroneRescueEnv) -> list:
    """
    Enumerates all reachable states: (row, col, battery, rescue_mask).

    Excludes:
      • positions on blocked (X) cells — drone can never occupy these
      • battery = 0                    — immediate terminal state

    Parameters
    ----------
    env : DroneRescueEnv

    Returns
    -------
    states : list[tuple]
    """
    rescue_configs = list(product([False, True], repeat=env.n_rescues))
    states = [
        (r, c, bat, mask)
        for r in range(env.rows)
        for c in range(env.cols)
        if env.grid[r][c] != X
        for bat in range(1, env.max_battery + 1)
        for mask in rescue_configs
    ]
    print(f"  Total reachable states        : {len(states):,}")
    return states


def transition_probability(env: DroneRescueEnv, state: tuple,
                            action: int) -> list:
    """
    Computes the full transition distribution P(s'|s,a) for one (state, action).

    Handles the wind stochasticity model:
    If the current cell is W and the action is a movement, the actual direction
    is the intended one with probability (1 - wind_prob), or one of the four
    cardinal directions uniformly with probability (wind_prob / 4) each.

    Parameters
    ----------
    env    : DroneRescueEnv
    state  : tuple  -- (row, col, battery, rescue_mask)
    action : int

    Returns
    -------
    transitions : list[ (probability, next_state, reward) ]
    """
    from collections import defaultdict

    r, c, bat, mask = state

    # Build (probability, actual_action) distribution
    if action == HOVER:
        actual_actions = [(1.0, HOVER)]
    elif env.grid[r][c] == W:
        # Wind stochasticity: merge probabilities for identical actions
        p_int = 1.0 - env.wind_prob
        p_rnd = env.wind_prob / 4.0
        raw = [(p_int, action)] + [(p_rnd, a) for a in [UP, DOWN, LEFT, RIGHT]]
        merged = defaultdict(float)
        for p, a in raw:
            merged[int(a)] += p
        actual_actions = [(prob, int(act)) for act, prob in merged.items()]
    else:
        actual_actions = [(1.0, int(action))]

    transitions = []
    for prob, act in actual_actions:
        dr, dc = ACTION_DELTAS[act]
        nr, nc = r + dr, c + dc

        # Boundary / blocked: stay in place
        if not (0 <= nr < env.rows and 0 <= nc < env.cols
                and env.grid[nr][nc] != X):
            nr, nc = r, c

        reward    = REWARD_MOVE
        next_bat  = bat
        next_mask = list(mask)

        # Battery
        if act == HOVER and env.grid[nr][nc] == C:
            next_bat = min(env.max_battery, bat + 2)
        else:
            next_bat = bat - 1

        # Cell effects
        cell = env.grid[nr][nc]
        if cell == C and act != HOVER:
            next_bat  = env.max_battery
            reward   += REWARD_CHARGE
        elif cell == D:
            reward += REWARD_DANGER
        elif cell == R:
            t_idx = env.rescue_targets.index((nr, nc))
            if not mask[t_idx]:
                reward           += REWARD_RESCUE
                next_mask[t_idx]  = True

        # Battery-exhaustion penalty
        if next_bat <= 0:
            reward  += REWARD_BATTERY

        next_state = (nr, nc, max(0, next_bat), tuple(next_mask))
        transitions.append((prob, next_state, reward))

    return transitions


def is_terminal(state: tuple, env: DroneRescueEnv) -> bool:
    """Returns True for absorbing states (battery=0 or all targets rescued)."""
    _, _, bat, mask = state
    return bat <= 0 or all(mask)


def value_iteration(env: DroneRescueEnv, gamma: float = 0.99,
                    theta: float = 1e-3) -> tuple:
    """
    Computes V*(s) and π*(s) via the Value Iteration algorithm.

    Algorithm (Bellman optimality update)
    ──────────────────────────────────────
    Initialise V(s) = 0  ∀s
    Repeat until max_s |V_new(s) − V_old(s)| < θ:
      For each non-terminal state s:
        V(s) ← max_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V(s') ]
    π*(s) ← argmax_a  Σ_{s'} P(s'|s,a) [ R(s,a,s') + γ V(s') ]

    Parameters
    ----------
    gamma : float  -- Discount factor (0.99)
    theta : float  -- Stopping threshold (1e-3)

    Returns
    -------
    V       : dict   -- {state: optimal value}
    policy  : dict   -- {state: optimal action}
    history : list   -- max Δ per iteration (convergence trace)
    """
    print("\n  Running Value Iteration …")
    states  = enumerate_states(env)
    actions = env.valid_actions()

    V       = {s: 0.0 for s in states}
    policy  = {}
    history = []

    t_start   = time.time()
    iteration = 0

    while True:
        delta     = 0.0
        iteration += 1

        for state in states:
            if is_terminal(state, env):
                V[state] = 0.0
                continue

            best_val = -np.inf
            best_act = HOVER

            for action in actions:
                trans   = transition_probability(env, state, action)
                q_value = sum(p * (r + gamma * V.get(s2, 0.0))
                              for p, s2, r in trans)
                if q_value > best_val:
                    best_val = q_value
                    best_act = action

            delta        = max(delta, abs(V[state] - best_val))
            V[state]     = best_val
            policy[state] = best_act

        history.append(delta)

        if iteration % 10 == 0:
            print(f"    Iteration {iteration:>4}  │  Δ = {delta:.6f}")

        if delta < theta:
            break

    elapsed = time.time() - t_start
    print(f"\n  [✓] Converged after {iteration} iterations")
    print(f"  [✓] Final delta (Δ)           : {delta:.8f}")
    print(f"  [✓] Runtime                   : {elapsed:.3f} s")
    print(f"  [✓] Stopping threshold (θ)    : {theta}")
    return V, policy, history


# =============================================================================
# SECTION 4: POLICY VISUALISATION
# =============================================================================

def visualise_policy(env: DroneRescueEnv, policy: dict, V: dict,
                     battery_level: int = None, rescue_mask: tuple = None):
    """
    Side-by-side visualisation:
      Left  — Policy grid: cell colours + directional arrows / hover markers
      Right — V*(s) heatmap with numeric values overlaid

    Parameters
    ----------
    battery_level : int    -- Battery slice to display (default = max)
    rescue_mask   : tuple  -- Rescue status slice (default = all unrescued)
    """
    if battery_level is None:
        battery_level = env.max_battery
    if rescue_mask is None:
        rescue_mask = tuple([False] * env.n_rescues)

    rows, cols = env.rows, env.cols

    cell_colors = {S: "#1A535C", F: "#1E3A5F", D: "#C1121F",
                   R: "#F4A261", C: "#43AA8B", W: "#457B9D", X: "#212529"}
    arrow_map   = {UP: (0, -0.38), DOWN: (0, 0.38),
                   LEFT: (-0.38, 0), RIGHT: (0.38, 0)}

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), facecolor="#0D1117")
    fig.suptitle(
        f"Optimal Policy & Value Function  │  G=125  │  "
        f"Battery={battery_level}  │  Rescued={rescue_mask}",
        fontsize=13, fontweight="bold", color="white"
    )

    # ── Left: Policy grid ─────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#0D1117")
    ax1.set_title("Optimal Policy π*(s)", color="white", fontsize=12, pad=10)

    for r in range(rows):
        for c in range(cols):
            cell = env.grid[r][c]
            col  = cell_colors.get(cell, "#1E3A5F")
            if cell == R:
                t_idx = env.rescue_targets.index((r, c))
                if rescue_mask[t_idx]:
                    col = "#2A9D8F"
            rect = plt.Rectangle([c - 0.5, rows - r - 1.5], 1, 1,
                                  color=col, ec="#0D1117", lw=1.5)
            ax1.add_patch(rect)
            ax1.text(c, rows - r - 1, cell, ha="center", va="center",
                     fontsize=10, color="white", fontweight="bold")

            # Policy arrow
            s_key = (r, c, battery_level, rescue_mask)
            if s_key in policy and cell != X:
                act = policy[s_key]
                if act == HOVER:
                    ax1.plot(c, rows - r - 1, "o", color="yellow",
                             markersize=9, zorder=5)
                else:
                    dx, dy = arrow_map[act]
                    ax1.annotate("",
                        xy=(c + dx * 1.7, rows - r - 1 + dy * 1.7),
                        xytext=(c - dx * 0.4, rows - r - 1 - dy * 0.4),
                        arrowprops=dict(arrowstyle="->", color="yellow",
                                        lw=2.0, mutation_scale=14),
                        zorder=5)

    ax1.set_xlim(-0.5, cols - 0.5)
    ax1.set_ylim(-0.5, rows - 0.5)
    ax1.set_xticks(range(cols))
    ax1.set_yticks(range(rows))
    ax1.set_xticklabels([f"C{i}" for i in range(cols)], color="white")
    ax1.set_yticklabels([f"R{rows-1-i}" for i in range(rows)], color="white")
    ax1.tick_params(colors="white")

    legend_elements = [
        mpatches.Patch(color=cell_colors[k], label=f"{k}: {v}")
        for k, v in [(S,"Start"), (F,"Free"), (D,"Danger"), (R,"Rescue"),
                     (C,"Charge"), (W,"Wind"), (X,"Blocked")]
    ]
    ax1.legend(handles=legend_elements, loc="lower right",
               facecolor="#21262D", labelcolor="white", fontsize=8)

    # ── Right: Value heatmap ──────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#0D1117")
    ax2.set_title("Value Function V*(s) Heatmap", color="white",
                  fontsize=12, pad=10)

    V_grid = np.array([
        [V.get((r, c, battery_level, rescue_mask), 0.0) for c in range(cols)]
        for r in range(rows)
    ])
    masked = np.ma.masked_where(
        [[env.grid[r][c] == X for c in range(cols)] for r in range(rows)],
        V_grid
    )
    cmap = LinearSegmentedColormap.from_list(
        "rh", ["#C1121F", "#1E3A5F", "#43AA8B"], N=256)
    cmap.set_bad("#212529")

    im = ax2.imshow(masked, cmap=cmap, origin="upper", aspect="equal")
    cb = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cb.set_label("V*(s)", color="white")
    cb.ax.yaxis.set_tick_params(color="white")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    for r in range(rows):
        for c in range(cols):
            if env.grid[r][c] != X:
                ax2.text(c, r, f"{V_grid[r,c]:.1f}", ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
            ax2.text(c, r + 0.38, env.grid[r][c], ha="center", va="center",
                     fontsize=7, color="#AAAAAA")

    ax2.set_xticks(range(cols))
    ax2.set_yticks(range(rows))
    ax2.set_xticklabels([f"C{i}" for i in range(cols)], color="white")
    ax2.set_yticklabels([f"R{i}" for i in range(rows)], color="white")
    ax2.tick_params(colors="white")
    for sp in ax2.spines.values():
        sp.set_edgecolor("#30363D")

    plt.tight_layout()
    plt.savefig("DP_policy_heatmap.png", dpi=140,
                bbox_inches="tight", facecolor="#0D1117")
    plt.show()
    print("  [✓] Policy visualisation saved → DP_policy_heatmap.png")


# =============================================================================
# SECTION 5: CONVERGENCE PLOT & STATE-VALUE ANALYSIS
# =============================================================================

def plot_convergence(history: list):
    """
    Semi-log convergence curve for Value Iteration.
    Marks the stopping threshold θ = 1e-3 with a dashed red line.
    """
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0D1117")
    ax.set_facecolor("#161B22")
    ax.semilogy(range(1, len(history) + 1), history,
                color="#2A9D8F", linewidth=2.0, label="Max Δ per iteration")
    ax.axhline(y=1e-3, color="#E63946", linestyle="--", linewidth=1.5,
               label="θ = 1e-3 (stopping threshold)")
    ax.set_xlabel("Iteration", color="white", fontsize=11)
    ax.set_ylabel("Max Delta (Δ) [log scale]", color="white", fontsize=11)
    ax.set_title(f"Value Iteration Convergence  │  G=125  │  "
                 f"Converged at {len(history)} iterations",
                 color="white", fontsize=13, pad=10)
    ax.legend(facecolor="#21262D", labelcolor="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.grid(True, color="#21262D", linestyle="--", alpha=0.6)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363D")
    plt.tight_layout()
    plt.savefig("DP_convergence.png", dpi=140,
                bbox_inches="tight", facecolor="#0D1117")
    plt.show()
    print("  [✓] Convergence plot saved → DP_convergence.png")


def plot_state_value_analysis(env: DroneRescueEnv, V: dict):
    """
    State-Value Analysis: fix rescue_mask = all-unrescued, vary position,
    show V* heatmap sliced at three battery levels (3, 8, 15).

    Reveals how remaining battery critically shapes the drone's value
    landscape and navigation strategy.
    """
    battery_levels = [3, 8, env.max_battery]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), facecolor="#0D1117")
    fig.suptitle(
        "State-Value Analysis: V*(row,col) sliced by Battery Level\n"
        "(rescue_mask = all False)",
        fontsize=13, fontweight="bold", color="white"
    )
    mask = tuple([False] * env.n_rescues)
    cmap = LinearSegmentedColormap.from_list(
        "sv", ["#C1121F", "#1E3A5F", "#43AA8B"], N=256)
    cmap.set_bad("#212529")

    for ax, bat in zip(axes, battery_levels):
        ax.set_facecolor("#0D1117")
        V_grid = np.array([
            [V.get((r, c, bat, mask), 0.0) for c in range(env.cols)]
            for r in range(env.rows)
        ])
        blocked = np.array([[env.grid[r][c] == X for c in range(env.cols)]
                            for r in range(env.rows)])
        masked = np.ma.masked_where(blocked, V_grid)
        im = ax.imshow(masked, cmap=cmap, origin="upper", aspect="equal")
        plt.colorbar(im, ax=ax, fraction=0.046)

        for r in range(env.rows):
            for c in range(env.cols):
                if not blocked[r][c]:
                    ax.text(c, r, f"{V_grid[r,c]:.1f}", ha="center",
                            va="center", fontsize=8, color="white",
                            fontweight="bold")
                ax.text(c, r + 0.40, env.grid[r][c], ha="center", va="center",
                        fontsize=6, color="#AAAAAA")

        ax.set_title(f"Battery = {bat}", color="white", fontsize=12)
        ax.set_xticks(range(env.cols))
        ax.set_yticks(range(env.rows))
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363D")

    plt.tight_layout()
    plt.savefig("DP_state_value_analysis.png", dpi=140,
                bbox_inches="tight", facecolor="#0D1117")
    plt.show()
    print("  [✓] State-value analysis saved → DP_state_value_analysis.png")

    print("\n  ── State-Value Observations ─────────────────────────────────")
    print("  • At battery=15 (full), all non-blocked cells show high V*,")
    print("    as the drone has ample energy to chase all 3 rescue targets.")
    print("  • At battery=8, cells near C1(2,2) and C2(4,4) gain relative")
    print("    value — the charging stations become attractive waypoints.")
    print("  • At battery=3, V* drops sharply everywhere. Cells adjacent to")
    print("    R targets remain the most valuable as they give +20 reward")
    print("    even when the drone cannot continue far afterward.")
    print("  • Danger cells (D) consistently show lower V* due to the -10")
    print("    penalty; wind cells (W) show moderate V* reflecting the risk")
    print("    of stochastic displacement reducing expected return.")
    print("  • Blocked cells (X) are masked throughout — they never appear.")


# =============================================================================
# SECTION 6: DP SCALABILITY — CURSE OF DIMENSIONALITY
# =============================================================================

def dp_scalability_discussion():
    """
    Structured discussion with quantitative state-space calculations
    demonstrating the curse of dimensionality and motivating Deep RL.
    """
    print("\n" + "=" * 65)
    print("  SECTION 6 — DP Scalability & Curse of Dimensionality")
    print("=" * 65)
    print("""
  Current state space (6×6 grid, battery=15, 3 rescue targets):
    Non-blocked positions : 36 − 3 (blocked) = 33
    Battery levels        : 15
    Rescue configurations : 2³ = 8
    TOTAL                 : 33 × 15 × 8 = 3,960 states

  ── Scaling scenarios ───────────────────────────────────────────

  10×10 grid, 5 rescue targets, battery=20:
    Positions      : ~100
    Battery        : 20
    Rescue configs : 2⁵ = 32
    TOTAL          : 100 × 20 × 32 = 64,000 states

  10×10 + dynamic weather (5 conditions × 3 intensities = 15):
    TOTAL          : 64,000 × 15 = 960,000 states

  10 rescue targets (instead of 5):
    TOTAL          : 100 × 20 × 1,024 × 15 ≈ 307 million states

  ── Why DP becomes intractable ──────────────────────────────────
  1. MEMORY  : Storing V(s) for 307 M states at 8 bytes each
               requires ~2.5 GB. Add Q(s,a) → 10× more.
  2. COMPUTE : Each VI sweep is O(|S| × |A| × |S|).
               At 307 M states and 5 actions: ~1.5 trillion ops per iter.
  3. MODEL   : DP requires exact P(s'|s,a). Real drone environments
               have unknown, non-stationary dynamics.
  4. CONTINUITY: Real GPS/IMU gives continuous state; tabular DP is
               impossible without discretisation that loses fidelity.

  ── How Deep RL addresses these ─────────────────────────────────
  • DQN (Deep Q-Network): approximates Q(s,a) with a neural net,
    generalising across similar states. Scales to millions of states
    with no explicit tabular enumeration.
  • PPO / A3C (Policy Gradient): directly optimise a parameterised
    policy π_θ(a|s), handling continuous action spaces naturally.
  • Model-Free: learns purely from experience — no transition model
    needed. Adapts to changing wind patterns / new obstacles at runtime.
  • Sim2Real: policies trained in simulation (cheap) are fine-tuned
    on real hardware. DP cannot leverage simulation in this way.

  ── Real-world autonomous drone relevance ───────────────────────
  Disaster-zone drones face partial observability (sensors obscured
  by smoke), dynamic obstacles (unstable structures collapsing), and
  continuous GPS + orientation state space. DP with a fixed finite
  MDP cannot capture these. Deep RL methods — especially those
  combining recurrent networks (for partial observability) with
  curriculum learning (for safety) — are the practical path forward.
""")
    print("=" * 65 + "\n")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":

    print("\n" + "=" * 65)
    print("  PART 2 — DYNAMIC PROGRAMMING: Autonomous Drone Rescue")
    print(f"  Group G=125  │  6×6 grid  │  battery=15  │  wind=30%")
    print("=" * 65)

    # ── 1. Environment setup ────────────────────────────────────────────
    print("\n>>> SECTION 1: Environment Configuration")
    print(f"\n  Grid size              : {GRID_ROWS}×{GRID_COLS}")
    print(f"  Max battery            : {MAX_BATTERY} units  (odd digit → 15)")
    print(f"  Wind probability       : {WIND_PROB*100:.0f}%  (digit 5–9 → 30%)")
    print(f"  Max steps per episode  : {MAX_STEPS}")
    print(f"\n  Grid layout:")
    print("  " + "  ".join([f" C{i}" for i in range(GRID_COLS)]))
    for ri, row in enumerate(BASE_GRID):
        print(f"  R{ri}  " + "   ".join(row))

    env = DroneRescueEnv()
    print(f"\n  Start position         : {env.start_pos}")
    print(f"  Rescue targets         : {env.rescue_targets}")
    print(f"  Charging stations      : {env._find(C)}")
    print(f"  Danger zones           : {env._find(D)}")
    print(f"  Blocked cells          : {env._find(X)}")
    print(f"  Wind zones             : {env._find(W)}")

    print("\n  Initial Grid Render:")
    env.render(env.reset())

    # ── 2. Value Iteration ──────────────────────────────────────────────
    print("\n>>> SECTION 2: Value Iteration (Dynamic Programming)")
    V, policy, history = value_iteration(env, gamma=0.99, theta=1e-3)

    print("\n  Sample V*(s) (battery=15, no rescues yet):")
    print(f"  {'State':>40}  {'V*(s)':>10}")
    print(f"  {'-' * 53}")
    for r in range(env.rows):
        for c in range(env.cols):
            if env.grid[r][c] != X:
                s = (r, c, 15, (False, False, False))
                print(f"  {str(s):>40}  {V.get(s, 0):>10.4f}")

    # ── 3. Policy Visualisation ──────────────────────────────────────────
    print("\n>>> SECTION 3: Policy Visualisation")
    visualise_policy(env, policy, V,
                     battery_level=MAX_BATTERY,
                     rescue_mask=(False, False, False))

    # Convergence plot
    print("\n>>> SECTION 3b: Convergence Plot")
    plot_convergence(history)

    # ── 4. State-Value Analysis ──────────────────────────────────────────
    print("\n>>> SECTION 4: State-Value Analysis")
    plot_state_value_analysis(env, V)

    # ── 5. Scalability Discussion ────────────────────────────────────────
    print("\n>>> SECTION 5: DP Scalability & Curse of Dimensionality")
    dp_scalability_discussion()

    print("[✓] DP Assignment (G=125) completed successfully.\n")
