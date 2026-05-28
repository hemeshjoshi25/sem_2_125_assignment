"""
PS13 routing — Dijkstra vs Ant Colony Optimization (two scenarios from GLOBAL config).

After ``END_GLOBAL`` the file lists graph edges only. **Source s and destination t are always
typed at the run prompt** (never read from the input file). Supports one flat edge list or
multiple ``Case …`` blocks sharing the same prompted ``(s,t)``.

  python PS13_ACO_Routing.py
  python PS13_ACO_Routing.py -i inputPS13.txt

"""

from __future__ import annotations

import argparse
import heapq
import math
import os
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


# -----------------------------------------------------------------------------
# Graph + Dijkstra (exact baseline) + ACO (two scenarios from BEGIN_GLOBAL).
# -----------------------------------------------------------------------------

def _project_root() -> Path:
    # Notebook cells often paste this module without __file__; cwd is the fallback.
    try:
        return Path(__file__).resolve().parent
    except NameError:
        return Path.cwd()


PROJECT_ROOT = _project_root()


def build_adjacency(
    edges: list[tuple[int, int, float]],
) -> dict[int, dict[int, float]]:
    """Return nonnegative undirected latencies keyed by router IDs.

    Zero latency is allowed for Dijkstra; ACO uses η = 1/max(w, ε) so the heuristic stays finite.
    """
    g: dict[int, dict[int, float]] = defaultdict(dict)
    for u, v, w in edges:
        if u == v:
            raise ValueError("Self-loops are not supported for routing.")
        if w < 0:
            raise ValueError("Negative latency cannot be routed with plain Dijkstra.")
        # Undirected: store both (u,v) and (v,u); last repeated edge wins for same pair.
        g[u][v] = float(w)
        g[v][u] = float(w)
    return dict(g)


def _vertices_from_edges(edges: Iterable[tuple[int, int, float]]) -> set[int]:
    nodes: set[int] = set()
    for u, v, _ in edges:
        nodes.add(u)
        nodes.add(v)
    return nodes


def shortest_path_latency(
    graph: dict[int, dict[int, float]],
    *,
    source: int,
    dest: int,
) -> tuple[list[int] | None, float]:
    """Shortest nonnegative path from ``source`` to ``dest`` (Dijkstra).

    Uses a binary min-heap without decrease-key: stale (distance, node) pairs are
    skipped when popped (``d > dist[u]``). Correct for nonnegative edge weights.
    """
    if source == dest:
        return [source], 0.0

    if source not in graph or dest not in graph:
        return None, float("inf")

    dist = {n: float("inf") for n in graph}
    prev: dict[int, int | None] = {}
    dist[source] = 0.0
    prev[source] = None
    # Entries are tentative shortest distances discovered so far.
    heap: list[tuple[float, int]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # Outdated relaxation; a shorter path to u was already finalized.
        if u == dest:
            return _walk_back(prev, dest), d
        for v, w in graph[u].items():
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev[v] = u
                heapq.heappush(heap, (nd, v))

    return None, float("inf")


def _walk_back(prev: dict[int, int | None], dest: int) -> list[int]:
    seq: list[int] = []
    cur: int | None = dest
    while cur is not None:
        seq.append(cur)
        cur = prev.get(cur)
    seq.reverse()
    return seq


# ---------- Ant Colony Optimization ----------
# Assignment heuristic: desirability of edge (i,j) proportional to η = 1 / latency_ij.
# Pheromone τ lives on undirected edges, keyed canonically via edge_key().
EdgeKey = tuple[int, int]


def edge_key(u: int, v: int) -> EdgeKey:
    """One storage slot per undirected link so τ(u,v) and τ(v,u) stay identical."""
    return (u, v) if u < v else (v, u)


# η = 1/w blows up when w == 0; clamp so ants can still traverse zero-latency links numerically.
_ACO_LATENCY_EPS = 1e-12


class ACOHyperParams:
    """One assignment scenario: colony size, α/β/ρ, iterations, optional Q and initial τ0."""

    __slots__ = (
        "n_ants",
        "alpha",
        "beta",
        "rho",
        "n_iterations",
        "q",
        "tau0",
        "seed",
        "explore_prob",
        "label",
    )

    def __init__(
        self,
        *,
        n_ants: int,
        alpha: float,
        beta: float,
        rho: float,
        n_iterations: int,
        q: float = 1.0,
        tau0: float | None = None,
        seed: int = 0,
        explore_prob: float = 0.05,
        label: str = "",
    ) -> None:
        # α, β scale pheromone vs heuristic in the random-proportional transition rule.
        # ρ is applied once per iteration (global evaporation before adding deposits).
        self.n_ants = n_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.n_iterations = n_iterations
        self.q = q
        self.tau0 = tau0
        self.seed = seed
        self.explore_prob = explore_prob
        self.label = label


def run_ant_colony(
    edges: list[tuple[int, int, float]],
    *,
    source: int,
    dest: int,
    scenario: ACOHyperParams,
) -> dict[str, object]:
    """Run one full ACO schedule; compare stochastic best tour vs Dijkstra on the same (s,t)."""
    graph = build_adjacency(edges)

    # Trivial query s==t: zero-length tour — Q/L deposition is undefined when L==0.
    if source == dest:
        dp, dc = shortest_path_latency(graph, source=source, dest=dest)
        history = [0.0] * scenario.n_iterations
        return {
            "best_path": [source],
            "best_cost": 0.0,
            "best_iteration": 0,
            "convergence_first_opt": 0,
            "optimal_path": dp,
            "optimal_cost": dc,
            "history_best": history,
        }

    # Unique undirected arcs for symmetric τ updates.
    edges_set: set[EdgeKey] = set()
    for u, nbrs in graph.items():
        for v in nbrs:
            edges_set.add(edge_key(u, v))

    n_nodes = len(graph)
    tau0 = scenario.tau0 if scenario.tau0 is not None else 1.0 / max(n_nodes, 1)
    tau_floor = tau0 / (5.0 * max(len(edges_set), 1))  # Keeps probabilities defined on weak edges.
    pheromone: dict[EdgeKey, float] = {e: tau0 for e in edges_set}

    # Exact optimum for convergence diagnostics (assignment part c).

    dij_path, dij_cost = shortest_path_latency(
        graph,
        source=source,
        dest=dest,
    )

    rng = random.Random(scenario.seed)
    best_path: list[int] | None = None
    best_cost = math.inf
    best_iter = -1
    convergence_first_opt: int | None = None
    history_best: list[float] = []

    def eta(cur: int, nxt: int) -> float:
        # Inverse latency; zero-cost edges would make 1/w infinite — clamp w for η (Dijkstra still uses true w).
        w = graph[cur][nxt]
        return 1.0 / max(float(w), _ACO_LATENCY_EPS)

    def tau_uv(cur: int, nxt: int) -> float:
        return pheromone[edge_key(cur, nxt)]

    for it in range(scenario.n_iterations):
        deposits: defaultdict[EdgeKey, float] = defaultdict(float)

        for _ant in range(scenario.n_ants):
            visited: set[int] = {source}
            snap: list[int] = [source]
            current = source

            while current != dest:
                candidates = [j for j in graph[current] if j not in visited]
                if not candidates:
                    break  # Dead-end: cannot reach t along a simple path this attempt.

                if rng.random() < scenario.explore_prob:
                    # ε exploration: unbiased random successor to escape premature exploitation.
                    nxt = rng.choice(candidates)
                else:
                    # Random-proportional rule: P(j) ∝ τ^α η^β over feasible neighbours j.
                    probs: list[float] = []
                    for j in candidates:
                        trails = max(tau_uv(current, j), 1e-12)
                        heur = eta(current, j)
                        probs.append((trails**scenario.alpha) * (heur**scenario.beta))
                    mass = sum(probs)
                    if mass <= 0:
                        break
                    r = rng.uniform(0.0, mass)
                    acc = 0.0
                    nxt = candidates[-1]
                    for j, p in zip(candidates, probs):
                        acc += p
                        if r <= acc:
                            nxt = j
                            break

                snap.append(nxt)
                current = nxt
                visited.add(current)

            if current == dest:
                length = sum(graph[snap[i]][snap[i + 1]] for i in range(len(snap) - 1))
                # Q/L rule needs L>0; also skip non-finite totals on degenerate paths.
                if length > 0 and math.isfinite(length):
                    dep = scenario.q / length
                    for i in range(len(snap) - 1):
                        deposits[edge_key(snap[i], snap[i + 1])] += dep
                if length < best_cost:
                    best_cost = length
                    best_path = list(snap)
                    best_iter = it
                    # First macro-iteration where best-so-far cost hits Dijkstra (convergence probe).
                    if math.isfinite(dij_cost) and math.isclose(length, dij_cost):
                        if convergence_first_opt is None:
                            convergence_first_opt = it

        for ek in edges_set:
            # τ ← (1−ρ)·τ + Δτ; ρ is evaporation rate from the assignment scenarios.
            pheromone[ek] = max(
                (1.0 - scenario.rho) * pheromone[ek] + deposits[ek],
                tau_floor,
            )
        history_best.append(best_cost)

    # best_* = global best tour so far; history_best = best cost at end of each iteration.
    return {
        "best_path": best_path,
        "best_cost": best_cost,
        "best_iteration": best_iter,
        "convergence_first_opt": convergence_first_opt,
        "optimal_path": dij_path,
        "optimal_cost": dij_cost,
        "history_best": history_best,
    }


# ---------- Input file: BEGIN_GLOBAL … END_GLOBAL; edges only (flat list or Case blocks) ----------
@dataclass(slots=True)
class RoutingCase:
    """Edges for one benchmark + display name; ``source``/``dest`` are set after stdin prompt."""

    name: str
    source: int
    dest: int
    edges: list[tuple[int, int, float]]


@dataclass(slots=True)
class RunConfig:
    """Parsed GLOBAL block: shared iteration count, PRNG seed, and both ACO scenarios."""

    output_file: str
    iterations: int
    seed: int
    explore_prob: float
    scenario1: ACOHyperParams
    scenario2: ACOHyperParams


def _filter_config_lines(lines: Iterable[str]) -> list[str]:
    """Drop blanks and # comments so input files can carry human-readable notes."""
    filtered: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        filtered.append(stripped)
    return filtered


def _read_filtered_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return _filter_config_lines(handle)


def _find_token(lines: list[str], token: str, start: int = 0) -> int:
    for idx in range(start, len(lines)):
        if lines[idx] == token:
            return idx
    return -1


def _parse_kv_block(chunk: list[str]) -> dict[str, str]:
    kv: dict[str, str] = {}
    for line in chunk:
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"Malformed GLOBAL line (expected KEY VALUE): {line!r}")
        key = parts[0].upper()
        value = " ".join(parts[1:])
        kv[key] = value
    return kv


def _materialize_run_config(kv: dict[str, str], *, source_path: str) -> RunConfig:
    """Build RunConfig plus two scenario objects from GLOBAL key=value lines."""

    def req_int(name: str) -> int:
        if name not in kv:
            raise KeyError(f"Missing required GLOBAL key {name} in {source_path!r}")
        return int(float(kv[name]))

    def req_float(name: str) -> float:
        if name not in kv:
            raise KeyError(f"Missing required GLOBAL key {name} in {source_path!r}")
        return float(kv[name])

    output_file = kv.get("OUTPUT", "outputPS13.txt")
    iterations = req_int("ITERATIONS")
    seed = req_int("SEED")
    explore = float(kv.get("EXPLORE_PROB", "0.05"))

    s1 = ACOHyperParams(
        n_ants=req_int("ANTS_S1"),
        alpha=req_float("ALPHA_S1"),
        beta=req_float("BETA_S1"),
        rho=req_float("RHO_S1"),
        n_iterations=iterations,
        q=req_float("Q") if "Q" in kv else 1.0,
        tau0=req_float("TAU0") if "TAU0" in kv else None,
        seed=seed,
        explore_prob=explore,
        label="Scenario 1",
    )
    s2 = ACOHyperParams(
        n_ants=req_int("ANTS_S2"),
        alpha=req_float("ALPHA_S2"),
        beta=req_float("BETA_S2"),
        rho=req_float("RHO_S2"),
        n_iterations=iterations,
        q=req_float("Q") if "Q" in kv else 1.0,
        tau0=req_float("TAU0") if "TAU0" in kv else None,
        seed=seed,
        explore_prob=explore,
        label="Scenario 2",
    )

    return RunConfig(
        output_file=output_file,
        iterations=iterations,
        seed=seed,
        explore_prob=explore,
        scenario1=s1,
        scenario2=s2,
    )


def _parse_edges_after_global(lines: list[str], *, path: str) -> list[tuple[int, int, float]]:
    """Lines after END_GLOBAL: each `u v w` is an undirected edge; other lines ignored."""
    edges: list[tuple[int, int, float]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
        except ValueError:
            continue
        edges.append((u, v, w))
    if not edges:
        raise ValueError(
            f"No edge lines (u v w) found after END_GLOBAL in {path!r}. "
            "List one edge per line."
        )
    return edges


def _line_is_edge_triple(line: str) -> bool:
    parts = line.split()
    if len(parts) != 3:
        return False
    try:
        int(parts[0])
        int(parts[1])
        float(parts[2])
    except ValueError:
        return False
    return True


def _parse_named_case_blocks(lines: list[str], *, path: str) -> list[RoutingCase]:
    """Parse ``Case name`` headings followed only by ``u v w`` edges; ``s,t`` come from stdin later."""
    cases: list[RoutingCase] = []
    idx = 0
    while idx < len(lines):
        name = lines[idx]
        idx += 1
        edge_rows: list[tuple[int, int, float]] = []
        while idx < len(lines):
            if not _line_is_edge_triple(lines[idx]):
                break
            parts = lines[idx].split()
            u, v, w = int(parts[0]), int(parts[1]), float(parts[2])
            edge_rows.append((u, v, w))
            idx += 1

        if not edge_rows:
            raise ValueError(
                f"No edge lines after case heading {name!r} (expected u v latency rows) "
                f"in {path!r}."
            )
        # Placeholders; ``main()`` sets ``source``/``dest`` after the interactive prompt.
        cases.append(RoutingCase(name=name, source=0, dest=0, edges=edge_rows))

    return cases


def parse_input_document(
    path: str,
) -> tuple[RunConfig, list[RoutingCase], list[tuple[int, int, float]] | None]:
    """Load GLOBAL plus graph edges; ``s``/``t`` are applied in ``main()`` after user input."""

    raw_lines = _read_filtered_lines(path)

    global_idx = _find_token(raw_lines, "BEGIN_GLOBAL")
    if global_idx == -1:
        raise ValueError(
            f"{path!r} must contain a BEGIN_GLOBAL … END_GLOBAL configuration block."
        )
    end_idx = _find_token(raw_lines, "END_GLOBAL", start=global_idx + 1)
    if end_idx == -1:
        raise ValueError(f"{path!r} is missing END_GLOBAL after BEGIN_GLOBAL.")

    kv = _parse_kv_block(raw_lines[global_idx + 1 : end_idx])
    run_cfg = _materialize_run_config(kv, source_path=path)

    tail = raw_lines[end_idx + 1 :]
    if not tail:
        raise ValueError(f"No graph data after END_GLOBAL in {path!r}.")

    # First row is an edge triple → entire tail is one flat undirected graph.
    if _line_is_edge_triple(tail[0]):
        return run_cfg, [], _parse_edges_after_global(tail, path=path)

    return run_cfg, _parse_named_case_blocks(tail, path=path), None


def prompt_source_dest(*, vertices: set[int]) -> tuple[int, int]:
    """Ask for ``s`` and ``t``; both must appear as endpoints in the loaded edge list(s)."""

    verts = sorted(vertices)
    print("Routers referenced in the input file:", ", ".join(str(v) for v in verts))
    while True:
        try:
            s = int(input("Source router s: ").strip())
            t_node = int(input("Destination router t: ").strip())
        except EOFError:
            raise SystemExit("EOF while reading s and t.") from None
        except ValueError:
            print("Enter integer router IDs.")
            continue
        if s not in vertices:
            print(f"Router {s} does not appear in any edge in the input file.")
            continue
        if t_node not in vertices:
            print(f"Router {t_node} does not appear in any edge in the input file.")
            continue
        return s, t_node


def resolve_input_path(user_path: str | None, script_dir: str) -> str:
    """Honor ``-i`` path; otherwise look beside the script/notebook for default filenames."""

    if user_path:
        candidates = [user_path, os.path.join(script_dir, user_path)]
        for candidate in candidates:
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        raise FileNotFoundError(f"Input file not found (tried {candidates}).")

    for fname in ("inputPS13.txt", "inputPSXX.txt"):
        probe = os.path.join(script_dir, fname)
        if os.path.isfile(probe):
            return os.path.abspath(probe)
    raise FileNotFoundError(
        "No input file. Pass -i path/to/inputPS13.txt or place "
        f"inputPS13.txt beside the script (looked in {script_dir!r})."
    )


def format_path(path: list[int] | None) -> str:
    """Pretty-print a hop list using arrow separators."""
    if not path:
        return "(no path)"
    return " → ".join(str(node) for node in path)


def format_latency(cost: float) -> str:
    """Plain numbers for integer latencies (matches sample OUTPUT style)."""
    if not math.isfinite(cost):
        return str(cost)
    near = round(cost)
    if math.isclose(cost, near, rel_tol=0.0, abs_tol=1e-9):
        return str(int(near))
    return f"{cost:.6g}"


def collect_case_benchmarks(
    case: RoutingCase,
    *,
    runner: RunConfig,
    lines_out: list[str],
    skip_dijkstra_banner: bool = False,
) -> dict[str, list[dict[str, object]]]:
    """Run Dijkstra once, both ACO scenarios, append human-readable benchmarks to ``lines_out``."""

    sep = "=" * 72
    graph = build_adjacency(case.edges)

    # Tabular rows later fed into assemble_report for side-by-side convergence notes.
    bundle: dict[str, list[dict[str, object]]] = {"scenario1": [], "scenario2": []}

    dij_path, dij_cost = shortest_path_latency(
        graph,
        source=case.source,
        dest=case.dest,
    )

    if not skip_dijkstra_banner:
        lines_out.append(sep)
        lines_out.append(f"{case.name} — Dijkstra (deterministic shortest path)")
        lines_out.append(sep)
        lines_out.append(f"Best Path: {format_path(dij_path)}")
        lines_out.append(f"Minimum Latency: {format_latency(dij_cost)}")
        lines_out.append("")
    else:
        lines_out.append(sep)
        lines_out.append(f"{case.name} — ACO detailed results")
        lines_out.append(sep)
        lines_out.append("")

    for key, hp, human_label in (
        (
            "scenario1",
            runner.scenario1,
            f"Scenario 1 (α={runner.scenario1.alpha}, β={runner.scenario1.beta}, ρ={runner.scenario1.rho})",
        ),
        (
            "scenario2",
            runner.scenario2,
            f"Scenario 2 (α={runner.scenario2.alpha}, β={runner.scenario2.beta}, ρ={runner.scenario2.rho})",
        ),
    ):
        result = run_ant_colony(
            case.edges,
            source=case.source,
            dest=case.dest,
            scenario=hp,
        )
        # Assignment (c): check if stochastic best equals provably shortest latency.
        matched = math.isfinite(dij_cost) and math.isclose(
            float(result["best_cost"]), dij_cost
        )
        bundle[key].append(
            {
                "case_name": case.name,
                "scenario": human_label,
                "first_opt_iter": result["convergence_first_opt"],
                "best_iteration": result["best_iteration"],
                "best_cost": result["best_cost"],
                "opt_cost": dij_cost,
                "matched_opt": matched,
            }
        )

        lines_out.append(f"--- ACO {human_label} ---")
        lines_out.append(f"Best Path: {format_path(result['best_path'])}")
        lines_out.append(f"Minimum Latency (best found): {format_latency(float(result['best_cost']))}")
        co_val = result["convergence_first_opt"]
        lines_out.append(
            "First iteration where best-so-far matched Dijkstra optimum: "
            f"{co_val if co_val is not None else 'not reached in run'}"
        )
        lines_out.append(f"Iteration of global best tour: {int(result['best_iteration'])}")
        lines_out.append("")

    lines_out.append("")
    return bundle


# Banner width for plaintext section titles in the OUTPUT file.
SEP = "=" * 72


def assemble_report(run_cfg: RunConfig, cases: list[RoutingCase], *, resolved_input: str) -> str:
    """Full assignment write-up: Dijkstra case index, PEAS, ACO detail, scenario comparison (b)(c)."""

    lines: list[str] = []
    collect: dict[str, list[dict[str, object]]] = {"scenario1": [], "scenario2": []}

    lines.append("ROUTING SUMMARY — Dijkstra vs ACO (two scenarios)")
    lines.append(f"INPUT FILE: {resolved_input}")
    lines.append("")

    # Up-front optimal paths (Dijkstra) per routing case — matches requested OUTPUT layout.
    for case in cases:
        g = build_adjacency(case.edges)
        dp, dc = shortest_path_latency(
            g,
            source=case.source,
            dest=case.dest,
        )
        lines.append(case.name)
        lines.append(f"Best Path: {format_path(dp)}")
        lines.append(f"Minimum Latency: {format_latency(dc)}")
        lines.append("")

    lines.append(SEP)
    lines.append("(a) PEAS for the routing agent")
    lines.append(SEP)
    lines.append("")
    lines.append(
        "Performance measure: minimise end-to-end latency on path s→t; track "
        "iterations until the best-found path matches Dijkstra’s optimum."
    )
    lines.append(
        "Environment: weighted undirected router graph; source and destination are "
        "entered at runtime (not taken from the input file)."
    )
    lines.append(
        "Actuators: choose next-hop routers; evaporate/deposit pheromone on edges."
    )
    lines.append(
        "Sensors: neighbour IDs, edge latencies, pheromone τ, heuristic η = 1/latency."
    )
    lines.append("")

    for case in cases:
        payload = collect_case_benchmarks(
            case,
            runner=run_cfg,
            lines_out=lines,
            skip_dijkstra_banner=True,
        )
        for key in collect:
            collect[key].extend(payload[key])

    lines.append(SEP)
    lines.append("(b)-(c) Scenarios vs Dijkstra, convergence, parameter effects")
    lines.append(SEP)
    lines.append(
        "η_ij = 1/latency_ij; transition probability uses τ^α η^β (with exploration ε "
        f"={run_cfg.explore_prob:g}). ρ controls trail evaporation between iterations."
    )

    ordered: list[dict[str, object]] = []
    n = len(collect["scenario1"])
    # Pair scenario-1 and scenario-2 rows per case for readability in the output file.
    for idx in range(n):
        ordered.append(collect["scenario1"][idx])
        ordered.append(collect["scenario2"][idx])
    for row in ordered:
        co = row["first_opt_iter"]
        lines.append(
            f"- {row['case_name']} / {row['scenario']}: "
            f"first iteration matching optimum {co if co is not None else 'never'}, "
            f"matched Dijkstra = {row['matched_opt']}."
        )

    lines.append("")
    lines.append(
        "α weights pheromone preference; β weights inverse-latency (short cheap edges); "
        "compare first-hit iterations above for convergence speed between scenarios."
    )

    any_fail = any(not bool(row["matched_opt"]) for row in ordered)
    if any_fail:
        lines.append("")
        lines.append(
            "[WARN] At least one ACO run did not match Dijkstra within the iteration "
            "budget — try more ITERATIONS or higher exploration ε."
        )
        lines.append("")

    return "\n".join(lines) + "\n"


def main(cli_input_path: str | None) -> None:
    """Load edges from disk, prompt for ``(s,t)``, save report."""
    script_root = PROJECT_ROOT
    try:
        resolved = resolve_input_path(cli_input_path, str(script_root))
        run_cfg, cases, flat_edges = parse_input_document(resolved)

        if flat_edges is not None:
            vertex_set = _vertices_from_edges(flat_edges)
            print(f"Loaded GLOBAL config + flat edge list from: {resolved}")
            s, t_node = prompt_source_dest(vertices=vertex_set)
            cases = [RoutingCase(name="Case 1", source=s, dest=t_node, edges=flat_edges)]
        else:
            vertex_set = set()
            for c in cases:
                vertex_set |= _vertices_from_edges(c.edges)
            print(
                f"Loaded GLOBAL config + {len(cases)} case graph(s) from: {resolved}"
            )
            s, t_node = prompt_source_dest(vertices=vertex_set)
            for c in cases:
                c.source = s
                c.dest = t_node

        print()
        report = assemble_report(run_cfg, cases, resolved_input=resolved)
        out_path = (
            Path(run_cfg.output_file)
            if Path(run_cfg.output_file).is_absolute()
            else script_root / run_cfg.output_file
        )
        out_path.write_text(report, encoding="utf-8")
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    except (ValueError, KeyError, OSError) as exc:
        print(f"Input/configuration error: {exc}", file=sys.stderr)
        sys.exit(5)
    else:
        print(f"\nWrote report: {out_path}")


def RUN_BATCH(cli_path: str | None = None) -> None:
    """Notebook shim: anchor paths to the kernel cwd, then defer to ``main``."""

    global PROJECT_ROOT
    PROJECT_ROOT = Path.cwd()
    main(cli_path)


def _cli() -> None:
    """``python PS13_ACO_Routing.py [-i path]`` — optional path defaults via ``resolve_input_path``."""
    parser = argparse.ArgumentParser(
        description="PS13: load graph from file, prompt for s/t, compare Dijkstra vs ACO.",
    )
    parser.add_argument(
        "--input",
        "-i",
        default=None,
        metavar="PATH",
        help="After GLOBAL: optional Case blocks or only u v w edges; s and t are prompted.",
    )
    args = parser.parse_args()
    main(args.input)


if __name__ == "__main__":
    _cli()
