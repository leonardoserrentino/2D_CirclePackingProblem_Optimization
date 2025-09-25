# 2-Dimensional Circle Packing in a Unit Square — Research Codebase

_A comprehensive and reproducible reference implementation for packing congruent circles inside a square under strict non-overlap and boundary constraints._

This repository brings together six algorithmic strategies—ranging from a naïve random baseline to projected gradient descent with explicit feasibility repair—to illustrate distinct optimization paradigms for geometric packing. The code is intentionally didactic and modular, enabling easy experimentation, benchmarking, and extension.

> **Scope:** The default domain is the unit square \(L=1\) with congruent circles of radius \(r\). All methods generalize straightforwardly to other square sizes \(L\) and, with minor changes, to heterogeneous radii and alternative planar domains.

---

## Table of Contents

- [1. Problem Statement](#1-problem-statement)
- [2. Repository Structure](#2-repository-structure)
- [3. Mathematical Formulation](#3-mathematical-formulation)
- [4. Algorithms and Scripts](#4-algorithms-and-scripts)
  - [4.1 Baseline Random Placement](#41-baseline-random-placement-01_baseline_randomplacementpy)
  - [4.2 Greedy Local Search](#42-greedy-local-search-02_localsearch_greedypackingpy)
  - [4.3 Gradient-Like Local Search](#43-gradient-like-local-search-03_localsearch_gradientlikepackingpy)
  - [4.4 Constraint-Aware Strategic Gradient](#44-constraint-aware-strategic-gradient-04_constraintaware_strategicgradientpy)
  - [4.5 Simulated Annealing](#45-simulated-annealing-05_metaheuristic_simulatedannealingpy)
  - [4.6 Projected Gradient Descent](#46-projected-gradient-descent-06_projectedgradientdescentpy)
- [5. Installation](#5-installation)
- [6. Usage](#6-usage)
- [7. Configuration Parameters](#7-configuration-parameters)

---

## 1. Problem Statement

We study the placement of \(N\) congruent circles of radius \(r\) inside a square of side length \(L\) (default \(L=1\)), subject to **hard constraints**:

- **Boundary feasibility:** each circle must lie entirely inside the square. For every center \(c_i=(x_i,y_i)\), we require
  \[ r \le x_i, y_i \le L - r.\]
- **Non-overlap:** for every distinct pair \(i \neq j\), 
  \[ \|c_i - c_j\| \ge 2r. \]

Operationally, each script maintains a set of centers and ensures feasibility either **a priori** (by sampling only from the admissible region) or **a posteriori** via projection and repair sweeps. The overarching practical objective is to push \(N\) as high as possible for a given \(r\) and \(L\), while preserving strict feasibility at all times.

---

## 2. Repository Structure

Six standalone Python scripts implement methodologically diverse strategies. Each script can be executed independently and produces both a **visualization** of the iterative process (GIF/MP4) and a **final static** plot of the resulting packing:

- `01_Baseline_RandomPlacement.py`
- `02_LocalSearch_GreedyPacking.py`
- `03_LocalSearch_GradientLikePacking.py`
- `04_ConstraintAware_StrategicGradient.py`
- `05_Metaheuristic_SimulatedAnnealing.py`
- `06_ProjectedGradientDescent.py`

---

## 3. Mathematical Formulation

While the _true_ objective (maximize \(N\) for given \(r\) and \(L\)) is discrete and combinatorial, our continuous optimization phases use a differentiable surrogate: the **sum of pairwise distances** among circle centers,
\[
f(C) \;=\; \sum_{1 \le i < j \le N} \|c_i - c_j\|, \qquad C = (c_1,\dots,c_N),\; c_i\in\mathbb{R}^2.
\]

Heuristically, **minimizing** \(f(C)\) compacts the set of centers while feasibility prevents overlap, acting as a proxy that encourages dense configurations. Gradient-based methods take steps along (sub)gradients of \(f\) and then **project back** to the feasible set, thereby coupling continuous optimization with strict constraint enforcement.

---

## 4. Algorithms and Scripts

### 4.1 Baseline Random Placement — `01_Baseline_RandomPlacement.py`

**Idea.** Uniformly sample candidate centers in the feasible box \([r,L-r]^2\); accept a candidate if all pairwise distances remain \(\ge 2r\). This simple strategy offers a **lower bound** on attainable counts and serves as a sanity check for feasibility routines.

**Typical parameters.** `L = 1.0`, `r = 0.1`, `max_iter = 2000` (see file for authoritative values).  
**Outputs.** `RandomAnimation.mp4`, `Random_FinalPacking.png`.

---

### 4.2 Greedy Local Search — `02_LocalSearch_GreedyPacking.py`

**Idea.** Alternate two phases:  
(i) **Greedy insertion** via feasible sampling until saturation;  
(ii) **Local stochastic moves** on a single center at a time. Accept a move if it preserves feasibility **and** reduces the total pairwise distance.

**Key knobs.** `OPT_ITERS`, `STEP_SIZE`, `FRAME_STEP`, and `MAX_ADD_FAIL`.  
**Outputs.** `packing_animation.gif`.

---

### 4.3 Gradient-Like Local Search — `03_LocalSearch_GradientLikePacking.py`

**Idea.** Repeated **short local-search phases** (random small displacements with improvement acceptance) interleaved with **attempted insertions** via feasible sampling. Terminates when insertions fail repeatedly or a maximum circle count is reached.

**Key knobs.** `OPT_ITERS = 2000`, `STEP_SIZE = 0.02`, `MAX_ADD_FAIL = 20000`, `MAX_CIRCLES = 26`.  
**Outputs.** `GradientAnimation.mp4`, `FinalPacking_Gradient.png`.

---

### 4.4 Constraint-Aware Strategic Gradient — `04_ConstraintAware_StrategicGradient.py`

**Idea.** Use a **fixed anchor** (e.g., the first circle at \((r,r)\)), then add new circles **preferentially along the square’s boundary** (left/right/top/bottom) before a local search pass that moves only a subset of circles. The boundary-biased insertions reduce initial conflicts and accelerate densification.

**Key knobs.** `MAX_ADD_FAIL`, `STEP_SIZE`, local-search iterations.  
**Outputs.** `packing_fixed_anchor.mp4`, `FinalPacking_FixedAnchor.png`.

---

### 4.5 Simulated Annealing (Metaheuristic) — `05_Metaheuristic_SimulatedAnnealing.py`

**Idea.** Treat the pairwise-distance sum as an **energy** and apply a Metropolis acceptance rule with geometric cooling. Proposals perturb one circle (excluding the anchor), then clamp to \([r,L-r]^2\) and check feasibility. Acceptance probability is \(\exp(-\Delta / T)\) for non-improving moves, with \(T\) decreasing over time.

**Key knobs.** `T0` (initial temperature), `ALPHA` (cooling factor), `T_MIN` (stop), `SA_ITERS`, `STEP_SIZE`.  
**Outputs.** `packing_simulated_annealing.mp4`, `FinalPacking_SimulatedAnnealing.png`.

---

### 4.6 Projected Gradient Descent — `06_ProjectedGradientDescent.py`

**Idea.** Alternate **gradient steps** on \(f(C)\) with **explicit projection** back to feasibility. Projection consists of:  
1) **Boundary clipping**: \(r \le x_i, y_i \le L-r\);  
2) **Pairwise repair sweeps**: if any pair violates \(d_{ij} \ge 2r\), move the two centers along their connecting direction to reinstate the clearance. Repeat sweeps until no violations remain or a sweep limit is reached.

**Key knobs.** `ETA` (learning rate), `ITERATIONS`, `MAX_SWEEPS`, `EPS` (numerical guard), `TOL` (convergence), `max_circles`.  
**Outputs.** `video.mp4`, `final.png`.

---

## 5. Installation

### 5.1 Python Dependencies

Install the required Python packages:

```bash
pip install numpy matplotlib imageio tqdm
```
Usage:

```bash
python 01_Baseline_RandomPlacement.py
python 02_LocalSearch_GreedyPacking.py
python 03_LocalSearch_GradientLikePacking.py
python 04_ConstraintAware_StrategicGradient.py
python 05_Metaheuristic_SimulatedAnnealing.py
python 06_ProjectedGradientDescent.py
```


**Artifacts** (saved in the working directory by default):

1) RandomAnimation.mp4, Random_FinalPacking.png (baseline).
2) packing_animation.gif (greedy).
3) GradientAnimation.mp4, FinalPacking_Gradient.png (gradient-like).
4) packing_fixed_anchor.mp4, FinalPacking_FixedAnchor.png (constraint-aware).
5) packing_simulated_annealing.mp4, FinalPacking_SimulatedAnnealing.png (SA).
6) video.mp4, final.png (PGD).

## 7. Configuration Parameters

Below are common parameters (see each script for authoritative values):

Geometry: `SQUARE_SIZE` = 1.0, `RADIUS` = 0.1; thus minimum center distance `MIN_DIST` = 2*RADIUS.

Insertion control: `MAX_ADD_FAIL` (maximum failed attempts before declaring saturation).

Local-search scale: `STEP_SIZE` (max random displacement per move, typically 0.02).

Budget limits: `OPT_ITERS`, `SA_ITERS`, or `ITERATIONS` (PGD).

PGD specifics: `ETA`, `MAX_SWEEPS`, `EPS`, `TOL`.

SA specifics: `T0`, `ALPHA`, `T_MIN`.

To explore different regimes (e.g., smaller radii or more aggressive optimization), edit these constants at the top of each script or wrap the scripts with a CLI (argparse) for batch experiments.
