import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

RADIUS       = 0.1
MIN_DIST     = 2 * RADIUS
SQUARE_SIZE  = 1.0
MAX_ADD_FAIL = 200
OPT_ITERS    = 2000
STEP_SIZE    = 0.02
FRAME_STEP   = 20

def is_valid(positions):
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < MIN_DIST:
                return False
    return True

def total_pairwise_distance(positions):
    dist_sum = 0.0
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            dist_sum += np.linalg.norm(positions[i] - positions[j])
    return dist_sum

def try_add_circle(positions):
    for _ in range(MAX_ADD_FAIL):
        p = np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS, 2)
        if all(np.linalg.norm(p - q) >= MIN_DIST for q in positions):
            return np.vstack([positions, p])
    return positions

# Inizializzazione greedy
positions = np.empty((0, 2))
while True:
    new_positions = try_add_circle(positions)
    if new_positions.shape[0] == positions.shape[0]:
        break
    positions = new_positions

# Preallocazione per metriche e raccolta frames
distances = np.zeros(OPT_ITERS + 1)
counts    = np.zeros(OPT_ITERS + 1, dtype=int)
frames    = []

# Stato iniziale
distances[0] = total_pairwise_distance(positions)
counts[0]    = len(positions)
frames.append(positions.copy())

# Ottimizzazione + aggiunta dinamica + raccolta frames
for it in range(1, OPT_ITERS + 1):
    idx = np.random.randint(0, len(positions))
    candidate = positions.copy()
    candidate[idx] += np.random.uniform(-STEP_SIZE, STEP_SIZE, 2)
    candidate[idx] = np.clip(candidate[idx], RADIUS, SQUARE_SIZE - RADIUS)
    if is_valid(candidate) and total_pairwise_distance(candidate) < total_pairwise_distance(positions):
        positions = candidate

    new_positions = try_add_circle(positions)
    if new_positions.shape[0] > positions.shape[0]:
        positions = new_positions

    distances[it] = total_pairwise_distance(positions)
    counts[it]    = len(positions)

    if it % FRAME_STEP == 0:
        frames.append(positions.copy())

# ANIMAZIONE
fig, ax = plt.subplots(figsize=(6,6))
def update(frame_positions):
    ax.clear()
    ax.set_xlim(0, SQUARE_SIZE)
    ax.set_ylim(0, SQUARE_SIZE)
    ax.set_aspect('equal')
    ax.add_patch(plt.Rectangle((0,0), SQUARE_SIZE, SQUARE_SIZE, fill=False))
    for p in frame_positions:
        ax.add_patch(plt.Circle(p, RADIUS, fill=True, alpha=0.6))
    return ax.patches

anim = FuncAnimation(fig, update, frames=frames, blit=False, interval=200)
anim.save('packing_animation.gif', writer='pillow', fps=5)
plt.close(fig)
print("Animazione salvata in 'packing_animation.gif'")
