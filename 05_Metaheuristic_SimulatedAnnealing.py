import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Parametri ---
RADIUS       = 0.1
MIN_DIST     = 2 * RADIUS
SQUARE_SIZE  = 1.0
SA_ITERS     = 5000    # iterazioni di Simulated Annealing per ogni passo
STEP_SIZE    = 0.02    # ampiezza del jitter casuale
FRAME_STEP   = 50      # salva frame ogni FRAME_STEP passi
MAX_CIRCLES  = 26
MAX_ADD_FAIL = 20000     # tentativi max per inserimento
T0           = 1.0     # temperatura iniziale
T_MIN        = 1e-3    # temperatura minima
ALPHA        = 0.995   # fattore di raffreddamento

# --- Funzioni di utilità ---
def is_valid(positions):
    """Verifica i vincoli hard: bordo e non sovrapposizione."""
    # bordo
    if np.any(positions < RADIUS) or np.any(positions > SQUARE_SIZE - RADIUS):
        return False
    # sovrapposizione
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < MIN_DIST:
                return False
    return True


def total_pairwise_distance(positions):
    """Somma delle distanze euclidee tra tutti i centri."""
    return sum(
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions))
        for j in range(i+1, len(positions))
    )

# --- Inizializzazione ---
# Primo cerchio fissato in (r, r)
positions = np.array([[RADIUS, RADIUS]])
frames = [positions.copy()]

# --- Ciclo principale: inserimento + SA optimization ---
for _ in range(MAX_CIRCLES - 1):
    # 1) Inserimento di un nuovo cerchio lungo i bordi
    added = False
    for _ in range(MAX_ADD_FAIL):
        side = np.random.choice(['left', 'right', 'bottom', 'top'])
        if side == 'left':
            p = np.array([RADIUS, np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS)])
        elif side == 'right':
            p = np.array([SQUARE_SIZE - RADIUS, np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS)])
        elif side == 'bottom':
            p = np.array([np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS), RADIUS])
        else:  # 'top'
            p = np.array([np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS), SQUARE_SIZE - RADIUS])

        # verifica distanza minima dalle altre posizioni
        if np.linalg.norm(p - positions, axis=1).min() >= MIN_DIST:
            positions = np.vstack([positions, p])
            frames.append(positions.copy())
            added = True
            break
    if not added:
        # impossibile aggiungere un ulteriore cerchio
        break

    # 2) Simulated Annealing per ottimizzare posizioni indice>=1
    T = T0
    current_dist = total_pairwise_distance(positions)
    for it in range(1, SA_ITERS + 1):
        idx = np.random.randint(1, len(positions))  # non muovere il primo cerchio
        candidate = positions.copy()
        candidate[idx] += np.random.uniform(-STEP_SIZE, STEP_SIZE, 2)
        candidate[idx] = np.clip(candidate[idx], RADIUS, SQUARE_SIZE - RADIUS)

        if not is_valid(candidate):
            # abbassa temperatura e continua
            T = max(T * ALPHA, T_MIN)
            continue
        new_dist = total_pairwise_distance(candidate)
        delta = new_dist - current_dist
        # criterio di Metropolis
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            positions = candidate
            current_dist = new_dist
        # raffreddamento
        T = max(T * ALPHA, T_MIN)
        # salva frame ogni FRAME_STEP
        if it % FRAME_STEP == 0:
            frames.append(positions.copy())

# --- Creazione e salvataggio MP4 ---
fig, ax = plt.subplots(figsize=(6,6))
canvas = FigureCanvas(fig)
writer = imageio.get_writer('packing_simulated_annealing.mp4', fps=5)

for pos in frames:
    ax.clear()
    ax.set_xlim(0, SQUARE_SIZE)
    ax.set_ylim(0, SQUARE_SIZE)
    ax.set_aspect('equal')
    ax.add_patch(plt.Rectangle((0, 0), SQUARE_SIZE, SQUARE_SIZE, fill=False))
    for p in pos:
        ax.add_patch(plt.Circle(p, RADIUS, fill=True, alpha=0.6))
    ax.text(
        0.02, 0.98,
        f"Count: {len(pos)}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor='none')
    )
    canvas.draw()
    buf = canvas.buffer_rgba()
    h, w = canvas.get_width_height()[::-1]
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    writer.append_data(img[:, :, :3])

writer.close()
print("MP4 salvato in 'packing_simulated_annealing.mp4'")

# --- Salvataggio immagine finale ---
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.set_xlim(0, SQUARE_SIZE)
ax2.set_ylim(0, SQUARE_SIZE)
ax2.set_aspect('equal')
ax2.set_title(f"Configurazione finale - {len(positions)} cerchi (1 fisso)")

# bordo del quadrato
ax2.add_patch(plt.Rectangle((0, 0), SQUARE_SIZE, SQUARE_SIZE, fill=False))

# disegno dei cerchi
for p in positions:
    ax2.add_patch(plt.Circle(p, RADIUS, fill=True, alpha=0.6))

# salva PNG
plt.savefig("FinalPacking_SimulatedAnnealing.png", dpi=300)
plt.close(fig2)
print(f"Immagine finale salvata in 'FinalPacking_SimulatedAnnealing.png' con {len(positions)} cerchi.")
