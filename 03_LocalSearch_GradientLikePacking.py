import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# --- Parametri ---
RADIUS       = 0.1
MIN_DIST     = 2 * RADIUS
SQUARE_SIZE  = 1.0
OPT_ITERS    = 2000    # iters di local search prima di ogni aggiunta
STEP_SIZE    = 0.02    # ampiezza passi casuali
FRAME_STEP   = 20      # ogni quanti its salvo un frame
MAX_CIRCLES  = 26
MAX_ADD_FAIL = 20000

# --- Funzioni di utilità ---
def is_valid(positions):
    if np.any(positions < RADIUS) or np.any(positions > SQUARE_SIZE - RADIUS):
        return False
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if np.linalg.norm(positions[i] - positions[j]) < MIN_DIST:
                return False
    return True

def total_pairwise_distance(positions):
    return sum(
        np.linalg.norm(positions[i] - positions[j])
        for i in range(len(positions))
        for j in range(i+1, len(positions))
    )

def try_add_circle(positions):
    """
    Prova fino a MAX_ADD_FAIL volte a inserire un nuovo cerchio in modo casuale:
    - campiona uniformemente un punto valido all’interno del quadrato
    - accetta solo se rispetta MIN_DIST da tutti gli altri cerchi
    """
    for _ in range(MAX_ADD_FAIL):
        p = np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS, 2)
        if all(np.linalg.norm(p - q) >= MIN_DIST for q in positions):
            return np.vstack([positions, p])
    # se non trova mai un punto valido, restituisce la configurazione invariata
    return positions


# --- Inizializzazione ---
positions = np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS, (1, 2))
frames = [positions.copy()]

# --- Ciclo principale: pre-opt + aggiunta mirata ---
for _ in range(MAX_CIRCLES - 1):
    # local search
    for it in range(1, OPT_ITERS + 1):
        idx = np.random.randint(0, len(positions))
        candidate = positions.copy()
        candidate[idx] += np.random.uniform(-STEP_SIZE, STEP_SIZE, 2)
        candidate[idx] = np.clip(candidate[idx], RADIUS, SQUARE_SIZE - RADIUS)
        if is_valid(candidate) and total_pairwise_distance(candidate) < total_pairwise_distance(positions):
            positions = candidate
        if it % FRAME_STEP == 0:
            frames.append(positions.copy())

    # aggiunta cerchio
    new_positions = try_add_circle(positions)
    if new_positions.shape[0] == positions.shape[0]:
        break
    positions = new_positions
    frames.append(positions.copy())

# --- Salvataggio MP4 con annotazione del conteggio cerchi ---
fig, ax = plt.subplots(figsize=(6,6))
canvas = FigureCanvas(fig)
writer = imageio.get_writer('GradientAnimation.mp4', fps=5)

for pos in frames:
    ax.clear()
    ax.set_xlim(0, SQUARE_SIZE)
    ax.set_ylim(0, SQUARE_SIZE)
    ax.set_aspect('equal')

    # bordo del quadrato
    ax.add_patch(plt.Rectangle((0,0), SQUARE_SIZE, SQUARE_SIZE, fill=False))

    # disegno dei cerchi
    for p in pos:
        ax.add_patch(plt.Circle(p, RADIUS, fill=True, alpha=0.6))

    # annotazione del numero di cerchi
    ax.text(
        0.02, 0.98,
        f"Count: {len(pos)}",
        fontsize=12,
        color="black",
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor='none')
    )

    # cattura frame
    canvas.draw()
    buf = canvas.buffer_rgba()
    h, w = canvas.get_width_height()[::-1]
    img = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 4))
    writer.append_data(img[:, :, :3])

writer.close()

# --- Salvataggio immagine finale ---
fig2, ax2 = plt.subplots(figsize=(6,6))
ax2.set_xlim(0, SQUARE_SIZE)
ax2.set_ylim(0, SQUARE_SIZE)
ax2.set_aspect('equal')
ax2.set_title(f"Configurazione finale - {len(positions)} cerchi")

# bordo del quadrato
ax2.add_patch(plt.Rectangle((0,0), SQUARE_SIZE, SQUARE_SIZE, fill=False))

# disegna tutti i cerchi finali
for p in positions:
    ax2.add_patch(plt.Circle(p, RADIUS, fill=True, alpha=0.6))

# Salva immagine
plt.savefig("FinalPacking_Gradient.png", dpi=300)
plt.close(fig2)
print(f"Immagine finale salvata in 'FinalPacking_Gradient.png' con {len(positions)} cerchi.")
