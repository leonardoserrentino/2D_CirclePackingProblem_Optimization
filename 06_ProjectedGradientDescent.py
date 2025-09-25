import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from tqdm import tqdm

# Parametri globali
RADIUS       = 0.1
MIN_DIST     = 2 * RADIUS
SQUARE_SIZE  = 1.0
ITERATIONS   = 6000       # limite massimo iterazioni per fase
ETA          = 0.001      # learning rate
EPS          = 1e-8
FRAME_STEP   = 10
MAX_SWEEPS   = 50
TOL          = 1e-6       # criterio stop su loss

# -----------------------------
# Funzione obiettivo e gradiente
# -----------------------------
def objective(C):
    n = len(C)
    return sum(np.linalg.norm(C[i]-C[j]) for i in range(n) for j in range(i+1, n))

def gradient(C):
    n = len(C)
    g = np.zeros_like(C)
    for i in range(n):
        for j in range(i+1, n):
            diff = C[i] - C[j]
            d = np.linalg.norm(diff)
            if d > EPS:
                dir_ = diff/d
                g[i] += dir_
                g[j] -= dir_
            else:
                rand = np.random.uniform(-1,1,2)
                rand /= (np.linalg.norm(rand)+EPS)
                g[i] += rand
                g[j] -= rand
    return g

# -----------------------------
# Proiezione vincoli
# -----------------------------
def project_feasible(C, max_sweeps=MAX_SWEEPS):
    P = C.copy()
    n = len(P)
    for _ in range(max_sweeps):
        violations = 0
        # bordo
        P[:,0] = np.clip(P[:,0], RADIUS, SQUARE_SIZE - RADIUS)
        P[:,1] = np.clip(P[:,1], RADIUS, SQUARE_SIZE - RADIUS)
        # coppie
        for i in range(n):
            for j in range(i+1, n):
                diff = P[i] - P[j]
                d = np.linalg.norm(diff)
                if d < MIN_DIST:
                    violations += 1
                    if d < EPS:
                        dir_ = np.random.uniform(-1,1,2)
                        dir_ /= (np.linalg.norm(dir_)+EPS)
                    else:
                        dir_ = diff/d
                    delta = 0.5*(MIN_DIST - d)*dir_
                    P[i] += delta
                    P[j] -= delta
        if violations == 0:
            break
    return P

# -----------------------------
# Aggiunta di un cerchio valido
# -----------------------------
def add_circle(positions):
    new_circle = np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS, 2)
    return np.vstack([positions, new_circle])


# -----------------------------
# Esecuzione unica da 1 a max_circles
# -----------------------------
def run_gd_session(max_circles=26):
    positions = np.random.uniform(RADIUS, SQUARE_SIZE - RADIUS, (1,2))
    positions = project_feasible(positions)

    frames = [positions.copy()]
    writer = imageio.get_writer("video.mp4", fps=5)
    fig, ax = plt.subplots(figsize=(6,6))
    canvas = FigureCanvas(fig)

    for n_circles in tqdm(range(1, max_circles+1)):
        # ottimizza configurazione corrente
        loss_log = [objective(positions)]
        for it in range(1, ITERATIONS+1):
            g = gradient(positions)
            positions = positions - ETA*g
            positions = project_feasible(positions)
            L = objective(positions)
            loss_log.append(L)

            if it % FRAME_STEP == 0:
                frames.append(positions.copy())
                # disegno frame
                ax.clear()
                ax.set_xlim(0,SQUARE_SIZE); ax.set_ylim(0,SQUARE_SIZE)
                ax.set_aspect('equal')
                ax.add_patch(plt.Rectangle((0,0),SQUARE_SIZE,SQUARE_SIZE,fill=False))
                for p in positions:
                    ax.add_patch(plt.Circle(p,RADIUS,fill=True,alpha=0.6))
                ax.text(0.02,0.98,f"Cerchi: {n_circles}\nIter: {it}\nLoss: {L:.4f}",
                        fontsize=12,color="black",va="top",
                        bbox=dict(facecolor="white",alpha=0.7,edgecolor="none"))
                canvas.draw()
                buf = canvas.buffer_rgba()
                h,w = canvas.get_width_height()[::-1]
                img = np.frombuffer(buf,dtype=np.uint8).reshape((h,w,4))
                writer.append_data(img[:,:,:3])

            if it > 5 and abs(loss_log[-2] - loss_log[-1]) < TOL:
                break

        # aggiungo un nuovo cerchio solo se non ho raggiunto il massimo
        if n_circles < max_circles:
            positions = add_circle(positions)
            positions = project_feasible(positions)

    writer.close()
    plt.close(fig)

    # immagine finale
    fig2, ax2 = plt.subplots(figsize=(6,6))
    ax2.set_xlim(0,SQUARE_SIZE); ax2.set_ylim(0,SQUARE_SIZE)
    ax2.set_aspect('equal')
    ax2.set_title(f"Final GD Packing - {max_circles} cerchi")
    ax2.add_patch(plt.Rectangle((0,0),SQUARE_SIZE,SQUARE_SIZE,fill=False))
    for p in positions:
        ax2.add_patch(plt.Circle(p,RADIUS,fill=True,alpha=0.6))
    plt.savefig("final.png", dpi=300)
    plt.close(fig2)

    return positions

if __name__ == "__main__":
    run_gd_session(25)
