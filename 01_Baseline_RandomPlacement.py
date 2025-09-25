import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation, patches

# Parametri del problema
L = 1.0            # lato del quadrato
r = 0.1            # raggio dei cerchi
max_iter = 2000    # numero massimo di tentativi

# Strutture dati
centers = []       # lista dei centri dei cerchi posizionati
snapshots = []     # snapshot dei centri dopo ogni inserimento riuscito

# Algoritmo "stupido" di posizionamento casuale
for _ in range(max_iter):
    x = np.random.uniform(r, L - r)
    y = np.random.uniform(r, L - r)
    # verifica vincolo di non sovrapposizione
    if all((x - xi)**2 + (y - yi)**2 >= (2*r)**2 for xi, yi in centers):
        centers.append((x, y))
        snapshots.append(list(centers))

# Creazione dell'animazione
fig, ax = plt.subplots()
ax.set_xlim(0, L)
ax.set_ylim(0, L)
ax.set_aspect('equal')
ax.set_title("Packing casuale di cerchi congruenti")
circles = []

def init():
    return []

def animate(i):
    # rimuove i cerchi precedenti
    for c in circles:
        c.remove()
    circles.clear()
    # disegna lo stato corrente
    for (x, y) in snapshots[i]:
        c = patches.Circle((x, y), radius=r, fill=True, alpha=0.6)
        ax.add_patch(c)
        circles.append(c)
    # annota il numero di cerchi in alto a sinistra
    ax.text(
        0.02, 0.98,
        f"Count: {len(snapshots[i])}",
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor='none')
    )
    return circles

anim = animation.FuncAnimation(
    fig, animate,
    frames=len(snapshots),
    init_func=init,
    interval=200,
    blit=True
)

# Salva l'animazione come MP4
from matplotlib.animation import FFMpegWriter
writer = FFMpegWriter(fps=5, bitrate=1800)
anim.save('RandomAnimation.mp4', writer=writer)

# --- Generazione immagine finale ---
fig2, ax2 = plt.subplots(figsize=(6, 6))
ax2.set_xlim(0, L)
ax2.set_ylim(0, L)
ax2.set_aspect('equal')
ax2.set_title(f"Random - {len(centers)} cerchi")

# Aggiunge rettangolo del bordo
ax2.add_patch(plt.Rectangle((0, 0), L, L, fill=False, edgecolor='black'))

# Disegna tutti i cerchi finali
for (x, y) in centers:
    circle = plt.Circle((x, y), radius=r, fill=True, alpha=0.6)
    ax2.add_patch(circle)

# Salva come PNG
plt.savefig("Random_FinalPacking.png", dpi=300)
plt.close(fig2)
print(f"Immagine finale salvata in 'FinalPacking.png' con {len(centers)} cerchi.")