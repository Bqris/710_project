# Simulate an ising model with given paramaters

from __future__ import annotations 
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation

def lattice_initial(l): 
    return np.random.choice([1, -1], size = (l, l))

def mmc_step(lattice, Beta, l, J = 1):
    """One single-spin Metropolis attempt. Returns the accepted energy change."""
    i = np.random.randint(0,l) 
    j = np.random.randint(0,l)

    neighbor_top = lattice[i, (j+1) % l]
    neighbor_bottom = lattice[i, (j-1)%l]
    neighbor_right = lattice[(i+1)%l, j] 
    neighbor_left = lattice[(i-1)%l, j]
    spin_neighbors = neighbor_right + neighbor_top \
            + neighbor_left + neighbor_bottom

    delta_energy = 2*lattice[i,j]*J*spin_neighbors

    if delta_energy <= 0 or np.random.random() < np.exp(-Beta * delta_energy):
        lattice[i,j] = -lattice[i,j]
        return delta_energy
    return 0

def mmc_evolution(lattice, Beta, l, steps, steps_per_frame = None, interval = 50):
    """Run the Metropolis evolution as a live animation.

    `steps` is the total number of single-spin attempts; each animation frame
    advances `steps_per_frame` of them (default: one sweep, l*l attempts).
    """
    if steps_per_frame is None:
        steps_per_frame = l * l
    frames = max(1, steps // steps_per_frame)

    energies = []
    configuration_energies = []
    energy = 0

    fig, ax = plt.subplots()
    image = ax.imshow(lattice, cmap = "coolwarm", vmin = -1, vmax = 1,
                      interpolation = "nearest")
    title = ax.set_title("", fontsize = 18)
    ax.set_xticks([])
    ax.set_yticks([])

    def update(frame):
        nonlocal energy
        for _ in range(steps_per_frame):
            delta_energy = mmc_step(lattice, Beta, l)
            if delta_energy:
                energy += delta_energy
                energies.append(energy)

        configuration_energy = energy_of(lattice, l)
        configuration_energies.append(configuration_energy)

        # imshow keeps a reference to the array, but set_data is what marks it stale
        image.set_data(lattice)
        title.set_text(f"sweep {frame + 1}/{frames}   E = {configuration_energy}")
        return image, title

    # Hold on to the animation object: letting it be garbage collected stops the loop.
    animation = FuncAnimation(fig, update, frames = frames, interval = interval,
                              repeat = False, blit = False)
    plt.show()
    return animation, energies, configuration_energies

def energy_of(lattice, l):
    energy = 0
    for i in range(l):
        for j in range(l): 
            spin_site = lattice[i,j] 
            sum_neighbors = lattice[(i+1)%l, j] + lattice[(i - 1)%l , j] \
                    + lattice[i, (j+1)%l] + lattice[i, (j-1)%l]
            energy += -spin_site * sum_neighbors
    return energy // 2   # every pair was counted twice


if __name__ == "__main__": 
    beta = 0.5
    l = 100
    steps = 2000 * l * l
    initial = lattice_initial(l)
    mmc_evolution(initial, beta, l, steps)
