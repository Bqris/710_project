import pennylane as qp 
import numpy as np 
import matplotlib.pyplot as plt 

def ising_circuit(n, layers: np.ndarray, J: dict, h: np.ndarray, g: np.ndarray, dt: float):
    for layer in layers: 
        for i, j in layer: 
            qp.IsingZZ(2*J[(i,j)]*dt, wires=[i,j])

    for w in range(n*n): qp.RZ(2*h*dt, w) 
    for w in range(n*n): qp.RX(2*g*dt, w) 

n = 3
num_wires = n**2
example_layers = np.array(
        [[(0,1),(3,4),(6,7)],
         [(1,2),(4,5),(7,8)],
         [(0,3),(1,4),(2,5)],
         [(3,6),(4,7),(5,8)]]
)

example_J = {
        (0,1): 0.1,
        (0,3): 0.3,
        (1,2): 1.2,
        (1,4): 1.4, 
        (2,5): 2.5,
        (3,4): 3.4,
        (3,6): 3.6,
        (4,5): 4.5,
        (4,7): 4.7,
        (5,8): 5.8,
        (6,7): 6.7,
        (7,8): 7.8,
    }

example_h = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
example_g = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

dt = 0.1

dev = qp.device('default.qubit', num_wires)

@qp.qnode(dev)
def circuit(n, layers: np.ndarray, J: dict, h: np.ndarray, g: np.ndarray, dt: float): 
    ising_circuit(n, layers, J, h, g, dt)

fig, ax = qp.draw_mpl(circuit)(n, example_layers, example_J, example_g, example_h, dt)
plt.show()

#print(qp.draw(circuit)(n, example_layers, example_J, example_g, example_h, dt))

