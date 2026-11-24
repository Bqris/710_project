Classical-Quantum Hybrid-algorithm for solving inverse heat problem for source identification

Under suitable boundary condition, heat equation admits a unique source distribution for a given boundary data. 
For a given geometry one can assemble system matrix that encodes geometry and boundary information to a single
matrix, inverting that matrix amounts to approximating Green's function $G_R$ associated with that IBV problem. 

Given the boundary flux measurements, our goal is to minimize $|| G_R Q - u_{measured} ||^2 $, this objective
is equivalent with minimizing Ising Hamiltonian $H(\sigma)$, where $\sigma$ corresponds a particular source
distribution. 

According to adiabatic theorem, if one starts with ground state of some Hamiltonian, and Hamiltonian is slightly
changed to another Hamiltonian, state evolve to ground state of the new Hamiltonian. Using quantum approximate
optimization algorithm, we use mixer and cost hamiltonians and drive the system according to 
$tH_M(\sigma) + (1-t)H_C(\sigma)$.  We will use a parametrized circuit to make this happen and finally sample 
the optimized circuit to find the optimal source distribution. 

