import numpy as np
from qutip import *



#%%
def Z(n, i):
    return tensor([sigmaz() if k == i else qeye(2) for k in range(n)])

def Y(n, i):
    return tensor([sigmay() if k == i else qeye(2) for k in range(n)])

def X(n, i):
    return tensor([sigmax() if k == i else qeye(2) for k in range(n)])

# define the evolution functions f(t), g(t), and h(t).
def f(t,args):
    t_f = args['t_f']
    return (1 - t / t_f) ** 0.5

def g(t,args):
    t_f = args['t_f']
    return (t / t_f) ** 0.5

def h(t,args):
    t_f = args['t_f']
    return 4 * np.sqrt((1 - t / t_f) * (t / t_f))

#%%
# define hamiltonian functions
def Hp_func(N, constraints):
    Hp = sum([-V_ijk * Z(N, i) * Z(N, j) * Z(N, k) for (i, j, k, V_ijk) in constraints])
    return Hp

def Hm_func(N):
    Hm = sum([-X(N, i) for i in range(N)])
    return Hm

def Href_func(N, string_seed):
    Href = sum([-int(string_seed[i]) * Z(N, i) for i in range(N)])
    return Href

def HST_func(N, string_seed, t):
    alpha = 0.6 * np.log(N)
    omega = 2 * np.pi * 6 * np.log(N)

    # Construct the phases based upon the string_seed.
    phases = [0 if int(string_seed[i]) == -1 else np.pi for i in range(N)]

    HST = sum([alpha * np.sin(omega * t + phases[j]) * Y(N, j) for j in range(N)])

    return HST


#%%
# define the evolution function

# Hp, Hm, Href are the Hamiltonians
def evolution_long(N, string_seed, tf, dt, constraints):
    tensor_dims = [[2]*N, [1]*N]
    times = np.arange(0, tf, dt)
    args = {'t_f': tf}
    Hp = Hp_func(N, constraints)
    Hm = Hm_func(N)
    Href = Href_func(N, string_seed)
    H = [[Hp, lambda t, args: g(t, args)], [Hm, lambda t, args: h(t, args)], [Href, lambda t, args: f(t, args)]]
    psi_0 = tensor([basis(2, 0) if int(string_seed[i]) == 1 else basis(2, 1) for i in range(N)])
    results = sesolve(H, psi_0, times, args = args)
    psi_times = results.states
    for psi in psi_times:
        psi.dims = tensor_dims
    return times, psi_times

# Hp,Hm,Hst are the Hamiltonians
def evolution_tran(N, string_seed, tf, dt, constraints,alpha=1, omega=1):

    static_operators = [alpha * Y(N, j) for j in range(N)]
    # Construct the phases based upon the string_seed.
    phases = [0 if int(string_seed[i]) == -1 else np.pi for i in range(N)]
    def coeff_j(t):
        phase = phases[j]
        return np.sin(omega * t + phase)

    tensor_dims = [[2]*N, [1]*N]
    times = np.arange(0, tf, dt)
    args = {'t_f': tf}
    Hp = Hp_func(N, constraints)
    Hm = Hm_func(N)
    H = [[Hp, lambda t, args: g(t, args)], [Hm, lambda t, args: f(t, args)]]
    for j in range(N):
        H.append([static_operators[j], lambda t, args: h(t, args) * coeff_j(t)])

    # Construct the initial state.
    psi_0 = tensor([(basis(2, 0) + basis(2, 1)).unit() for _  in range(N)])
    results = sesolve(H, psi_0, times, args = args)
    psi_times = results.states
    for psi in psi_times:
        psi.dims = tensor_dims
    return times, psi_times

# Hp,Hm are the Hamiltonians,the most classical case
def evolution(N, tf, dt, constraints):
    tensor_dims = [[2]*N, [1]*N]
    times = np.arange(0, tf, dt)
    args = {'t_f': tf}
    Hp = Hp_func(N, constraints)
    Hm = Hm_func(N)
    H = [[Hp, lambda t, args: g(t, args)], [Hm, lambda t, args: f(t, args)]]
    psi_0 = tensor([(basis(2, 0) + basis(2, 1)).unit() for _  in range(N)])
    results = sesolve(H, psi_0, times, args = args)
    psi_times = results.states
    for psi in psi_times:
        psi.dims = tensor_dims
    return times, psi_times

#%%
def get_probabilities(psi_times):
    probabilities = []
    for psi in psi_times:
        # 计算所有基态的概率
        state_vector = psi.full().flatten()
        probs = np.abs(state_vector) ** 2
        probabilities.append(probs)
    return np.array(probabilities)  # shape: (num_times, 2^N)


def do_measurement(psi, N):
    state_vector = psi.full().flatten()  # 获取态向量
    probabilities = np.abs(state_vector) ** 2  # 各基态的概率分布
    measured_index = np.random.choice(len(probabilities), p=probabilities)
    measured_seed = [1 if int(bit) == 0 else -1 for bit in f"{measured_index:0{N}b}"]
    return measured_seed

def ground_prob(H, psi, tol=1e-10):
    # Get the eigenvalues and eigenstates of H
    eigenvalues, eigenstates = H.eigenstates()

    # Find the ground energy and corresponding eigenstates
    ground_energy = np.min(eigenvalues)

    # Collect all degenerate ground states
    ground_states = [eigenstates[i] for i in range(len(eigenvalues)) if np.isclose(eigenvalues[i], ground_energy, atol=tol)]

    # Construct the projection operator onto the ground state subspace
    P_gs = sum([gs.proj() for gs in ground_states])

    # Probability of psi being in the ground state subspace
    prob = abs((psi.dag() * P_gs * psi))  # This is the correct projection calculation

    return prob