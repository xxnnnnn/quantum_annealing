import random

def construct_ppsp_instance(N,proportion=2,epsilon=0.1):
    # Number of constraints (NC >> N)
    NC = int(proportion * N)  # Arbitrary value ensuring NC >> N

    # Generate a random planted solution bitstring G
    planted_solution = [random.choice([1, -1]) for _ in range(N)]

    # Generate the hypergraph with NC constraints
    hypergraph = []
    for _ in range(NC):
        # Randomly pick 3 different variables to form a constraint
        i, j, k = random.sample(range(N), 3)
        # Randomly assign the sign of the constraint
        V_ijk = random.choice([1, -1])
        hypergraph.append((i, j, k, V_ijk))

    # Select a fraction (1 - epsilon) of constraints to be satisfied by the planted solution
    num_satisfied_constraints = int((1 - epsilon) * NC)
    satisfied_indices = random.sample(range(NC), num_satisfied_constraints)

    # Adjust the selected constraints to be satisfied by the planted solution
    for idx in satisfied_indices:
        i, j, k, _ = hypergraph[idx]
        V_ijk = planted_solution[i] * planted_solution[j] * planted_solution[k]
        hypergraph[idx] = (i, j, k, V_ijk)
    return hypergraph, planted_solution


def count_satisfied_constraints(hypergraph, solution):
    count = 0
    for i, j, k, V_ijk in hypergraph:
        if solution[i] * solution[j] * solution[k] == V_ijk:
            count += 1
    return count