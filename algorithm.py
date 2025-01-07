from abstract_definition import *
from constraint_construct import *
import math

#%%
# classical Algorithm:
def simulated_annealing_solution(N, hypergraph, max_iterations, initial_temperature=100, cooling_rate=0.99):
    # Initialize the current solution and best solution
    current_solution = [random.choice([1, -1]) for _ in range(N)]
    current_satisfied = count_satisfied_constraints(hypergraph, current_solution)
    max_satisfied = current_satisfied
    best_solution = current_solution

    # Set the initial temperature
    temperature = initial_temperature

    # Simulated annealing loop
    for _ in range(max_iterations):
        # Generate a new solution by flipping a random variable
        new_solution = current_solution[:]
        flip_index = random.randint(0, N - 1)
        new_solution[flip_index] *= -1


        # Calculate the number of satisfied constraints for the new solution
        new_satisfied = count_satisfied_constraints(hypergraph, new_solution)

        # Calculate the change in the number of satisfied constraints
        delta = new_satisfied - current_satisfied

        # Decide whether to accept the new solution
        if delta > 0 or random.uniform(0, 1) < math.exp(delta / temperature):
            current_solution = new_solution
            current_satisfied = new_satisfied

            # Update the best solution if the new solution is better
            if current_satisfied > max_satisfied:
                best_solution = current_solution
                max_satisfied = current_satisfied

        # Cool down the temperature
        temperature *= cooling_rate

    return best_solution, max_satisfied


def greedy_solution(N, hypergraph):
    # Initialize the current solution randomly
    current_solution = [random.choice([1, -1]) for _ in range(N)]

    max_satisfied = count_satisfied_constraints(hypergraph, current_solution)
    best_solution = current_solution[:]
    # Flag to check if any improvement was made in the current iteration
    improvement = True

    while improvement:
        improvement = False
        # Iterate over each variable
        for i in range(N):
            # Flip the value of the current variable
            current_solution[i] *= -1
            # Count the number of satisfied constraints after the flip
            current_satisfied = count_satisfied_constraints(hypergraph, current_solution)
            # If the number of satisfied constraints increased, keep the change
            if current_satisfied > max_satisfied:
                max_satisfied = current_satisfied
                best_solution = current_solution[:]
                improvement = True
            else:
                # Revert the flip if no improvement
                current_solution[i] *= -1

    return best_solution, max_satisfied


def simulated_annealing_for_hybrid_solution(N, hypergraph, planted_solution, max_iterations, initial_temperature=100, cooling_rate=0.99):
    # Initialize the current solution and best solution
    current_solution = planted_solution[:]
    best_solution = current_solution[:]
    current_satisfied = count_satisfied_constraints(hypergraph, current_solution)
    max_satisfied = current_satisfied

    # Set the initial temperature
    temperature = initial_temperature

    # Simulated annealing loop
    for _ in range(max_iterations):
        # Generate a new solution by flipping a random variable
        new_solution = current_solution[:]
        flip_index = random.randint(0, N - 1)
        new_solution[flip_index] *= -1


        # Calculate the number of satisfied constraints for the new solution
        new_satisfied = count_satisfied_constraints(hypergraph, new_solution)

        # Calculate the change in the number of satisfied constraints
        delta = new_satisfied - current_satisfied

        # Decide whether to accept the new solution
        if delta > 0 or random.uniform(0, 1) < math.exp(delta / temperature):
            current_solution = new_solution
            current_satisfied = new_satisfied

            # Update the best solution if the new solution is better
            if current_satisfied > max_satisfied:
                best_solution = current_solution
                max_satisfied = current_satisfied

        # Cool down the temperature
        temperature *= cooling_rate

    return best_solution, max_satisfied


#%%
def forward_annealing_solution(N, constraints, steps=100):
    t_f = N
    dt = t_f / steps
    times, psi_times = evolution(N, t_f, dt, constraints)
    forward_solution = do_measurement(psi_times[-1], N)
    forward_satisfied = count_satisfied_constraints(constraints, forward_solution)

    return forward_solution, forward_satisfied

def reverse_annealing_solution(N,string_seed,constraints,steps=100):
    t_f = N
    dt =t_f / steps
    times, psi_times = evolution_long(N, string_seed, t_f, dt, constraints)
    reverse_solution = do_measurement(psi_times[-1], N)
    reverse_satisfied = count_satisfied_constraints(constraints, reverse_solution)

    return reverse_solution, reverse_satisfied

#%%
def total_annealing_tran_solution(N, string_seed, constraints, iterations):
    tf = iterations * N
    omega = 2 * np.pi * 6 * np.log(N)
    dt = 0.4 / omega

    times, psi_times = evolution_tran(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)


    total_solution_tran = measured_seed
    total_satisfied_tran = count_satisfied_constraints(constraints, total_solution_tran)
    return total_solution_tran, total_satisfied_tran


#%%
def iterative_annealing_tran_solution(N, string_seed, constraints, iterations):
    tf = N
    omega = 2 * np.pi * 6 * np.log(N)
    dt = 0.4 / omega

    times, psi_times = evolution_tran(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)


    if iterations > 1:
        return iterative_annealing_tran_solution(N, measured_seed, constraints, iterations - 1)
    else:
        iterative_solution_tran = measured_seed
        iterative_satisfied_tran = count_satisfied_constraints(constraints, iterative_solution_tran)
        return iterative_solution_tran, iterative_satisfied_tran


def iterative_annealing_long_solution(N, string_seed, constraints, iterations, steps=100):
    tf = N
    dt = tf / steps

    times, psi_times = evolution_long(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)
    print('measured_seed =', measured_seed)

    if iterations > 1:
        return iterative_annealing_long_solution(N, measured_seed, constraints, iterations - 1)
    else:
        iterative_solution_long = measured_seed
        iterative_satisfied_long = count_satisfied_constraints(constraints, iterative_solution_long)
        return iterative_solution_long, iterative_satisfied_long


#%%
def iterative_sa_tran_hybrid_solution(N, string_seed, constraints, iterations, satisfied=0):
    tf = N
    omega = 2 * np.pi * 6 * np.log(N)
    dt = 0.4 / omega


    times, psi_times = evolution_tran(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)
    measured_satisfied = count_satisfied_constraints(constraints, measured_seed)
    print('measured_seed =', measured_seed)
    print('measured_satisfied =', measured_satisfied)

    if measured_satisfied >= satisfied:
        string_seed = measured_seed


    best_solution, satisfied = simulated_annealing_for_hybrid_solution(N, constraints, string_seed, iterations)
    string_seed = best_solution
    print('Best solution =', best_solution)
    print('Satisfied =', satisfied)


    if iterations > 1:
        return iterative_sa_tran_hybrid_solution(N, string_seed, constraints, iterations - 1, satisfied)
    else:
        reverse_solution_tran_hybrid = string_seed
        reverse_satisfied_tran_hybrid = satisfied
        return reverse_solution_tran_hybrid, reverse_satisfied_tran_hybrid


def iterative_sa_long_hybrid_solution(N, string_seed, constraints, iterations,satisfied=0, steps=100):
    tf = N
    dt = tf / steps

    times, psi_times = evolution_long(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)
    measured_satisfied = count_satisfied_constraints(constraints, measured_seed)
    print('measured_seed =', measured_seed)
    print('measured_satisfied =', measured_satisfied)

    if measured_satisfied >= satisfied:
        string_seed = measured_seed


    best_solution, satisfied = simulated_annealing_for_hybrid_solution(N, constraints, string_seed, iterations)
    string_seed = best_solution
    print('Best solution =', best_solution)
    print('Satisfied =', satisfied)

    if iterations > 1:
        return iterative_sa_long_hybrid_solution(N, string_seed, constraints, iterations - 1, satisfied)
    else:
        reverse_solution_long_hybrid = string_seed
        reverse_satisfied_long_hybrid = satisfied
        return reverse_solution_long_hybrid, reverse_satisfied_long_hybrid

#%%
def iterative_annealing_tran_solution_for_plot(N, string_seed, constraints, iterations):
    tf = N
    omega = 2 * np.pi * 6 * np.log(N)
    dt = 0.4 / omega

    # 用于记录每次迭代的解
    solutions = []

    times, psi_times = evolution_tran(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)
    solutions.append(measured_seed)  # 记录解
    print('measured_seed =', measured_seed)

    if iterations > 1:
        next_solution, next_satisfied, next_solutions = iterative_annealing_tran_solution_for_plot(N, measured_seed, constraints, iterations - 1)
        solutions.extend(next_solutions)  # 合并解的记录
        return next_solution, next_satisfied, solutions
    else:
        iterative_solution_tran = measured_seed
        iterative_satisfied_tran = count_satisfied_constraints(constraints, iterative_solution_tran)
        return iterative_solution_tran, iterative_satisfied_tran, solutions


def iterative_annealing_long_solution_for_plot(N, string_seed, constraints, iterations, steps=100):
    tf = N
    dt = tf / steps

    # 用于记录每次迭代的解
    solutions = []

    times, psi_times = evolution_long(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)
    solutions.append(measured_seed)  # 记录解
    print('measured_seed =', measured_seed)

    if iterations > 1:
        next_solution, next_satisfied, next_solutions = iterative_annealing_long_solution_for_plot(N, measured_seed, constraints, iterations - 1, steps)
        solutions.extend(next_solutions)  # 合并解的记录
        return next_solution, next_satisfied, solutions
    else:
        iterative_solution_long = measured_seed
        iterative_satisfied_long = count_satisfied_constraints(constraints, iterative_solution_long)
        return iterative_solution_long, iterative_satisfied_long, solutions


def iterative_sa_tran_hybrid_solution_for_plot(N, string_seed, constraints, iterations, satisfied=0):
    tf = N
    omega = 2 * np.pi * 6 * np.log(N)
    dt = 0.4 / omega

    # 用于记录每次迭代的解
    solutions = []

    times, psi_times = evolution_tran(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)
    measured_satisfied = count_satisfied_constraints(constraints, measured_seed)
    print('measured_seed =', measured_seed)
    print('measured_satisfied =', measured_satisfied)

    if measured_satisfied >= satisfied:
        string_seed = measured_seed

    best_solution, satisfied = simulated_annealing_for_hybrid_solution(N, constraints, string_seed, iterations)
    string_seed = best_solution
    solutions.append(string_seed)  # 记录解
    print('Best solution =', best_solution)
    print('Satisfied =', satisfied)

    if iterations > 1:
        next_solution, next_satisfied, next_solutions = iterative_sa_tran_hybrid_solution_for_plot(N, string_seed, constraints, iterations - 1, satisfied)
        solutions.extend(next_solutions)  # 合并解的记录
        return next_solution, next_satisfied, solutions
    else:
        reverse_solution_tran_hybrid = string_seed
        reverse_satisfied_tran_hybrid = satisfied
        return reverse_solution_tran_hybrid, reverse_satisfied_tran_hybrid, solutions


def iterative_sa_long_hybrid_solution_for_plot(N, string_seed, constraints, iterations, satisfied=0, steps=100):
    tf = N
    dt = tf / steps

    # 用于记录每次迭代的解
    solutions = []

    times, psi_times = evolution_long(N, string_seed, tf, dt, constraints)
    measured_seed = do_measurement(psi_times[-1], N)
    measured_satisfied = count_satisfied_constraints(constraints, measured_seed)
    print('measured_seed =', measured_seed)
    print('measured_satisfied =', measured_satisfied)

    if measured_satisfied >= satisfied:
        string_seed = measured_seed

    best_solution, satisfied = simulated_annealing_for_hybrid_solution(N, constraints, string_seed, iterations)
    string_seed = best_solution
    solutions.append(string_seed)  # 记录解
    print('Best solution =', best_solution)
    print('Satisfied =', satisfied)

    if iterations > 1:
        next_solution, next_satisfied, next_solutions = iterative_sa_long_hybrid_solution_for_plot(N, string_seed, constraints, iterations - 1, satisfied, steps)
        solutions.extend(next_solutions)  # 合并解的记录
        return next_solution, next_satisfied, solutions
    else:
        reverse_solution_long_hybrid = string_seed
        reverse_satisfied_long_hybrid = satisfied
        return reverse_solution_long_hybrid, reverse_satisfied_long_hybrid, solutions

