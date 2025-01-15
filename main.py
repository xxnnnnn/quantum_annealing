from algorithm import *
from constraint_construct import construct_ppsp_instance
import matplotlib.pyplot as plt


def compare_iterative_tran_algorithm(N_values,iterations=5,epsilon=0.1,num_trials=50): # 比较trans算法中纯mT和迭代m*T以及经典算法的区别
    classical_average_results_pure = []
    iterative_tran_average_results = []
    total_tran_average_results = []


    for N in N_values:
        classical_satisfied_ratio = 0
        iterative_tran_satisfied_ratio = 0
        total_tran_satisfied_ratio = 0


        for _ in range(num_trials):
            print('N =', N, '_ =', _)
            # Construct problem instance
            constraints, planted_solution = construct_ppsp_instance(N, epsilon)

            # Solve using pure classical SA as a benchmark
            classical_solution, classical_satisfied = simulated_annealing_solution(N, constraints, max_iterations=100)
            classical_satisfied_ratio += classical_satisfied / classical_satisfied

            iterative_tran_solution, iterative_tran_satisfied,_ = iterative_annealing_tran_solution(N,planted_solution,constraints,iterations)
            iterative_tran_satisfied_ratio += iterative_tran_satisfied / classical_satisfied

            total_tran_solution, total_tran_satisfied = total_annealing_tran_solution(N,planted_solution,constraints,iterations)
            total_tran_satisfied_ratio += total_tran_satisfied / classical_satisfied



        # Calculate average satisfied constraints for both methods
        classical_average_results_pure.append(classical_satisfied_ratio / num_trials)
        iterative_tran_average_results.append(iterative_tran_satisfied_ratio / num_trials)
        total_tran_average_results.append(total_tran_satisfied_ratio / num_trials)


    # Plotting the comparison
    plt.figure(figsize=(10, 6))

    plt.plot(N_values, classical_average_results_pure, label = 'Classical Simulating Annealing', marker = 's', linestyle = '--')
    plt.plot(N_values, iterative_tran_average_results, label='Iterative Tran Annealing', marker='x', linestyle='--', color='y')
    plt.plot(N_values, total_tran_average_results, label = 'Total Tran Annealing', marker = '^', linestyle = '-')
    plt.xlabel('Number of Variables (N)')
    plt.ylabel('Average Ratio of Satisfied Constraints')
    plt.title('Comparison of Forward and Classical Initial Solutions for Reverse Annealing')
    plt.legend(loc = 'best')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    N_values = list(range(4, 13))
    compare_iterative_tran_algorithm(N_values,iterations=5,epsilon=0.1,num_trials=50)

