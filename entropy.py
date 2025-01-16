from algorithm import iterative_annealing_tran_solution, total_annealing_tran_solution
from constraint_construct import construct_ppsp_instance
import matplotlib.pyplot as plt

def entropy_plot(N_list, iterations, num_trial):

    for N in N_list:
        total_entropies = []
        iterative_entropies = []

        for num_trial in range(num_trial):
            constraints, planted_solution = construct_ppsp_instance(N)
            _, _, total_entropy_list = total_annealing_tran_solution(N, planted_solution, constraints, iterations)
            _, _, _, iterative_entropy_list = iterative_annealing_tran_solution(N, planted_solution, constraints, iterations)

            total_entropies.append(total_entropy_list)
            iterative_entropies.append(iterative_entropy_list)
            print(f'Finished trial {num_trial + 1} for N={N}')

        # Calculate average entropy across trials
        avg_total_entropy = [sum(trial) / len(trial) for trial in zip(*total_entropies)]
        avg_iterative_entropy = [sum(trial) / len(trial) for trial in zip(*iterative_entropies)]

        # Plot for current N with different line styles and labels
        plt.plot(range(1 + iterations), avg_total_entropy, label=f'Total Tran Annealing (N={N})', marker='s', linestyle='--')
        plt.plot(range(1 + iterations), avg_iterative_entropy, label=f'Iterative Tran Annealing (N={N})', marker='x', linestyle='--', color='y')

    # Configure and display plot after iterating through all N values
    plt.xlabel('Iterations')
    plt.ylabel('Entropy (Average over Trials)')
    plt.title(f'Average Entropy of Iterative and Total Tran Annealing (N={", ".join(map(str, N_list))})')  # Updated title with N list
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
  N_list = list(range(4,6))
  entropy_plot(N_list, 6, 20)