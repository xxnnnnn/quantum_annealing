import numpy as np
import matplotlib.pyplot as plt
from abstract_definition import evolution_tran, do_measurement
from constraint_construct import construct_ppsp_instance, count_satisfied_constraints

def parameter_heatmap(n_list, discrete_num, num_trials=10):
    num_subplots = len(n_list)
    cols = min(3, num_subplots)  # 每行最多放 3 张子图
    rows = (num_subplots + cols - 1) // cols

    fig_width = 6 * cols
    fig_height = 6 * rows
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))
    axes = axes.flatten()

    alpha_ini = 0.01
    alpha_final = 1
    f_ini = 0.01
    f_final = 1
    for idx, (N, ax) in enumerate(zip(n_list, axes)):
        # 定义参数范围
        alpha_values = np.linspace(alpha_ini * np.log(N), alpha_final * np.log(N), discrete_num)
        omega_values = np.linspace(f_ini * 2 * np.pi * np.log(N), f_final * 2 * np.pi * np.log(N), discrete_num)

        # 初始化结果矩阵
        results = np.zeros((discrete_num, discrete_num))
        constraints, string_seed = construct_ppsp_instance(N)

        dt = 0.1
        tf = N

        # 填充结果矩阵
        # 填充结果矩阵
        for i, alpha in enumerate(alpha_values):
            for j, omega in enumerate(omega_values):
                # 累积满意约束数量以计算平均
                total_satisfied = 0
                for trial in range(num_trials):
                    _, psi_times, _ = evolution_tran(N, string_seed, tf, dt, constraints, alpha, omega)
                    measured_seed = do_measurement(psi_times[-1], N)
                    satisfied_constraints = count_satisfied_constraints(constraints, measured_seed)
                    total_satisfied += satisfied_constraints
                    print(f"Size {N}, alpha = {alpha}, omega = {omega}, trial {trial + 1}, satisfied constraints = {satisfied_constraints}")

                # 计算平均值并存入矩阵
                results[i, j] = total_satisfied / num_trials
    # 画热力图，强制正方形显示
        im = ax.imshow(
            results,
            extent=(0, discrete_num, 0, discrete_num),  # 保证正方形网格
            origin='lower',
            aspect='equal',  # 强制比例一致
            cmap='viridis'
        )

        # 显示实际刻度值
        ax.set_xticks(np.linspace(0, discrete_num - 1, 5))  # 在离散点中选 5 个显示
        ax.set_yticks(np.linspace(0, discrete_num - 1, 5))
        ax.set_xticklabels([f'{omega_values[int(x)]:.1f}' for x in np.linspace(0, discrete_num - 1, 5)])
        ax.set_yticklabels([f'{alpha_values[int(y)]:.1f}' for y in np.linspace(0, discrete_num - 1, 5)])

        ax.set_title(f'Size {N}', fontsize=12)
        ax.set_xlabel('ω (Frequency)', fontsize=10)
        ax.set_ylabel('α (Drive Strength)', fontsize=10)

        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, orientation='vertical', shrink=0.8)
        cbar.set_label('Satisfied Constraints', fontsize=10)

    # 删除多余子图
    for ax in axes[len(n_list):]:
        fig.delaxes(ax)

    plt.tight_layout()
    plt.show()


def main():
    n_list = list(range(4,6))
    print(n_list)
    discrete_num = 10
    num_trials = 10
    parameter_heatmap(n_list, discrete_num, num_trials)

if __name__ == "__main__":
    main()
