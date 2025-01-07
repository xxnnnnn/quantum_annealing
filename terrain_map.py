import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from constraint_construct import construct_ppsp_instance, count_satisfied_constraints
from algorithm import iterative_annealing_long_solution_for_plot, iterative_annealing_tran_solution_for_plot, iterative_sa_long_hybrid_solution_for_plot, iterative_sa_tran_hybrid_solution_for_plot

#%% Constants
PAD_WIDTH = 2  # Padding for the energy matrix

#%% Helper Functions
def gray_code(n):
    if n == 0:
        return ['']
    else:
        prev_codes = gray_code(n - 1)
        result = []
        for code in prev_codes:
            result.append('0' + code)
        for code in reversed(prev_codes):
            result.append('1' + code)
        return result

def create_gray_code_matrix(string_length):
    if string_length % 2 == 0:
        gray_codes = gray_code(string_length // 2)
        size = len(gray_codes)
        matrix = np.empty((size, size), dtype=object)
        for i, row_code in enumerate(gray_codes):
            for j, col_code in enumerate(gray_codes):
                matrix[i, j] = row_code + col_code
        return matrix
    else:
        gray_codes1 = gray_code(string_length // 2 + 1)
        gray_codes2 = gray_code(string_length // 2)
        size1 = len(gray_codes1)
        size2 = len(gray_codes2)
        matrix = np.empty((size1, size2), dtype=object)
        for i, row_code in enumerate(gray_codes1):
            for j, col_code in enumerate(gray_codes2):
                matrix[i, j] = row_code + col_code
        return matrix

def solution_to_bitstring(solution):
    """Converts a solution list (1, -1) to a bitstring."""
    return ''.join(['0' if x == 1 else '1' for x in solution])

def compute_energy(bitstring, constraints, planted_solution):
    """Computes the energy of a given bitstring."""
    bit_values = [1 if bit == '0' else -1 for bit in bitstring]
    c1 = count_satisfied_constraints(constraints, bit_values)
    c2 = count_satisfied_constraints(constraints, planted_solution)
    return c2 - 2 * c1

#%%
def plot_energy_landscape(matrix, n, constraints, planted_solution):
    size_row, size_col = matrix.shape
    energies = np.zeros((size_row, size_col))
    print("Constraints:", constraints)

    # 创建原始坐标网格，用于绘制离散能量点
    X_orig, Y_orig = np.meshgrid(np.arange(size_col), np.arange(size_row))

    # 计算能量值
    for i in range(size_row):
        for j in range(size_col):
            bitstring = matrix[i, j]
            energies[i, j] = compute_energy(bitstring, constraints, planted_solution)

    # 对能量矩阵进行填充，确保边缘平坦
    padded_energies = np.pad(
        energies,
        pad_width=PAD_WIDTH,
        mode='constant',
        constant_values=0
    )

    size_row_padded, size_col_padded = padded_energies.shape

    # 创建填充后的坐标网格
    X_padded, Y_padded = np.meshgrid(np.arange(size_col_padded), np.arange(size_row_padded))

    # 增加数据点密度，使用插值
    grid_x, grid_y = np.mgrid[0:size_col_padded - 1:200j, 0:size_row_padded - 1:200j]
    grid_energies = griddata(
        (X_padded.flatten(), Y_padded.flatten()),
        padded_energies.flatten(),
        (grid_x, grid_y),
        method='cubic'
    )

    # 对能量值进行高斯平滑
    grid_energies = gaussian_filter(grid_energies, sigma=2)  # 调整 sigma 以获得适当的平滑效果

    # 绘制平滑的能量地形表面
    surface = go.Surface(
        z=grid_energies.T,  # 转置以匹配 Plotly 的坐标系
        x=grid_x.T,
        y=grid_y.T,
        colorscale='Viridis',
        opacity=0.7,
        showscale=False,
        name='Smoothed Energy Surface'
    )

    # 绘制原始的离散能量点（需要位移坐标以匹配填充后的网格）
    scatter = go.Scatter3d(
        x=(X_orig.flatten() + PAD_WIDTH),
        y=(Y_orig.flatten() + PAD_WIDTH),
        z=energies.flatten(),
        mode='markers',
        marker=dict(
            size=4,
            color=energies.flatten(),
            colorscale='Viridis',
            opacity=0.8
        ),
        name='Discrete Energy Points'
    )

    # 突出显示植入解
    bitstring_planted = ''.join(['0' if x == 1 else '1' for x in planted_solution])
    position = np.argwhere(matrix == bitstring_planted)
    if position.size > 0:
        i, j = position[0]
        planted_energy = energies[i, j]
        x_planted = j + PAD_WIDTH
        y_planted = i + PAD_WIDTH
        scatter_planted = go.Scatter3d(
            x=[x_planted],
            y=[y_planted],
            z=[planted_energy],
            mode='markers',
            marker=dict(
                size=8,
                color='red',
                symbol='diamond'
            ),
            name='Planted Solution'
        )
        fig = go.Figure(data=[surface, scatter, scatter_planted])
    else:
        fig = go.Figure(data=[surface, scatter])

    # 更新布局
    fig.update_layout(
        title='3D Energy Landscape with Discrete Points and Smoothed Surface',
        scene=dict(
            xaxis_title='Column Index',
            yaxis_title='Row Index',
            zaxis_title='Energy',
            xaxis=dict(showgrid=True),
            yaxis=dict(showgrid=True),
            zaxis=dict(showgrid=True)
        ),
        width=1000,
        height=800
    )

    return fig, energies


#%% Evolution Trajectory Plot
def plot_evolution_trajectories(matrix, n, constraints, planted_solution, algorithm_solutions):
    fig, energies = plot_energy_landscape(matrix, n, constraints, planted_solution)

    # 定义算法名称
    algorithm_names = [
        "Iterative Annealing Long",
        "Iterative Annealing Tran",
        "Iterative SA Long Hybrid",
        "Iterative SA Tran Hybrid"
    ]

    # 遍历所有算法的解轨迹
    for algo_index, (solution, _, trajectories) in enumerate(algorithm_solutions):
        # 将解轨迹转换为二进制字符串以便于查找
        trajectory_bits = [solution_to_bitstring(sol) for sol in trajectories]

        # 获取每个解在能量矩阵中的位置
        trajectory_positions = [np.argwhere(matrix == bitstring)[0] for bitstring in trajectory_bits]

        # 提取对应的 x, y, z 坐标
        x_coords = [pos[1] + PAD_WIDTH for pos in trajectory_positions]
        y_coords = [pos[0] + PAD_WIDTH for pos in trajectory_positions]
        z_coords = [compute_energy(bitstring, constraints, planted_solution) for bitstring in trajectory_bits]

        # 绘制每个算法的轨迹
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            marker=dict(size=5),
            line = dict(width=10),
            name=algorithm_names[algo_index]  # 使用算法名称
        ))

    # 更新布局以适应新添加的轨迹
    fig.update_layout(
        title='3D Energy Landscape with Evolution Trajectories',
    )
    fig.show()
    return fig





#%% Main Function
def main():
    N = 4  # Bitstring length
    constraints, planted_solution = construct_ppsp_instance(N)
    matrix = create_gray_code_matrix(N)

    # Verify planted solution satisfies constraints
    c2 = count_satisfied_constraints(constraints, planted_solution)
    print("Number of constraints satisfied by planted solution:", c2)

    # Generate solutions for each algorithm
    algo_solutions = [
        iterative_annealing_long_solution_for_plot(N, planted_solution, constraints, iterations=5),
        iterative_annealing_tran_solution_for_plot(N, planted_solution, constraints, iterations=5),
        iterative_sa_long_hybrid_solution_for_plot(N, planted_solution, constraints, iterations=5),
        iterative_sa_tran_hybrid_solution_for_plot(N, planted_solution, constraints, iterations=5)
    ]
    print(f"algo{[len(i[2]) for i in algo_solutions]}")

    # Plot evolution trajectories for all algorithms
    plot_evolution_trajectories(matrix, N, constraints, planted_solution, algo_solutions)

if __name__ == '__main__':
    main()
