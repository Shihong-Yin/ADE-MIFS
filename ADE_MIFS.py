import time
import numpy as np
import env_utils
from environment import Environment
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
from matplotlib.patches import Circle, Ellipse, Arrow


def evaluate_model(env, policy_weight=None, step_count=150):
    """根据权重向量评估 MIFS 策略的适应度"""
    if policy_weight is None:
        policy_weight = np.tile([0.12275435, 1.0, 0.55398284, 0.44272054],
                                (env.num_agents, 1))
    else:
        policy_weight = np.tile(policy_weight, (env.num_agents, 1))

    # 初始化环境并获取初始观测
    obs_n = env.reset(initialization=False, label='ifs')

    # 记录轨迹历史（用于可视化）
    positions_history = [(np.copy(env.agent_positions),
                          np.copy(env.agent_directions),
                          np.copy(env.goal_positions),
                          np.copy(env.obstacle_positions))]

    for t in range(step_count):
        actions = env.calculate_action(obs_n, t, policy_weight=policy_weight)
        obs_n, rew_n, dones_n = env.step(np.array(actions))
        positions_history.append((np.copy(env.agent_positions),
                                  np.copy(env.agent_directions),
                                  np.copy(env.goal_positions),
                                  np.copy(env.obstacle_positions)))
        # 所有 USV 任务结束则提前终止
        if all(dones_n):
            break

    # 统计碰撞信息
    agent_collision, obstacle_collision, total_step = env.get_collision_statistics()

    # 计算总行驶距离
    distance_traveled = np.sum(
        np.linalg.norm(
            np.diff(np.array([pos[0] for pos in positions_history]), axis=0),
            axis=2
        )
    )

    # 计算未到达目的地的 USV 数量
    num_unfinished = len(dones_n) - sum(dones_n)

    # 适应度（要最小化）
    fitness = distance_traveled / num_agents + \
              (agent_collision + obstacle_collision + num_unfinished) * 1000
    return fitness


def plot_positions(env, positions_history):
    """渲染历史状态（可选，用于画 Fig.14~16 那种图）"""
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)

    # 颜色映射
    colormap = cm.get_cmap('hsv', env.num_agents + 1)

    for step, (agent_positions, agent_directions, goal_positions,
               obstacle_positions) in enumerate(positions_history):

        # 绘制 USV
        for idx, (pos, direction) in enumerate(
                zip(agent_positions, agent_directions)):
            color = colormap(idx)

            # 观测扇区
            observation_angle = env.observation_angle
            start_angle = np.degrees(direction - observation_angle / 2)
            end_angle = np.degrees(direction + observation_angle / 2)
            wedge = patches.Wedge(
                pos, env.observation_radius, start_angle, end_angle,
                color='orange', alpha=0.01, label='Observation'
            )
            ax.add_patch(wedge)

            # USV 边界圆
            agent_circle = Circle(
                pos, radius=env.agent_radius,
                edgecolor=color, fill=False, alpha=0.5, linestyle='dashed'
            )
            ax.add_patch(agent_circle)

            # 椭圆船体
            agent_ellipse = Ellipse(
                pos, width=env.agent_radius * 2, height=env.agent_radius,
                angle=np.degrees(direction), color=color,
                fill=True, alpha=0.3, linewidth=1.5, label='Agent'
            )
            ax.add_patch(agent_ellipse)

            # 方向箭头
            arrow_dx = np.cos(direction) * env.agent_radius
            arrow_dy = np.sin(direction) * env.agent_radius
            agent_arrow = Arrow(
                pos[0], pos[1], arrow_dx, arrow_dy,
                width=0.01 * env.area_size, color=color
            )
            ax.add_patch(agent_arrow)

            if step == 0:
                ax.text(pos[0], pos[1], str(idx + 1), color=color,
                        fontsize=12, ha='center', va='center')

        if step == 0:
            # 目标
            for idx, pos in enumerate(goal_positions):
                goal_circle = Circle(
                    pos, env.goal_radius, color='green',
                    fill=True, alpha=0.3, label='Goal'
                )
                ax.add_patch(goal_circle)
                ax.text(pos[0], pos[1], str(idx + 1), color='green',
                        fontsize=12, ha='center', va='center')

            # 障碍物
            for idx, (pos, radius) in enumerate(
                    zip(obstacle_positions, env.obstacle_radii)):
                obstacle_circle = Circle(
                    pos, radius, color='red',
                    fill=True, alpha=0.3, linewidth=1.5, label='Obstacle'
                )
                ax.add_patch(obstacle_circle)
                ax.text(pos[0], pos[1], str(idx + 1), color='red',
                        fontsize=12, ha='center', va='center')

    plt.grid(True)
    plt.legend(handles=[goal_circle, obstacle_circle, wedge], loc='best')
    plt.show()


def ade_differential_evolution(weights_matrix,
                               num_generations,
                               F=0.8,
                               CR=0.9,
                               step_count=150):
    """
    ADE 算法：
    - 多算子 DE（4 个变异算子）
    - 对每个算子维护 Beta(α_k, β_k)
    - 使用汤普森采样自适应选择算子
    """
    pop_size, variable_dim = weights_matrix.shape

    # 初始化 Beta 分布参数（每个算子一对 alpha / beta）
    num_ops = 4
    alpha_params = np.ones(num_ops)
    beta_params = np.ones(num_ops)

    # 记录算子选择和成功次数（可选，用于分析）
    op_select_count = np.zeros(num_ops, dtype=int)
    op_success_count = np.zeros(num_ops, dtype=int)

    best_fitness = np.inf
    best_weight = None
    fitness_history = []

    # 初始化所有个体适应度
    fitness_values = []
    for weight in weights_matrix:
        env = Environment(num_agents=num_agents, area_size=area_size,
                          agent_positions=agent_positions,
                          agent_directions=agent_directions,
                          goal_positions=goal_positions,
                          obstacle_positions=obstacle_positions,
                          obstacle_radii=obstacle_radii)
        fitness = evaluate_model(env, policy_weight=weight,
                                 step_count=step_count)
        fitness_values.append(fitness)
    fitness_values = np.array(fitness_values)

    for generation in range(num_generations):
        new_population = np.copy(weights_matrix)

        # 当前最好个体（用于包含 best 的算子）
        best_idx = np.argmin(fitness_values)
        w_best = weights_matrix[best_idx]

        for i in range(pop_size):
            # ---- 1) 汤普森采样：从每个算子的 Beta 分布采样 ----
            samples = np.random.beta(alpha_params, beta_params)
            op_id = int(np.argmax(samples))   # 选样值最大的算子
            op_select_count[op_id] += 1

            # ---- 2) 准备随机个体索引（不含 i）----
            idxs = np.arange(pop_size)
            idxs = idxs[idxs != i]
            r1, r2, r3 = np.random.choice(idxs, 3, replace=False)

            w_i = weights_matrix[i]
            wr1, wr2, wr3 = weights_matrix[r1], weights_matrix[r2], weights_matrix[r3]

            # ---- 3) 按选择的算子生成变异向量 mi ----
            if op_id == 0:
                # DE/rand/1: m = wr1 + F * (wr2 - wr3)
                mutant = wr1 + F * (wr2 - wr3)
            elif op_id == 1:
                # DE/best/1: m = w_best + F * (wr1 - wr2)
                mutant = w_best + F * (wr1 - wr2)
            elif op_id == 2:
                # DE/current-to-best/1:
                # m = w_i + F*(w_best - w_i) + F*(wr1 - wr2)
                mutant = w_i + F * (w_best - w_i) + F * (wr1 - wr2)
            else:
                # DE/rand-to-best/2:
                # m = wr1 + F*(w_best - wr1) + F*(wr2 - wr3)
                mutant = wr1 + F * (w_best - wr1) + F * (wr2 - wr3)

            # 变量范围裁剪到 [0, 1]
            mutant = np.clip(mutant, 0.0, 1.0)

            # ---- 4) 二项交叉（binomial crossover）生成试验个体 trial ----
            cross_points = np.random.rand(variable_dim) < CR
            if not np.any(cross_points):
                cross_points[np.random.randint(0, variable_dim)] = True
            trial = np.where(cross_points, mutant, w_i)

            # ---- 5) 评估试验个体适应度 ----
            env = Environment(num_agents=num_agents, area_size=area_size,
                              agent_positions=agent_positions,
                              agent_directions=agent_directions,
                              goal_positions=goal_positions,
                              obstacle_positions=obstacle_positions,
                              obstacle_radii=obstacle_radii)
            trial_fitness = evaluate_model(env, policy_weight=trial,
                                           step_count=step_count)

            # ---- 6) 选择：决定是否接受 trial，并计算奖励 r ----
            if trial_fitness < fitness_values[i]:
                new_population[i] = trial
                fitness_values[i] = trial_fitness
                reward = 1
                op_success_count[op_id] += 1
            else:
                reward = 0

            # ---- 7) Beta 分布参数更新（汤普森采样的贝叶斯更新）----
            alpha_params[op_id] += reward
            beta_params[op_id] += (1 - reward)

        # 更新种群
        weights_matrix = new_population

        # 记录当前代的最佳个体
        min_fitness = np.min(fitness_values)
        if min_fitness < best_fitness:
            best_fitness = min_fitness
            best_weight = weights_matrix[np.argmin(fitness_values)]

        fitness_history.append(best_fitness)

        if (generation + 1) % 10 == 0:
            end_time = time.time()
            execution_time = end_time - start_time
            print(f"执行时间: {execution_time:.4f} 秒")

        print(f"Generation {generation + 1}: "
              f"Best Fitness = {best_fitness:.4f}; "
              f"Best Weight = {best_weight}")

    print("算子选择次数:", op_select_count)
    print("算子成功次数:", op_success_count)
    print("算子成功率:", op_success_count / np.maximum(op_select_count, 1))

    return best_weight, best_fitness, fitness_history


if __name__ == '__main__':
    np.random.seed(100)
    num_agents = 10
    area_size = 100
    pop_size = 20      # 种群大小
    num_iter = 50      # 迭代次数
    F = 0.8
    CR = 0.9
    print(f'num_agents: {num_agents}; pop_size: {pop_size}; '
          f'num_iter: {num_iter}; F: {F}; CR: {CR}')
    variable_dim = 4   # 权重维度 [α, β, γ, κ]
    scenario = 1       # 场景选择

    if scenario == 1:
        # 圆形场景
        agent_positions = np.array(
            env_utils.generate_circle_points((0.5, 0.5), 0.4, num_agents, 0)
        ) * area_size
        agent_directions = np.random.uniform(-np.pi, np.pi, num_agents)
        goal_positions = np.array(
            env_utils.generate_circle_points((0.5, 0.5), 0.43, num_agents, np.pi)
        ) * area_size
        obstacle_positions = np.array(
            [[0.22, 0.55], [0.32, 0.29], [0.68, 0.34],
             [0.51, 0.70], [0.45, 0.46]]
        ) * area_size
        obstacle_radii = np.array([0.06, 0.07, 0.08, 0.10, 0.05]) * area_size

    elif scenario == 2:
        # 随机场景
        agent_positions = np.random.rand(num_agents, 2) * area_size
        agent_directions = np.random.uniform(-np.pi, np.pi, num_agents)
        goal_positions = np.random.rand(num_agents, 2) * area_size
        obstacle_positions = np.random.rand(10, 2) * area_size
        obstacle_radii = np.random.uniform(0.05, 0.1, 10) * area_size

    elif scenario == 3:
        # 走廊场景
        corridor_width = 1.0
        corridor_length = 10.0
        agent_positions = np.zeros((num_agents, 2))
        agent_positions[:, 0] = np.random.uniform(0.2 * area_size,
                                                  0.3 * area_size, num_agents)
        agent_positions[:, 1] = np.linspace(0.2 * area_size,
                                            0.8 * area_size, num_agents)
        agent_directions = np.zeros(num_agents)
        agent_directions[:] = 0.0
        goal_positions = np.zeros((num_agents, 2))
        goal_positions[:, 0] = np.random.uniform(0.7 * area_size,
                                                 0.8 * area_size, num_agents)
        goal_positions[:, 1] = np.linspace(0.2 * area_size,
                                           0.8 * area_size, num_agents)
        obstacle_positions = np.column_stack((
            np.random.uniform(0.4 * area_size, 0.6 * area_size, 5),
            np.random.uniform(0.25 * area_size, 0.75 * area_size, 5)
        ))
        obstacle_radii = np.random.uniform(0.05, 0.1, 5) * area_size

    # 初始化种群
    weights_matrix = np.random.rand(pop_size, variable_dim)

    start_time = time.time()
    # 执行 ADE 算法（动态汤普森采样自适应多算子 DE）
    best_weight, best_fitness, fitness_history = ade_differential_evolution(
        weights_matrix, num_generations=num_iter, F=F, CR=CR, step_count=150
    )
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"执行时间: {execution_time:.4f} 秒")
    print("最佳权重:\n", best_weight)
    print("最佳适应度值:\n", best_fitness)
    print("适应度历史:\n", fitness_history)
