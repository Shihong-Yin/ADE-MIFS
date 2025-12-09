#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Cooperative Path planning for unmanned surface vehicles """

__author__ = 'Shihong Yin'

# 加随机性，局部观测，障碍物大小和数量可变

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Ellipse, Arrow
import matplotlib.patches as patches
from scipy.optimize import linear_sum_assignment  # 匈牙利算法
import time
from rvo_inter import rvo_inter
import env_utils
import matplotlib.cm as cm
import os


class Environment:
    def __init__(self, num_agents=3, area_size=100,
                 agent_positions=None, agent_directions=None, goal_positions=None,
                 obstacle_positions=None, obstacle_radii=None):  # 添加可选参数
        self.num_agents = num_agents
        self.num_obstacles = 0 if obstacle_positions is None else len(obstacle_positions)
        self.num_goals = num_agents
        self.area_size = area_size  # 任务区域大小
        self.agent_radius = 0.01 * area_size
        self.goal_radius = 0.015 * area_size  # 设定目标半径用于计算避免 UAV 初始化位置位于目的地区域
        self.goal_threshold = 0.2 * area_size  # 目的地阈值，开始减速
        self.observation_angle = np.pi  # 观测角度
        self.observation_radius = 0.4 * area_size  # 观测半径
        self.observation_number = 36  # 观测向量维数
        self.neighbors_region = 0.2 * area_size  # 邻居范围  # # # 仅用于初始化 rvo_inter # # #
        self.neighbors_num = 10  # 邻居最大数量  # # # 仅用于初始化 rvo_inter # # #
        self.vel_max = np.array([0.15 * area_size, 3.14/4])  # 最大线速度、最大角速度
        self.vel_min = 0.02 * area_size  # 最小线速度，为了防止原地不动
        self.acceleration = np.array([0.03 * area_size, 0.2])  # 最大线加速度、最大角加速度（未使用）
        self.vx_max = 0.15 * area_size  # # # 仅用于初始化 rvo_inter # # #
        self.vy_max = 0.15 * area_size  # # # 仅用于初始化 rvo_inter # # #
        self.radius_exp = 0.02 * area_size  # USV 扩展的碰撞半径 sigma
        self.step_time = 0.2  # 每一步的采样时间
        self.ctime_threshold = 5  # rvo 碰撞时间阈值
        self.neighborhood_size = 1  # Size of the neighborhood to consider for averaging
        self.max_attempts = 1000  # reset() 超过1000次还没成功会提示
        self.env_train = False

        self.rvo = rvo_inter(self.neighbors_region, self.neighbors_num, self.vx_max, self.vy_max, self.acceleration,
                             self.env_train, self.radius_exp-self.agent_radius)

        self.agent_positions = agent_positions.copy()
        self.agent_directions = agent_directions.copy()
        self.agent_omni_velocities = np.zeros((self.num_agents, 2))  # 初始化全向速度为零
        self.agent_diff_velocities = np.zeros((self.num_agents, 2))  # 初始化差速速度为零
        self.goal_positions = goal_positions.copy()
        self.obstacle_positions = obstacle_positions.copy()
        self.obstacle_velocities = np.zeros((self.num_obstacles, 2))  # 初始化速度为零
        self.obstacle_radii = obstacle_radii.copy()
        self.prev_distances_to_goal = [None] * self.num_agents  # 保存上一步到目标的距离
        self.agent_collisions = None
        self.obstacle_collisions = None
        self.total_steps = None
        self.reset_statistics()  # 初始化统计数据

    def reset(self, initialization=False, label='ifs'):  # 用于信息融合策略的观测
        """ Reset the environment at the beginning of each episode. """
        if initialization:
            self.num_obstacles = np.random.randint(1, 6)  # 随机生成1到5个障碍物
            self.obstacle_positions = np.random.uniform(0, self.area_size, (self.num_obstacles, 2))
            self.obstacle_velocities = np.zeros((self.num_obstacles, 2))  # 初始化速度为零
            self.obstacle_radii = np.random.uniform(3, 6, self.num_obstacles)  # 每个障碍物的半径在3到6之间随机生成
            self.goal_positions = self.initialize_goals()  # 初始化目的地位置不能与威胁区域冲突
            self.agent_positions, self.agent_directions, self.agent_omni_velocities, self.agent_diff_velocities = self.initialize_agents()  # 返回位置和速度
        # 初始化每个智能体到目标的距离
        self.prev_distances_to_goal = np.linalg.norm(self.agent_positions - self.goal_positions, axis=1)
        self.reset_statistics()  # 每次重置环境时重置统计数据
        obs_n = []  # record observations for each agent
        for agent_i in range(self.num_agents):
            obs_n.append(self.get_local_obs(agent_i, label=label))
        return obs_n

    def initialize_goals(self):
        """ Initialize goal positions ensuring they do not overlap with obstacle zones. """
        positions = np.zeros((self.num_goals, 2))
        for i in range(self.num_goals):
            safe = False
            attempts = 0
            while not safe:
                new_position = np.random.uniform(0, self.area_size, (1, 2))
                if self.is_goal_safe(new_position):
                    positions[i] = new_position
                    safe = True
                attempts += 1
            if attempts >= self.max_attempts and attempts % self.max_attempts == 0:
                print("Failed to place all goals without conflict after", self.max_attempts, "attempts.")
        return positions

    def initialize_agents(self):
        """ Initialize agent positions ensuring they do not overlap with obstacle zones or goal zones. """
        positions = np.zeros((self.num_agents, 2))
        directions = np.random.uniform(-np.pi, np.pi, self.num_agents)  # 初始化在 [-pi, pi] 范围内的随机方向
        omni_velocities = np.zeros((self.num_agents, 2))  # 初始化速度为零
        diff_velocities = np.zeros((self.num_agents, 2))  # 初始化速度为零
        for i in range(self.num_agents):
            safe = False
            attempts = 0
            while not safe:
                new_position = np.random.uniform(0, self.area_size, (1, 2))
                if self.is_agent_safe(new_position):
                    positions[i] = new_position
                    safe = True
                attempts += 1
            if attempts >= self.max_attempts and attempts % self.max_attempts == 0:
                print("Failed to place all agents without conflict after", self.max_attempts, "attempts.")
        return positions, directions, omni_velocities, diff_velocities  # 同时返回位置和速度

    def is_goal_safe(self, position):
        """ Check if the goal position is safe from obstacles. """
        for obstacle_pos, obstacle_radius in zip(self.obstacle_positions, self.obstacle_radii):
            if np.linalg.norm(position - obstacle_pos) < self.goal_radius + obstacle_radius:
                return False
        return True

    def is_agent_safe(self, position):
        """ Check if the agent position is safe from obstacles and goals. """
        if not self.is_goal_safe(position):  # Reuse the same check for goals as well
            return False
        for goal in self.goal_positions:
            if np.linalg.norm(position - goal) < self.agent_radius + self.goal_radius:
                return False
        return True

    # 以上代码是 初始化环境，确保位置不冲突

    def get_local_obs(self, agent_index=0, label='ifs'):  # Agent 的局部观测信息用于 Actor 网络选择执行的动作
        """ Get the local observation for a specific agent. """
        # 获取激光测距仪观测
        distances = self.get_laser_obs(agent_index)
        # 获取目的地相对位置观测
        goal_relative_pos = self.get_goal_obs(agent_index)
        # 获取智能体的 RVO 观测
        # obs_vo_list: [apex_vector(2), vo_left_vector(2), vo_right_vector(2), min_distance, 1/(min_exp_time+0.2)]
        obs_vo_list, rec_exp_time = self.get_rvo_obs(agent_index)
        # 获取海事避碰规则观测
        colregs_info, ur_list = self.get_colregs_obs(agent_index)
        # 评估智能体的碰撞风险
        collision_risk, _ = self.collision_risk(obs_vo_list, agent_index)
        if label == 'ifs':
            return distances, goal_relative_pos, obs_vo_list, colregs_info
        elif label == 'rl':
            observation = np.concatenate([rec_exp_time, collision_risk, ur_list])
            return observation  # 经过归一化，都是越大越紧急

    def collision_risk(self, vo_list, agent_index=0):
        """ 计算智能体的碰撞风险 """
        # # 初始化碰撞风险数组为0
        # collision_risks = [0] * self.neighbors_num
        if len(vo_list) == 0:
            cr = 0
            collision_risks = [cr] * self.neighbors_num
            return collision_risks, cr  # 如果没有 VO 向量，返回全0列表
        # 获取当前智能体的速度和方向信息
        liner_velocity = self.agent_diff_velocities[agent_index][0]
        current_direction = self.agent_directions[agent_index]

        # 根据当前速度、加速度、最大速度和最小速度限制的速度值
        max_speed = min(liner_velocity + self.acceleration[0] * self.step_time, self.vel_max[0])
        min_speed = max(liner_velocity - self.acceleration[0] * self.step_time, self.vel_min)

        # 生成可能的速度值
        possible_speeds = np.linspace(min_speed, max_speed, num=3, endpoint=True)  # # # 线速度的可能值

        start_angle = current_direction - self.observation_angle / 2  # # # 判断整个圆周会不会更好
        end_angle = current_direction + self.observation_angle / 2
        possible_angles = np.linspace(start_angle, end_angle, num=self.observation_number, endpoint=True)  # 角速度的可能值

        # 生成所有可能的速度向量
        possible_velocities = np.array([
            [speed * np.cos(angle), speed * np.sin(angle)]
            for speed in possible_speeds for angle in possible_angles
        ])

        # 判断每个可能速度是否在VO集内
        n_collision = 0
        n_total = possible_velocities.shape[0]
        for velocity in possible_velocities:
            for vo_vector in vo_list:
                if not self.rvo.vo_out_jud_vector2(velocity[0], velocity[1], vo_vector):
                    n_collision += 1
                    break
        # 计算碰撞风险，等于在VO集内的速度向量占所有可能速度的比例
        cr = n_collision / n_total
        collision_risks = [cr] * self.neighbors_num
        # # 对每个 VO 向量单独计算碰撞风险
        # n_total = possible_velocities.shape[0]
        # for i, vo_vector in enumerate(vo_list):
        #     n_collision = 0
        #     for velocity in possible_velocities:
        #         if not self.rvo.vo_out_jud_vector2(velocity[0], velocity[1], vo_vector):
        #             n_collision += 1
        #             # break
        #     # 计算碰撞风险，等于在VO集内的速度向量占所有可能速度的比例
        #     collision_risks[i] = n_collision / n_total
        return collision_risks, cr

    def get_laser_obs(self, agent_index=0):
        """ 获取激光测距仪的观测 """
        agent_pos = self.agent_positions[agent_index]
        agent_dir = self.agent_directions[agent_index]
        directions = np.linspace(agent_dir - self.observation_angle/2, agent_dir + self.observation_angle/2, num=self.observation_number, endpoint=True)  # 激光的方向(弧度制)

        obstacles_relative_pos = [(agent_pos, self.radius_exp)]  # 计算障碍物的位置和半径
        # 其他 UAV 的相对位置信息
        for i, other_agent_pos in enumerate(self.agent_positions):
            if i != agent_index:
                obstacles_relative_pos.append((other_agent_pos, self.radius_exp))  # 以元组形式添加位置和半径
        # 添加观测范围内的障碍物的位置信息
        for obstacle_pos, obstacle_radius in zip(self.obstacle_positions, self.obstacle_radii):
            obstacles_relative_pos.append((obstacle_pos, obstacle_radius))  # 以元组形式添加位置和半径

        # Calculate distances
        distances = self.calculate_distances(agent_pos, obstacles_relative_pos, directions)
        return distances

    def get_goal_obs(self, agent_index=0):
        """ 获取目标区域的相对位置信息 """
        agent_pos = self.agent_positions[agent_index]
        goal_relative_pos = []  # 所有目标区域的相对位置信息
        for goal_pos in self.goal_positions:
            goal_relative_pos.append(goal_pos - agent_pos)  # 添加相对位置
        return goal_relative_pos

    def get_rvo_obs(self, agent_index=0):
        """ 获取智能体的 RVO 观测 """
        # 邻居智能体的状态列表
        nei_state_list = []
        for i in range(self.num_agents):
            # 每个智能体的状态：位置（x, y），速度（vx, vy），半径
            state = np.array(list(self.agent_positions[i]) + list(self.agent_omni_velocities[i]) + [self.agent_radius])
            nei_state_list.append(state)
        # 圆形障碍物的列表
        obs_obstacle_list = []
        for j in range(self.num_obstacles):
            # 每个障碍物的状态：位置（x, y），速度（vx, vy），半径
            obstacle = np.array(list(self.obstacle_positions[j]) + list(self.obstacle_velocities[j]) + [self.obstacle_radii[j]])
            obs_obstacle_list.append(obstacle)

        # 构建智能体的全向状态 agent_state: [x, y, vx, vy, radius]
        agent_pos = self.agent_positions[agent_index]
        agent_vel = self.agent_omni_velocities[agent_index]
        radius = np.array([self.radius_exp])
        agent_state = np.concatenate((agent_pos, agent_vel, radius))  # 构建USV的位置、速度、半径、期望速度

        # 计算与障碍物相关的 VO 信息，返回障碍物 VO 列表和预期碰撞时间
        # obs_vo_list: [apex_vector(2), vo_left_vector(2), vo_right_vector(2), min_distance, 1/(min_exp_time+0.2), state, circular, mode]
        obs_line_list = []
        obs_vo_list, _, _, _ = self.rvo.config_vo_inf(agent_state, nei_state_list, obs_obstacle_list, obs_line_list, action=agent_vel)

        # 初始化 rec_exp_time 列表
        rec_exp_time = [0] * self.neighbors_num
        # 检查 obs_vo_list 是否为空，并更新 rec_exp_time
        if obs_vo_list:
            for i in range(len(obs_vo_list)):
                rec_exp_time[i] = obs_vo_list[i][7] / 5  # 提取 obs_vo_list 元素的最后一个值（归一化）
        return obs_vo_list, rec_exp_time

    def get_colregs_obs(self, agent_index=0):
        """ 根据 智能体的速度和其他智能体的状态 判断海事避碰情况，并返回相应情况的编号及紧急程度 ur """
        os_pos = self.agent_positions[agent_index]
        # os_vel = self.agent_omni_velocities[agent_index]
        os_direction = self.agent_directions[agent_index]  # os的当前方向角度
        os_dir = np.array([np.cos(os_direction), np.sin(os_direction)])
        colregs_info = []
        ur_list = []
        # 计算与指定智能体的距离
        distances = np.linalg.norm(self.agent_positions - self.agent_positions[agent_index], axis=1)
        for i in range(self.num_agents):
            if i == agent_index:
                colregs_info.append((None, None, None))
                ur_list.append(0)
                continue  # 排除自己
            if distances[i] > self.observation_radius:
                colregs_info.append((None, None, None))
                ur_list.append(0)
                continue  # 如果距离大于观测半径，跳过
            # 获取其他USV的位置和速度
            ts_pos = self.agent_positions[i]
            ts_direction = self.agent_directions[i]  # ts 的当前方向角度
            ts_dir = np.array([np.cos(ts_direction), np.sin(ts_direction)])
            ts_relative_os = env_utils.calculate_angle(os_dir, np.squeeze(ts_pos - os_pos))  # ts 相对于 os 的角度
            os_relative_ts = env_utils.calculate_angle(ts_dir, np.squeeze(os_pos - ts_pos))  # os 相对于 ts 的角度
            line_speed = self.agent_diff_velocities[i][0]  # ts 船的线速度

            colregs_case = None  # 初始化避碰规则情况

            # 头对头相遇 (Head-on)
            if - 22.5 <= ts_relative_os <= 22.5 and - 22.5 <= os_relative_ts <= 22.5:
                colregs_case = 1  # 对遇
            # 右舷横越 (Crossing Give-way1)
            elif - 22.5 <= ts_relative_os <= 67.5 and - 112.5 <= os_relative_ts <= -22.5:
                colregs_case = 2  # 右舷相交
            # 右舷横越 (Crossing Give-way2)
            elif 67.5 <= ts_relative_os <= 112.5 and - 112.5 <= os_relative_ts <= -22.5:
                colregs_case = 3  # 大角度右舷相交

            if colregs_case:
                relative_angle_radians = np.radians(ts_relative_os)  # ts 相对于 os 的角度(弧度)
                ur = 1 - (abs(relative_angle_radians) / np.pi)  # 根据公式计算紧急程度 ur
                # 存储触发避碰规则的船只信息
                colregs_info.append((colregs_case, line_speed, relative_angle_radians))
                ur_list.append(ur)
            else:
                colregs_info.append((None, None, None))
                ur_list.append(0)
        return colregs_info, ur_list

    def calculate_boundary_intersection(self, usv_position, direction_vector):
        min_distance = float('inf')  # 初始化最小距离为无穷大
        for dim in range(2):  # 检查x和y维度
            if direction_vector[dim] != 0:  # 如果方向向量在该维度上的分量不为0
                # 计算与边界0和area_size的交点
                for bound in [0, self.area_size]:
                    t = (bound - usv_position[dim]) / direction_vector[dim]
                    if t > 0:  # 交点必须在方向向量的前进方向上
                        # 计算交点的位置
                        intersection_point = usv_position + t * direction_vector
                        # 检查交点是否真的在边界内（在另一个维度上也要在0到area_size之间）
                        if 0 <= intersection_point[1 - dim] <= self.area_size:
                            distance = np.linalg.norm(t * direction_vector)
                            if distance < min_distance:
                                min_distance = distance
        return min_distance

    def calculate_distances(self, uav_position, obstacles, directions):
        """ 计算测距仪的观测距离 """
        distances = []
        for angle in directions:
            direction_vector = np.array([np.cos(angle), np.sin(angle)])
            closest_distance = self.calculate_boundary_intersection(uav_position, direction_vector)  # 边界的距离
            for obstacle, radius in obstacles:
                obstacle_vector = np.array(obstacle) - np.array(uav_position)
                distance_along_direction = np.dot(obstacle_vector, direction_vector)
                perpendicular_distance = np.linalg.norm(obstacle_vector - distance_along_direction * direction_vector)
                if perpendicular_distance <= radius:
                    delta = np.sqrt(radius ** 2 - perpendicular_distance ** 2)
                    obstacle_distance = distance_along_direction - delta
                    if 0 <= obstacle_distance <= self.observation_radius:
                        closest_distance = min(closest_distance, obstacle_distance)
            distances.append(min(closest_distance, self.observation_radius))
        return distances

    def reset_statistics(self):
        """ Reset collision statistics at the beginning of each episode. """
        self.agent_collisions = 0
        self.obstacle_collisions = 0
        self.total_steps = 0

    def calculate_action(self, obs_n, t, policy_weight=None):
        """ 根据策略权重计算动作 """
        if policy_weight is None:
            policy_weight = np.tile([0.39112343, 0.99680752, 0.56221616, 0.44493471], (self.num_agents, 1))
        actions = np.zeros((self.num_agents, 2))  # Initialize actions array
        # Parameters for motion and decision-making
        size = self.area_size
        linear_speed_limit = self.vel_max[0]  # Maximum speed in any direction
        # 计算每个智能体的动作
        for i, obs in enumerate(obs_n):
            alpha = policy_weight[i][0] / size  # 目标的重要度
            beta = policy_weight[i][1] / size
            gamma = policy_weight[i][2]
            omega = policy_weight[i][3]
            # obs_vo_list: [apex_vector(2), vo_left_vector(2), vo_right_vector(2), min_distance, 1/(min_exp_time+0.2), state, circular, mode]
            distances, goal_relative_pos, obs_vo_list, colregs_info = obs
            goal_position = goal_relative_pos[i]  # 解析分配结果
            current_direction = self.agent_directions[i]  # usv 的当前方向
            # current_omni_velocity = self.agent_omni_velocities[i]  # usv 的当前全向速度
            current_diff_velocity = self.agent_diff_velocities[i]  # usv 的当前差速速度

            # Determine direction to the goal
            distance_to_goal = np.linalg.norm(goal_position)
            goal_direction = goal_position / distance_to_goal  # Normalize to unit vector, goal_position 是相对位置，本身就具有方向

            # Determine desired speed based on distance to goal
            if distance_to_goal < self.goal_threshold:  # If close to the goal, slow down proportionally
                desired_speed = distance_to_goal * (linear_speed_limit / self.goal_threshold)
            else:
                desired_speed = linear_speed_limit

            # Avoid obstacles based on distance measurements
            start_angle = current_direction - self.observation_angle / 2
            end_angle = current_direction + self.observation_angle / 2
            directions = np.linspace(start_angle, end_angle, num=self.observation_number, endpoint=True)  # 激光的方向(弧度制)

            # 每个方向的 RVO 成本
            direction_rvo_costs = []
            for k, angle in enumerate(directions):
                direction = np.array([np.cos(angle), np.sin(angle)])
                diff_speed, theta = self.calculate_velocity_and_direction(current_diff_velocity, current_direction, direction, desired_speed)
                expected_speed = np.array([diff_speed[0] * np.cos(theta), diff_speed[0] * np.sin(theta)])  # 这一步不够精确
                min_exp_time = self.calculate_min_exp_time(obs_vo_list, expected_speed)  # 计算最小预期碰撞时间，越大越好
                convert_exp_time = - 1 / (min_exp_time + 0.1) / 50  # 转换碰撞时间，越大越好(-0.2,0) 0.1 RVO 的效果更弱
                direction_rvo_costs.append(convert_exp_time)

            # 每个方向的 colregs 成本
            direction_colregs_costs = np.zeros(directions.shape)
            for colregs_case, ts_line_speed, relative_angle in colregs_info:
                if colregs_case == 1 or colregs_case == 2:  # 如果 colregs_info 不为空，则计算每个方向的 colregs 成本
                    indices = np.where(directions < current_direction - max(0, relative_angle))[0]  # 找到所有小于 TS 方向角的索引
                    # ur = 1 - (abs(relative_angle) / np.pi)  # 根据公式计算紧急程度 ur
                    direction_colregs_costs[indices] += ts_line_speed / self.vel_max[0] / 5  # ts_line_speed / self.vel_max[0] 归一化为[0, 0.2]，将满足条件的索引对应的 direction_colregs_costs 元素加 ts_line_speed / self.vel_max[0]
                elif colregs_case == 3:
                    indices = np.where(directions > current_direction - min(0, relative_angle))[0]  # 找到所有小于 TS 方向角的索引
                    # ur = 1 - (abs(relative_angle) / np.pi)  # 根据公式计算紧急程度 ur
                    direction_colregs_costs[indices] += ts_line_speed / self.vel_max[0] / 5  # 归一化为[0, 0.2]，将满足条件的索引对应的 direction_colregs_costs 元素加 ts_line_speed / self.vel_max[0]

            # 每个方向的 goal 成本和障碍物距离成本
            direction_goal_costs1 = []
            direction_goal_costs2 = []
            parameter = np.sqrt(1 / ((self.goal_radius / distance_to_goal) ** 2 + 1))  # 判断目的地与观测是否处于同向
            for k, angle in enumerate(directions):
                dist = distances[k]
                direction_vector = np.array([np.cos(angle), np.sin(angle)])

                dot_product = np.dot(direction_vector, goal_direction)
                if dot_product > parameter and distance_to_goal < min(self.observation_radius, dist):
                    dist = self.observation_radius
                # 该成本基于两个主要因素：与目标方向 (base_direction) 的对齐程度和在该方向上的距离 (dist)
                direction_cost1 = alpha * (self.observation_radius / 2) * dot_product
                direction_goal_costs1.append(direction_cost1)
                direction_cost2 = beta * dist
                direction_goal_costs2.append(direction_cost2)

            avg_direction_goal_costs1 = []
            avg_direction_goal_costs2 = []
            for k, angle in enumerate(directions):
                direction_costs1 = []
                direction_costs2 = []
                for offset in range(-self.neighborhood_size, self.neighborhood_size + 1):
                    idx = k + offset
                    if idx < 0:
                        idx = 0
                    elif idx >= len(distances):
                        idx = len(distances) - 1
                    direction_costs1.append(direction_goal_costs1[idx])
                    direction_costs2.append(direction_goal_costs2[idx])
                avg_direction_cost1 = np.mean(direction_costs1)
                avg_direction_goal_costs1.append(avg_direction_cost1)
                avg_direction_cost2 = np.mean(direction_costs2)
                avg_direction_goal_costs2.append(avg_direction_cost2)

            merge_direction_cost = np.array(avg_direction_goal_costs1) + np.array(avg_direction_goal_costs2) + gamma * np.array(
                direction_rvo_costs) + omega * direction_colregs_costs
            angle = directions[np.argmax(merge_direction_cost)]
            # if 35 > t > 33 and i == 0:
            #     plt.figure(figsize=(8, 8))
            #     plt.plot(np.array(avg_direction_goal_costs1), label='Goal costs', marker='o')
            #     plt.plot(np.array(avg_direction_goal_costs2), label='Obstacle costs', marker='^')
            #     plt.plot(gamma * np.array(direction_rvo_costs), label='RVO costs', marker='x')
            #     plt.plot(omega * direction_colregs_costs, label='COLREGs costs', marker='s')
            #     plt.plot(merge_direction_cost, label='Merge costs', marker='d')
            #
            #     plt.xlabel('Directions')
            #     plt.ylabel('Costs')
            #     plt.legend()
            #     plt.grid(True)
            #     # plt.draw()
            #     # plt.pause(0.01)
            #     # 根据 t 生成 PDF 文件名
            #     pdf_filename = f"direction_cost_t{t}.pdf"
            #     pdf_path = os.path.join(output_folder, pdf_filename)
            #
            #     # 保存图像到指定文件夹和文件名
            #     plt.savefig(pdf_path, format='pdf', bbox_inches='tight')
            #     render_fig = self.render(agent_index=0)
            #     # 生成第二个 PDF 文件名
            #     render_pdf_filename = f"render_t{t}.pdf"
            #     render_pdf_path = os.path.join(output_folder, render_pdf_filename)
            #
            #     # 保存 render 函数生成的图像
            #     render_fig.savefig(render_pdf_path, format='pdf', bbox_inches='tight')
            #     plt.show()
            best_direction = np.array([np.cos(angle), np.sin(angle)])
            # 计算新的速度向量
            actions[i, :], _ = self.calculate_velocity_and_direction(current_diff_velocity, current_direction, best_direction, desired_speed)
        return actions

    def calculate_min_exp_time(self, obs_vo_list, expected_speed):
        """ Calculate the minimum expected collision time from a list of velocity obstacles """
        min_exp_time = float('inf')
        # min_dis = float('inf')
        for obs_vo in obs_vo_list:
            vo_vector = obs_vo[0:6]
            agent_state = obs_vo[8]  # 智能体的全向状态 agent_state: [x, y, vx, vy, r]
            obstacle_state = obs_vo[9]  # 障碍物的全向状态 agent_state: [mx, my, mvx, mvy, mr]
            vo_mode = obs_vo[10]
            if self.rvo.vo_out_jud_vector2(expected_speed[0], expected_speed[1], vo_vector):  # 期望速度在 VO 向量之外
                exp_time = float('inf')  # 没有碰撞可能
            else:
                x, y, vx, vy, r = agent_state
                mx, my, mvx, mvy, mr = obstacle_state
                rel_x = x - mx
                rel_y = y - my
                # real_dis_mr = np.sqrt(rel_x ** 2 + rel_y ** 2)
                # 计算相对速度
                if vo_mode == 'vo':
                    rel_vx = expected_speed[0] - mvx
                    rel_vy = expected_speed[1] - mvy
                elif vo_mode == 'rvo':
                    rel_vx = 2 * expected_speed[0] - mvx - vx
                    rel_vy = 2 * expected_speed[1] - mvy - vy
                exp_time = self.rvo.cal_exp_tim(rel_x, rel_y, rel_vx, rel_vy, r + mr)  # 计算期望碰撞时间
                if exp_time > self.ctime_threshold:
                    exp_time = float('inf')
            # temp_min_dis = real_dis_mr - mr
            # if temp_min_dis < min_dis:
            #     min_dis = temp_min_dis  # 越大越好
            if exp_time < min_exp_time:
                min_exp_time = exp_time  # 越小越紧急
        return min_exp_time

    def calculate_velocity_and_direction(self, current_diff_velocity, current_direction, best_direction, desired_speed):
        """ 根据当前方向和最佳方向，计算线速度和角速度 """
        # Calculate the desired velocity in the best direction
        desired_velocity = best_direction * desired_speed
        # 计算当前速度的线速度和方向
        current_speed = current_diff_velocity[0]
        current_angle = current_direction
        # 计算期望速度的线速度和方向
        desired_angle = np.arctan2(desired_velocity[1], desired_velocity[0])
        # 计算速度和方向的差异
        angle_diff = desired_angle - current_angle
        # 限制角度差异在 [-pi, pi] 范围内
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        elif angle_diff < -np.pi:
            angle_diff += 2 * np.pi
        # 如果角度差异超过了采样时间内的最大转向角度，调整速度和方向
        if np.abs(angle_diff) > self.vel_max[1] * self.step_time:  # self.vel_max[1] 表示最大角速度
            # 角度差很大，超过了采样时间的最大转弯角度，就最大程度降低线速度，并且以最大角速度转向期望方向
            linear_speed = min(desired_speed, max(self.vel_min, current_speed - self.acceleration[0] * self.step_time))
            angle_speed = np.sign(angle_diff) * self.vel_max[1]
        else:
            # 角度差较小，在采样时间内能够转向期望方向，就尽可能接近期望线速度，并且以匀速转向行驶
            linear_speed = min(desired_speed, current_speed + self.acceleration[0] * self.step_time)
            angle_speed = angle_diff / self.step_time
        theta = current_angle + angle_speed * self.step_time
        # 计算新的速度向量
        return np.array([linear_speed, angle_speed]), theta

    def step(self, actions):
        """ Update environment states based on agent velocity actions and return observations according to the specified mode """
        delta_time = self.step_time  # 使用定义的采样时间
        self.prev_distances_to_goal = np.linalg.norm(self.agent_positions - self.goal_positions, axis=1)
        # 对每个智能体进行更新
        for i in range(self.num_agents):
            linear_speed = actions[i, 0]  # 获取线速度
            angular_speed = actions[i, 1]  # 获取角速度
            # 获取当前状态
            x, y = self.agent_positions[i]
            theta = self.agent_directions[i]
            # 更新位置和方向
            if abs(angular_speed) > 0.01:
                ratio = linear_speed / angular_speed
                x_new = x - ratio * np.sin(theta) + ratio * np.sin(theta + angular_speed * delta_time)
                y_new = y + ratio * np.cos(theta) - ratio * np.cos(theta + angular_speed * delta_time)
                theta_new = theta + angular_speed * delta_time
            else:  # 直线行驶
                x_new = x + linear_speed * np.cos(theta) * delta_time
                y_new = y + linear_speed * np.sin(theta) * delta_time
                theta_new = theta
            # 确保 theta_new 在 [-pi, pi] 范围内
            # theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi
            # 更新智能体的位置和方向
            self.agent_positions[i] = [x_new, y_new]
            self.agent_directions[i] = theta_new
            # 更新速度
            self.agent_omni_velocities[i] = [linear_speed * np.cos(theta_new), linear_speed * np.sin(theta_new)]
            self.agent_diff_velocities[i] = [linear_speed, angular_speed]

        # self.agent_positions = np.clip(self.agent_positions, 0, self.area_size)  # 确保智能体不会超出边界
        rewards, dones = self.calculate_rewards(), self.is_done()  # 计算奖励和是否完成
        self.total_steps += self.num_agents  # 增加步数统计

        # self.render(0)

        return [self.get_local_obs(i) for i in range(self.num_agents)], rewards, dones


    def calculate_rewards(self):
        """ Calculate the rewards for the current state of the environment """
        # Reward agents based on closeness to the goals
        distances_to_goal = np.linalg.norm(self.agent_positions - self.goal_positions, axis=1)
        rewards = self.prev_distances_to_goal - distances_to_goal  # 如果当前距离小于上一步距离，给予奖励，反之给予惩罚
        # Add penalties for collisions
        for i in range(self.num_agents):
            if distances_to_goal[i] < self.goal_radius:  # 智能体到达了目标
                rewards[i] += 100
            else:  # 智能体未到达目标，给予时间步数惩罚，鼓励智能体尽快完成任务
                rewards[i] -= 1
            # 获取智能体的 RVO 观测
            obs_vo_list, rec_exp_time = self.get_rvo_obs(i)
            rewards[i] -= max(rec_exp_time)  # 预期碰撞时间的惩罚，0到1之间
            # 获取海事避碰规则观测
            _, ur_list = self.get_colregs_obs(i)
            rewards[i] -= max(ur_list)  # 紧急程度的惩罚，0到1之间
            # 评估智能体的碰撞风险
            _, cr = self.collision_risk(obs_vo_list, i)
            rewards[i] -= cr  # 碰撞风险的惩罚，0到1之间
            # Penalty for being too close to another agent
            for j in range(self.num_agents):
                if i != j:
                    if np.linalg.norm(self.agent_positions[i] - self.agent_positions[j]) < 2 * self.radius_exp:
                        self.agent_collisions += 1
                        rewards[i] -= 50
            # Penalty for being too close to a obstacle
            for obstacle_pos, obstacle_radius in zip(self.obstacle_positions, self.obstacle_radii):
                if np.linalg.norm(self.agent_positions[i] - obstacle_pos) < self.radius_exp + obstacle_radius:
                    self.obstacle_collisions += 1
                    rewards[i] -= 50
        return rewards.tolist()

    def is_done(self):
        """ Check if each USV's episode is done based on several conditions """
        done_status = [False] * self.num_agents  # Initialize list to keep track of each USV's done status
        # Check if there are any collisions
        # if self.agent_collisions > 0 or self.obstacle_collisions > 0:
        #     done_status = [True] * self.num_agents
        #     return done_status  # 有碰撞就停止

        # Check if each USV has reached any goal
        for i, agent_position in enumerate(self.agent_positions):
            for goal_position in self.goal_positions:
                if np.linalg.norm(agent_position - goal_position) < self.goal_radius:
                    done_status[i] = True  # This USV has reached a goal and is done
                    break  # No need to check other goals for this USV
        return done_status  # List of done status for each USV

    def get_collision_statistics(self):
        """ Return collision statistics as ratios of total steps. """
        # agent_collision_ratio = self.agent_collisions / self.total_steps if self.total_steps > 0 else 0
        # obstacle_collision_ratio = self.obstacle_collisions / self.total_steps if self.total_steps > 0 else 0
        # 直接统计碰撞次数
        agent_collision = self.agent_collisions
        obstacle_collision = self.obstacle_collisions
        total_step = self.total_steps
        return agent_collision, obstacle_collision, total_step

    def render(self, agent_index=0):
        """ 渲染函数 """
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim(0, self.area_size)
        ax.set_ylim(0, self.area_size)
        # Generate a colormap
        colormap = cm.get_cmap('hsv', self.num_agents + 1)
        # 绘制智能体的位置和方向
        for idx, (pos, direction) in enumerate(zip(self.agent_positions, self.agent_directions)):
            color = colormap(idx)
            # 绘制椭圆，椭圆的长轴与智能体方向一致
            agent_circle = Circle(pos, radius=self.radius_exp, edgecolor=color, fill=False, alpha=0.5,
                                  linestyle='dashed')  # 使用圆形表示USV边界
            ax.add_patch(agent_circle)
            agent_ellipse = Ellipse(pos, width=self.agent_radius * 2, height=self.agent_radius,
                                    angle=np.degrees(direction), color=color, fill=True, alpha=0.3, linewidth=1.5,
                                    label='USV')
            ax.add_patch(agent_ellipse)
            # 绘制方向箭头
            arrow_dx = np.cos(direction) * self.agent_radius
            arrow_dy = np.sin(direction) * self.agent_radius
            agent_arrow = Arrow(pos[0], pos[1], arrow_dx, arrow_dy, width=0.01 * self.area_size, color=color)
            ax.add_patch(agent_arrow)
            ax.text(pos[0], pos[1], str(idx + 1), color=color, fontsize=12, ha='center', va='center')
        for idx, pos in enumerate(self.goal_positions):
            goal_circle = Circle(pos, self.goal_radius, color='green', fill=True, alpha=0.3, label='Goal')
            ax.add_patch(goal_circle)
            ax.text(pos[0], pos[1], str(idx + 1), color='green', fontsize=12, ha='center', va='center')
        for idx, (pos, radius) in enumerate(zip(self.obstacle_positions, self.obstacle_radii)):
            obstacle_circle = Circle(pos, radius, color='red', fill=True, alpha=0.3, linewidth=1.5, label='Obstacle')
            ax.add_patch(obstacle_circle)
            ax.text(pos[0], pos[1], str(idx + 1), color='red', fontsize=12, ha='center', va='center')
        # 获取观测信息
        distances, goal_relative_pos, obs_vo_list, colregs_info = self.get_local_obs(agent_index)
        pos = self.agent_positions[agent_index]
        # 绘制激光测距仪的观测距离
        agent_dir = self.agent_directions[agent_index]
        start_angle = agent_dir - self.observation_angle / 2
        end_angle = agent_dir + self.observation_angle / 2
        angles = np.linspace(start_angle, end_angle, num=self.observation_number, endpoint=True)  # 激光的方向(弧度制)
        for distance, angle in zip(distances, angles):
            end_pos = pos + distance * np.array([np.cos(angle), np.sin(angle)])
            ax.plot([pos[0], end_pos[0]], [pos[1], end_pos[1]], color='orange', linestyle='dashed', label='Laser observation')

        # 绘制目的地相对位置
        ax.plot([pos[0], pos[0] + goal_relative_pos[agent_index][0]], [pos[1], pos[1] + goal_relative_pos[agent_index][1]], color='green',
                linestyle='dotted', label='Goal orientation')
        #
        # # 绘制 RVO 观测信息
        # for obs_vo in obs_vo_list:
        #     scale_factor = 0.5 * self.area_size  # 这个因子用于缩放箭头的长度
        #     apex_vector = pos + obs_vo[:2]
        #     vo_left_vector = np.array(obs_vo[2:4]) * scale_factor
        #     vo_right_vector = np.array(obs_vo[4:6]) * scale_factor
        #     ax.arrow(apex_vector[0], apex_vector[1], vo_left_vector[0], vo_left_vector[1], color='blue', width=0.001 * self.area_size,
        #              alpha=0.5, label='RVO left')
        #     ax.arrow(apex_vector[0], apex_vector[1], vo_right_vector[0], vo_right_vector[1], color='red', width=0.001 * self.area_size,
        #              alpha=0.5, label='RVO right')
        handles = [agent_ellipse, goal_circle, obstacle_circle]
        labels = ['USV', 'Goal', 'Obstacle']
        # handles.append(plt.Line2D([], [], color='orange', linestyle='dashed', label='Laser'))
        # labels.append('Laser observation')
        # handles.append(plt.Line2D([], [], color='green', linestyle='dotted', label='Goal orientation'))
        # labels.append('Goal orientation')
        # handles.append(plt.Line2D([], [], color='blue', linestyle='-', label='RVO left'))
        # labels.append('RVO left')
        # handles.append(plt.Line2D([], [], color='red', linestyle='-', label='RVO right'))
        # labels.append('RVO right')
        plt.grid(True)
        plt.legend(handles=handles, labels=labels, loc='best')
        plt.show()
        return fig


def evaluate_model(env, n_episode=3, step_count=10):
    """ 用于评估训练的 agent 模型 """
    # Assuming the environment and other parameters are defined globally or passed to this function
    total_agent_collisions = 0
    total_obstacle_collisions = 0
    total_steps = 0
    # total_distance_traveled = np.zeros(env.num_agents)

    for episode in range(n_episode):
        obs_n = env.reset(initialization=False, label='ifs')
        positions_history = [(np.copy(env.agent_positions), np.copy(env.agent_directions),
                              np.copy(env.goal_positions), np.copy(env.obstacle_positions))]

        for t in range(step_count):
            # if 10 < t:
            #     env.render()
            start_time = time.time()
            actions = env.calculate_action(obs_n, t, policy_weight=None)
            end_time = time.time()
            # 计算执行时间
            execution_time = end_time - start_time
            print(f"决策步: {t}, 执行时间: {execution_time:.4f} 秒")
            obs_n, rew_n, dones_n = env.step(np.array(actions))
            positions_history.append((np.copy(env.agent_positions), np.copy(env.agent_directions),
                                      np.copy(env.goal_positions), np.copy(env.obstacle_positions)))
            if all(dones_n):  # 检查是否所有 UAV 的任务都已结束
                break  # 如果是，则终止当前回合的循环

        plot_positions(env, positions_history, episode)  # Plot the history of positions
        agent_collision, obstacle_collision, total_step = env.get_collision_statistics()
        total_agent_collisions += agent_collision
        total_obstacle_collisions += obstacle_collision
        total_steps += total_step

        # 计算总行驶距离
        total_distance_traveled = np.sum(np.linalg.norm(np.diff(np.array([pos[0] for pos in positions_history]), axis=0), axis=2))

    avg_agent_collision = total_agent_collisions / n_episode
    avg_obstacle_collision = total_obstacle_collisions / n_episode
    avg_total_step = total_steps / env.num_agents / n_episode
    avg_distance_traveled = total_distance_traveled / env.num_agents / n_episode

    print(f'Average Collision Rate Between Agents: {avg_agent_collision:.4f}')
    print(f'Average Collision Rate Between Agents and Obstacles: {avg_obstacle_collision:.4f}')
    print(f'Average number of decision steps: {avg_total_step:.4f}')
    print(f'Average Distance Traveled by Agents: {avg_distance_traveled:.4f}')


def plot_positions(env, positions_history, episode):
    """ 渲染历史状态 """
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, env.area_size)
    ax.set_ylim(0, env.area_size)
    # Generate a colormap
    colormap = cm.get_cmap('hsv', env.num_agents+1)
    for step, (agent_positions, agent_directions, goal_positions, obstacle_positions) in enumerate(positions_history):
        # Plot agents
        for idx, (pos, direction) in enumerate(zip(agent_positions, agent_directions)):
            color = colormap(idx)  # Assign color to each agent
            # 绘制观测区域
            observation_angle = env.observation_angle
            start_angle = np.degrees(direction - observation_angle / 2)
            end_angle = np.degrees(direction + observation_angle / 2)
            wedge = patches.Wedge(pos, env.observation_radius, start_angle, end_angle, color='orange', alpha=0.01, label='Observation')
            ax.add_patch(wedge)
            # 绘制智能体位置和方向
            agent_circle = Circle(pos, radius=env.radius_exp, edgecolor=color, fill=False, alpha=0.5, linestyle='dashed')  # 使用圆形表示USV边界
            ax.add_patch(agent_circle)
            agent_ellipse = Ellipse(pos, width=env.agent_radius * 2, height=env.agent_radius,
                                    angle=np.degrees(direction), color=color, fill=True, alpha=0.3, linewidth=1.5, label='Agent')
            ax.add_patch(agent_ellipse)
            # 绘制方向箭头
            arrow_dx = np.cos(direction) * env.agent_radius
            arrow_dy = np.sin(direction) * env.agent_radius
            agent_arrow = Arrow(pos[0], pos[1], arrow_dx, arrow_dy, width=0.01 * env.area_size, color=color)
            ax.add_patch(agent_arrow)
            if step == 0:
                ax.text(pos[0], pos[1], str(idx + 1), color=color, fontsize=12, ha='center', va='center')
        if step == 0:
            # Plot goals
            for idx, pos in enumerate(goal_positions):
                goal_circle = Circle(pos, env.goal_radius, color='green', fill=True, alpha=0.3, label='Goal')
                ax.add_patch(goal_circle)
                ax.text(pos[0], pos[1], str(idx + 1), color='green', fontsize=12, ha='center', va='center')
            # Plot obstacles
            for idx, (pos, radius) in enumerate(zip(obstacle_positions, env.obstacle_radii)):
                obstacle_circle = Circle(pos, radius, color='red', fill=True, alpha=0.3, linewidth=1.5, label='Obstacle')
                ax.add_patch(obstacle_circle)
                ax.text(pos[0], pos[1], str(idx + 1), color='red', fontsize=12, ha='center', va='center')
        # Connect positions with arrows
        # if step > 0:
        #     prev_agent_positions = positions_history[step - 1][0]
        #     for prev_pos, current_pos in zip(prev_agent_positions, agent_positions):
        #         arrow = FancyArrowPatch(prev_pos, current_pos, arrowstyle='->', color=color, linewidth=1.5, mutation_scale=10)
        #         ax.add_patch(arrow)

    plt.grid(True)
    plt.legend(handles=[goal_circle, obstacle_circle, wedge], loc='best')
    # plt.savefig("USV_MATD3_observation_R.pdf", format='pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    np.random.seed(100)
    area_size = 100
    num_agents = 5

    output_folder = "output_folder"
    # 确保文件夹存在，如果不存在则创建它
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    scenario = 1  # 场景选择

    if scenario == 1:
        # 圆形场景
        agent_positions = np.array(env_utils.generate_circle_points((0.5, 0.5), 0.4, num_agents, 0)) * area_size  # 生成圆周上的点
        agent_directions = np.random.uniform(-np.pi, np.pi, num_agents)  # 初始化在 [-pi, pi] 范围内的随机方向
        goal_positions = np.array(env_utils.generate_circle_points((0.5, 0.5), 0.43, num_agents, np.pi)) * area_size  # 生成圆周上的点
        obstacle_positions = np.array([[0.22, 0.55], [0.32, 0.29], [0.68, 0.34], [0.51, 0.70], [0.45, 0.46]]) * area_size
        obstacle_radii = np.array([0.06, 0.07, 0.08, 0.10, 0.05]) * area_size
    elif scenario == 2:
        # 随机场景
        agent_positions = np.random.rand(num_agents, 2) * (area_size - 6) + 3  # 在区域内随机生成位置
        agent_directions = np.random.uniform(-np.pi, np.pi, num_agents)  # 初始化在 [-pi, pi] 范围内的随机方向
        goal_positions = np.random.rand(num_agents, 2) * (area_size - 6) + 3  # 在区域内随机生成目标位置
        # 随机生成障碍物位置和半径
        obstacle_positions = np.array([[37.47956, 40.01902], [17.80810, 23.76942], [52.48623, 77.54314], [37.62525, 59.28054],
                                      [62.99419, 15.26003], [60.22967, 45.77663], [36.31880, 20.43453]])  # 随机生成7个障碍物的位置
        obstacle_radii = np.array([5.86804, 5.83305, 7.78506, 7.98987, 8.25650, 5.46028, 7.31749])  # 随机生成障碍物的半径
    elif scenario == 3:
        # 走廊场景
        # 智能体的位置随机分布在走廊的两端
        agent_positions = np.zeros((num_agents, 2))
        agent_positions[:, 0] = np.random.uniform(0.2 * area_size, 0.3 * area_size, num_agents)  # X轴范围内随机分布
        agent_positions[:, 1] = np.linspace(0.2 * area_size, 0.8 * area_size, num_agents)  # Y轴均匀分布
        # 智能体的方向初始化为朝向走廊的另一端
        agent_directions = np.zeros(num_agents)
        agent_directions[:] = 0.0  # 所有智能体朝向相同方向（X轴正方向）
        # 目标位置位于走廊的另一端
        goal_positions = np.zeros((num_agents, 2))
        goal_positions[:, 0] = np.random.uniform(0.7 * area_size, 0.8 * area_size, num_agents)  # X轴范围内随机分布
        goal_positions[:, 1] = np.linspace(0.8 * area_size, 0.2 * area_size, num_agents)  # Y轴均匀分布
        # 障碍物分布在走廊内部
        obstacle_positions = np.array(
            [[48.63408, 43.64160], [58.80060, 25.28443], [46.72224, 68.78313], [43.50821, 25.76275]])  # 随机生成7个障碍物的位置
        obstacle_radii = np.array([7.99422, 8.01902, 6.90972, 5.18238])  # 随机生成障碍物的半径

    env = Environment(num_agents=num_agents, area_size=area_size,
                      agent_positions=agent_positions, agent_directions=agent_directions, goal_positions=goal_positions,
                      obstacle_positions=obstacle_positions, obstacle_radii=obstacle_radii)
    # env = Environment(num_agents=num_agents, area_size=area_size,
    #                   agent_positions=None, agent_directions=None, goal_positions=None,
    #                   obstacle_positions=None, obstacle_radii=None)
    # env.reset()
    # render_fig = env.render(agent_index=0)
    # # 生成文件名
    # render_pdf_filename1 = f"scenario_{scenario}.pdf"
    # render_pdf_filename2 = f"scenario_{scenario}.png"
    # render_pdf_path1 = os.path.join(output_folder, render_pdf_filename1)
    # render_pdf_path2 = os.path.join(output_folder, render_pdf_filename2)
    # # 保存 render 函数生成的图像
    # render_fig.savefig(render_pdf_path2, dpi=300, bbox_inches='tight')
    # render_fig.savefig(render_pdf_path1, format='pdf', bbox_inches='tight')

    evaluate_model(env, n_episode=1, step_count=150)


