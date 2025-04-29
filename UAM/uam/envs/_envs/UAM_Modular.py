import os
import random
from copy import copy
from string import ascii_letters
from typing import List, Union

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from bluesky.tools.geo import qdrpos
from gym import spaces
from PIL import Image

from uam.agents.base import _Agent
from uam.envs._envs.Plane import *


COLORS = sns.color_palette("husl", 30)
WIDTH  = 1000
HEIGTH = 1000
MY_DPI = 100

class Destination:
    def __init__(self, dt, c) -> None:
        # 尺寸
        self.radius          = c * 200  # [m]
        self.restricted_area = c * 200  # [m]
        self.spawn_radius    = c * 1000 # [m]
        self.respawn_radius  = c * 1200 # [m]
        
        # 位置
        self.lat = 60  # 度
        self.lon = 9  # 度
        self.N, self.E, _ = to_utm(self.lat, self.lon) # 米

        # timing
        self.dt = dt             # [s],模拟步长
        self._t_close = 60       # [s],已关闭时间
        self._t_nxt_open = 0     # [s],下次开放时间
        self._t_open_since = 0   # [s],已开放时间
        self._was_open = True
        self.open()

    def reset(self):
        self.open()

    def step(self, planes: List[Plane]):
        just_opened = False

        # 计算下次开放的时间
        if self._is_open is False:
            self._t_nxt_open -= self.dt
            if self._t_nxt_open <= 0:
                self.open()
                just_opened = True
        else:
            self._t_open_since += self.dt

        # store opening status
        self._was_open = copy(self._is_open)

        # 检查哪个飞行器进入了一个封闭或开放的目的地
        entered_close = np.zeros(len(planes), dtype=bool)
        entered_open  = np.zeros(len(planes), dtype=bool)

        for i, p in enumerate(planes):
            if p.D_dest <= self.radius:            
                if self._is_open:
                    entered_open[i] = True
                else:
                    entered_close[i] = True

        #  close only if the correct AC entered
        for i, p in enumerate(planes):
            if entered_open[i] and p.fly_to_goal == 1.0:
                self.close()
        return entered_close, entered_open, just_opened

    def open(self):
        self._t_open_since = 0
        self._t_nxt_open = 0
        self._is_open = True
        self.color = "green"
    
    def close(self):
        self._t_open_since = 0
        self._is_open = False
        self._t_nxt_open = copy(self._t_close)
        self.color = "red"

    @property
    def t_nxt_open(self):
        return self._t_nxt_open

    @property
    def t_close(self):
        return self._t_close

    @property
    def t_open_since(self):
        return self._t_open_since

    @property
    def is_open(self):
        return self._is_open

    @property
    def was_open(self):
        return self._was_open

class Path:
    """设定飞行器路径"""
    def __init__(self, lat=None, lon=None, north=None, east=None) -> None:
        # 检查
        assert not ((lat is None and lon is None) and (north is None and east is None)), "Need some data to construct the path."

        # UTM and lat-lon
        if (lat is None) or (lon is None):
            self._store(north, "north")
            self._store(east, "east")
            self.lat, self.lon = to_latlon(north=self.north, east=self.east, number=32)
        else:
            self._store(lat, "lat")
            self._store(lon, "lon")

        # UTM coordinates
        if (north is None) or (east is None):
            self.north, self.east, _ = to_utm(lat=self.lat, lon=self.lon)
        else:
            self._store(north, "north")
            self._store(east, "east")

    def _store(self, data:Union[list,np.ndarray], name:str):
        """Stores data by transforming to np.ndarray."""
        if not isinstance(data, np.ndarray):
            setattr(self, name, np.array(data))
        else:
            setattr(self, name, data)

    def _reverse(self):
        """Reverse the path."""
        self.north = np.flip(self.north)
        self.east  = np.flip(self.east)
        self.lat   = np.flip(self.lat)
        self.lon   = np.flip(self.lon)

class UAM_Modular(gym.Env):
    """Urban air mobility simulation env based on the BlueSky simulator of Ellerbroek and Hoekstra."""
    def __init__(self, 
                 N_agents_max :int, 
                 N_cutters_max :int,
                 w_coll:float, 
                 w_goal:float,
                 w_comf:float,
                 r_goal_norm:float,
                 c:float,
                 N_agents_min :int = 1,
                 N_cutters_min : int = 0,
                 curriculum : bool = False):
        super(UAM_Modular, self).__init__()

        # 新增存储列表
        self.N_incidents = 0  # 事件计数器
        self.N_accidents = 0  # 事故计数器
        self.N_entered_closed = 0  # 进入关闭的垂直起降区计数器
        self.planes_with_incidents = set()
        self.planes_with_accidents = set()
        self.planes_entered_closed = set()
        self.flight_durations = []
        self.unique_id_counter = 0  # 唯一ID计数器
        # setup
        self.curriculum = curriculum
        
        if curriculum:
            self.N_agents_max = 25
        else:
            self.N_agents_max  = N_agents_max
        
        self.N_agents_min  = N_agents_min
        self.N_cutters_max = N_cutters_max
        self.N_cutters_min = N_cutters_min
        
        assert min([N_agents_min, N_agents_max]) >= 1, "Please at least on RL-controlled aircraft."
        #assert 0.5 <= c <= 1.0, "The scaling parameter c should be in [0.5, 1.0]."

        self.acalt = 200 # [m]
        self.actas = 10  # [m/s]
        self.delta_tas = 3 # [m/s]
        self.actype = "MAVIC"

        self.w_coll = w_coll
        self.w_goal = w_goal
        self.w_comf = w_comf
        self.r_goal_norm = r_goal_norm
        self.w = self.w_coll + self.w_goal + self.w_comf
        self.c = c

        # 事故和事件
        self.incident_dist = 80 # [m]
        self.accident_dist = 10  # [m]
        self.clock_degs = np.linspace(0.0, 360.0, num=100, endpoint=True)

        # destination
        self.dt = 1.0
        self.dest = Destination(dt=self.dt, c=c)

        # 飞行器性能模块
        self.perf = OpenAP(self.actype, self.actas, self.acalt)

        # 配置
        self.history_length = 2
        self.OS_obs     = 3     # abs bng goal, rel bng goal, dist goal, fly to goal
        self.obs_per_TS = 6     # 距离,相对方位,速度差,航向交叉,DCPA,TCPA
        self.obs_size   = self.OS_obs + self.obs_per_TS*(self.N_agents_max-1) + self.obs_per_TS*self.N_cutters_max

        self.observation_space = spaces.Box(low  = np.full(self.obs_size, -np.inf, dtype=np.float32), 
                                            high = np.full(self.obs_size,  np.inf, dtype=np.float32))
        self.act_size = 1
        self.action_space = spaces.Box(low  = np.full(self.act_size, -1.0, dtype=np.float32), 
                                        high = np.full(self.act_size, +1.0, dtype=np.float32))
        self._max_episode_steps = 250

        # 可视化部分
        self.plot_reward = True
        self.plot_state  = False
        self.prod_video  = True

        assert not (self.plot_state and self.N_agents_max > 2), "State plotting is only reasonable for two flight taxis."
        if self.plot_state:
            atts = ["D_TS", "bng_TS", "V_R", "C_T", "DCPA", "TPCA", "Goal"]

            other_names = []
            for i in range(self.N_agents_max-1):
                others = [ele + ascii_letters[i] for ele in atts]
                other_names += others

            self.obs_names = ["bng_goal", "d_goal", "fly to goal"] + other_names
    def seed(self, seed=None):
        """Sets the seed for the environment's random number generators."""
        # Seed Python's built-in random module
        random.seed(seed)
        # Seed NumPy's random module (used for most sampling in the env)
        np.random.seed(seed)
        # Seed the action and observation space random number generators
        # (Not always necessary with modern gym, but good practice for older interfaces)
        if self.action_space is not None:
            self.action_space.seed(seed)
        if self.observation_space is not None:
            self.observation_space.seed(seed)
        
        # 你可能还需要为环境内部使用的其他特定随机对象设置种子（如果存在）
        # e.g., self.my_random_object.seed(seed)
        # Gym's standard practice is to return a list of seeds used.
        return [seed]
    def reset(self):
        """重设环境为初始状态"""
        self.step_cnt = 0           # 模拟步骤
        self.sim_t    = 0           # 模拟时间
        self.N_accs = 0             # 事故数
        self.N_incs = 0             # 事件数
        self.N_enterances_closed_d =0      # 进入关闭的vtol区数
        #创建飞行器
        self.planes:List[Plane] = []
        
        if self.curriculum:
            if 0 <= self.step_cnt <= 1e6:
                self.N_RL = random.randint(a=18, b=25)
            
            elif 1e6 < self.step_cnt <= 1.5e6:
                self.N_RL = random.randint(a=18, b=30)
            
            else:
                self.N_RL = random.randint(a=18, b=30)
        else:
            self.N_RL = random.randint(a=self.N_agents_min, b=self.N_agents_max)

        self.N_cutters = random.randint(self.N_cutters_min, self.N_cutters_max)
        self.N_planes = self.N_RL + self.N_cutters

        for n in range(self.N_planes):
            if n < self.N_RL:
                role = "RL" 
            else:
                role = "CUT"

            self.planes.append(self._spawn_plane(role))
      
        # interface to high-level module including goal decision
        self._high_level_control()

        # reset dest
        self.dest.reset()

        # init state
        self._set_state()
        self.state_init = self.state
        return self.state

    def _spawn_plane(self, role:str):
        assert role in ["RL", "VFG", "RND", "CUT"], "Unknown role."

        if role == "CUT":
            p0 = self.planes[0]

            while True:
                # sample time
                dt = self.c * random.uniform(30.0, 45.0)

                # linear prediction of p0's path
                vE, vN = xy_from_polar(r=p0.tas, angle=dtr(p0.hdg))
                x0_E = p0.e + vE * dt
                x0_N = p0.n + vN * dt

                # sample angle and speed
                if bool(random.getrandbits(1)):
                    ang = random.uniform(0.0, 2*np.pi)  # all random
                else:
                    ang = angle_to_2pi(dtr(p0.hdg) + np.random.uniform(-np.pi/6, np.pi/6)) # head-on
                tas = random.uniform(self.actas - self.delta_tas, self.actas + self.delta_tas)
                d = dt * tas

                # set position and heading
                E_add, N_add = xy_from_polar(r=d, angle=ang)
                lat, lon = to_latlon(north=x0_N + N_add, east=x0_E + E_add, number=32)
                hdg = (rtd(ang) + 180) % 360

                if ED(N0=x0_N + N_add, E0=x0_E + E_add, N1=self.dest.N, E1=self.dest.E) <= self.dest.respawn_radius:
                    break
        else:
            # 采样速度和方位
            tas = random.uniform(self.actas - self.delta_tas, self.actas + self.delta_tas)
            qdr = random.uniform(0.0, 360.0)

            # determine origin
            E_add, N_add = xy_from_polar(r=self.dest.spawn_radius, angle=dtr(qdr))
            lat, lon = to_latlon(north=self.dest.N+N_add, east=self.dest.E+E_add, number=32)

            # 添加航向噪声
            hdg = (qdr + 180) % 360
            sgn = 1 if bool(random.getrandbits(1)) else -1
            hdg = (hdg + sgn * random.uniform(20.0, 45.0)) % 360

        # construct plane
        p = Plane(role=role, dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=hdg, tas=tas,plane_id=self.unique_id_counter)
        p.spawn_time = self.sim_t  # 当前模拟时间记录为生成时间
        p.land_time = None  # 初始降落时间为空

        # set UTM coordinates
        p.n, p.e, _ = to_utm(lat=lat, lon=lon)

        # 计算到目的地的初始距离
        p.D_dest     = ED(N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e)
        p.D_dest_old = copy(p.D_dest)
        self.unique_id_counter += 1  # 每次生成递增ID
        return p

    def _high_level_control(self):
        """决定脱离机群飞向目的地的飞行器"""
        for i, _ in enumerate(self.planes):
            if i == 0 and self.step_cnt >= 200:
                self.planes[i].fly_to_goal = 1.0
            else:
                self.planes[i].fly_to_goal = -1.0

    def _set_state(self):
        if len(self.planes) == 0:
            self.state = None
            return

        # state observation of id0 will be used for learning
        self.state = self._get_state(0)

        for i, p in enumerate(self.planes):

            # 计算当前状态
            if i == 0:
                p.s = self.state
            else:
                p.s = self._get_state(i)

            # 更新历史信息
            if not hasattr(p, "s_hist"):
                p.s_hist = np.zeros((self.history_length, self.obs_size))
                p.hist_len = 0
            else:
                if p.hist_len == self.history_length:
                    p.s_hist = np.roll(p.s_hist, shift=-1, axis=0)
                    p.s_hist[self.history_length - 1] = p.s_old
                else:
                    p.s_hist[p.hist_len] = p.s_old
                    p.hist_len += 1
            
            # 保存旧状态
            p.s_old = copy(p.s)

    def _get_state(self, i:int) -> np.ndarray:
        """Computes the state from the perspective of the i-th agent of the internal plane array."""

        # select plane of interest
        p = self.planes[i]

        # relative bearing to goal, distance, fly to goal
        rel_bng_goal = bng_rel(N0=p.n, E0=p.e, N1=self.dest.N, E1=self.dest.E, head0=dtr(p.hdg), to_2pi=False)/np.pi
        d_goal   = ED(N0=p.n, E0=p.e, N1=self.dest.N, E1=self.dest.E)/self.dest.spawn_radius
        task     = p.fly_to_goal
        s_i = np.array([rel_bng_goal, d_goal, task])

        # 其他飞行器的信息
        TS_info = []
        for j, other in enumerate(self.planes):
            if i != j:
                # 对其他飞行器的位置添加高斯噪声
                noise_n = np.random.normal(0, 0)  # 北分量初始标准差为0
                noise_e = np.random.normal(0, 0)  # 东分量初始标准差为0
                noisy_n = other.n + noise_n
                noisy_e = other.e + noise_e

                # 相对速度
                v_r = (other.tas - p.tas) / (2 * self.delta_tas)

                # 相对方位，使用添加噪声后的位置计算
                bng = bng_rel(N0=p.n, E0=p.e, N1=noisy_n, E1=noisy_e, head0=dtr(p.hdg), to_2pi=False) / np.pi

                # 距离，使用添加噪声后的位置计算
                d = ED(N0=p.n, E0=p.e, N1=noisy_n, E1=noisy_e) / self.dest.spawn_radius

                # heading intersection
                C_T = angle_to_pi(dtr(other.hdg - p.hdg)) / np.pi

                # CPA ，使用添加噪声后的位置计算
                DCPA, TCPA = cpa(NOS=p.n, EOS=p.e, NTS=noisy_n, ETS=noisy_e, chiOS=np.radians(p.hdg),
                                 chiTS=dtr(other.hdg),
                                 VOS=p.tas, VTS=other.tas)
                DCPA = DCPA / 100.0
                TCPA = TCPA / 60.0

                # 综合
                TS_info.append([d, bng, v_r, C_T, DCPA, TCPA])

        # no TS is in sight: pad a 'ghost ship' to avoid confusion for the agent
        if len(TS_info) == 0:
            TS_info.append([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])

        # sort array according to distance
        TS_info = np.hstack(sorted(TS_info, key=lambda x: x[0], reverse=True)).astype(np.float32)

        # pad NA's as usual in single-agent LSTMRecTD3
        desired_length = self.obs_per_TS * (self.N_agents_max-1) + self.obs_per_TS * self.N_cutters_max
        TS_info = np.pad(TS_info, (0, desired_length - len(TS_info)), 'constant', constant_values=np.nan).astype(np.float32)

        s_i = np.concatenate((s_i, TS_info))
        return s_i

    def step(self, agent: _Agent): # MODIFIED: Changed input from 'a' to 'agent'
        """Takes an agent object and performs one step in the environment.
        Returns new_state (for agent 0), r (for agent 0), done, {}."""
        # increase step cnt and overall simulation time
        #if self.step_cnt % 60 == 0:
           # self.save_screenshot()
        # 如果飞行器完成任务并记录了起飞和降落时间
        self.step_cnt += 1
        self.sim_t += self.dt
        # 确保每个飞机都有 spawn_time 和 land_time 属性
        for p in self.planes:
            if not hasattr(p, "spawn_time"):
                p.spawn_time = self.sim_t
            if not hasattr(p, "land_time"):
                p.land_time = None
        # Initialize action for agent 0 (needed for comfort reward calculation)
        a0 = 0.0
        # Iterate through all planes
        for i, p in enumerate(self.planes):
            # Control the plane based on its role
            if p.role == "RL":
                # Use the provided agent object to select an action for the current plane's state and history
                act = agent.select_action(s        = p.s,
                                          s_hist   = p.s_hist,
                                          a_hist   = None,  # Assuming a_hist is not used by this agent type
                                          hist_len = p.hist_len)
                # Update the plane's dynamics using the selected action
                p.upd_dynamics(a=act, discrete_acts=False, perf=self.perf, dest=None)
                # Store the action of plane 0 for comfort reward calculation
                if i == 0:
                     # Assuming the action is a 1D numpy array or scalar
                    a0 = act[0] if isinstance(act, np.ndarray) and act.ndim > 0 else act
            elif p.role == "CUT":
                 # Update CUT planes without specific control input (they follow their internal logic or simple dynamics)
                 p.upd_dynamics(perf=self.perf, dest=None)
            else:
                # Handle other roles like VFG or RND if implemented
                # For now, assume they also just update dynamics without external action
                p.upd_dynamics(perf=self.perf, dest=None)
        # update UTM coordinates after all planes have moved
        for p in self.planes:
            p.n, p.e, _ = to_utm(lat=p.lat, lon=p.lon)
        # update distances to destination after all planes have moved
        for p in self.planes:
            p.D_dest_old = copy(p.D_dest)
            p.D_dest = ED(N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e)
        # check destination entries and update destination status
        entered_close, entered_open, just_opened = self.dest.step(self.planes)
        # 记录降落时间
        for i, p in enumerate(self.planes):
            if entered_open[i] and p.fly_to_goal == 1.0 and p.land_time is None:
                p.land_time = self.sim_t

        # 收集飞行时长
        for p in self.planes:
            if p.fly_to_goal == 1.0 and p.spawn_time is not None and p.land_time is not None:
                self.flight_durations.append(p.land_time - p.spawn_time)
                # 防止重复记录
                p.spawn_time = None
                p.land_time = None

        # 统计进入关闭的垂直起降区的次数
        for i, entered in enumerate(entered_close):
            if entered:
                self.planes_entered_closed.add(i)  # 记录进入关闭区域的飞机索引
        # 记录降落时间
        for i, p in enumerate(self.planes):
            if entered_open[i] and p.fly_to_goal == 1.0 and p.land_time is None:
                p.land_time = self.sim_t

        # 检查事件和事故
        # 在step方法的事件检测部分：
        for i, pi in enumerate(self.planes):
            for j, pj in enumerate(self.planes):
                if i < j:
                    D = latlondist(latd1=pi.lat, lond1=pi.lon, latd2=pj.lat, lond2=pj.lon)
                    if D <= self.accident_dist:
                        print(f"Accident between Plane {pi.id} and {pj.id}, D={D}")
                        self.planes_with_accidents.add((pi.id,pj.id))
                    elif D <= self.incident_dist:
                        print(f"Incident between Plane {pi.id} and {pj.id}, D={D}")
                        self.planes_with_incidents.add((pi.id,pj.id))
        # 在重生之前计算奖励（可能使用 a0 来获得舒适奖励）
        self._calculate_reward(a0)
        # handle respawning based on destination entry and map boundaries
        self._handle_respawn(entered_open)
        # ---- 验证场景 ----
        # 飞机后续生成设定
        if "Validation" in type(self).__name__:
            if self.id_counter < self.N_agents_max:
                if self.sim_study:
                    if self.sim_t > 0 and self.sim_t % 30 == 0:
                        if len(self.planes) < self.N_agents_max:
                            rng = np.random.default_rng()
                            candidate_gates = list(range(8))
                            gate = rng.choice(candidate_gates)

                            def is_gate_clear(g):
                                temp_plane = self._spawn_plane(gate=g, noise=True)
                                for other in self.planes:
                                    if ED(N0=temp_plane.n, E0=temp_plane.e, N1=other.n, E1=other.e) < 400.0:
                                        return False
                                return True

                            # 先尝试随机点
                            if is_gate_clear(gate):
                                p = self._spawn_plane(gate=gate, noise=True)
                            else:
                                # 遍历所有进近点
                                p = None
                                for g in candidate_gates:
                                    if is_gate_clear(g):
                                        p = self._spawn_plane(gate=g, noise=True)
                                        print(f"随机选择的进近点 {gate} 距离过近，改为使用进近点 {g}")
                                        break

                            if p is not None:
                                p.id = self.unique_ids[self.id_counter]
                                self.planes.append(p)
                                self.N_planes = len(self.planes)
                                self.id_counter += 1
                elif self.situation == 1:
                    # 初始化集群信息
                    if not hasattr(self, 'cluster_info'):
                        self.cluster_info = []
                        for i in range(8):
                            # 每个集群信息包括：起始时间、已生成飞机数、总飞机数
                            cluster_start_time = 30 + i * 60
                            rng = np.random.default_rng()  # 创建一个局部随机数生成器
                            num_planes = rng.poisson(3)
                            remaining_slots = self.N_agents_max - self.id_counter
                            num_planes = min(num_planes, remaining_slots)
                            self.cluster_info.append({
                                'start_time': cluster_start_time,
                                'generated_planes': 0,
                                'total_planes': num_planes
                            })

                    # 遍历所有集群
                    for cluster_idx, cluster in enumerate(self.cluster_info):
                        start_time = cluster['start_time']
                        generated_planes = cluster['generated_planes']
                        total_planes = cluster['total_planes']

                        # 检查是否到了该集群开始生成飞机的时间
                        if self.sim_t >= start_time and generated_planes < total_planes:
                            # 计算当前飞机的进入时间
                            entry_time = start_time + generated_planes * 30
                            if self.sim_t == entry_time:
                                # 每个集群使用不同的进近点（gate 0 - 7）
                                cluster_gate = cluster_idx % 8
                                # 检查是否距离已有飞机过近
                                for i in range(8):  # 最多尝试8次
                                    # 检查是否距离已有飞机过近
                                    too_close = False
                                    t_close = False
                                    for other in self.planes:
                                        p = self._spawn_plane(gate=cluster_gate, noise=False)
                                        if ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e) < 320.0:
                                            too_close = True
                                            break
                                    if too_close:
                                        for other in self.planes:
                                            p = self._spawn_plane(gate=i, noise=False)
                                            if ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e) < 320.0:
                                                t_close = True
                                                break
                                    if t_close:
                                        print("[Skip] 距离已有飞机过近，重试...")
                                        continue  # 合法使用，跳过这轮，进入下一轮尝试
                                    p.id = self.unique_ids[self.id_counter]
                                    self.planes.append(p)
                                    self.id_counter += 1
                                    self.N_planes = len(self.planes)
                                    # 更新该集群已生成的飞机数量
                                    self.cluster_info[cluster_idx]['generated_planes'] += 1
                                    break

                elif self.situation == 2:
                    # 初始化集群信息
                    if not hasattr(self, 'cluster_info'):
                        self.cluster_info = []
                        for i in range(8):
                            rng = np.random.default_rng()  # 创建一个局部随机数生成器
                            num_planes = rng.poisson(3)
                            remaining_slots = self.N_agents_max - self.id_counter
                            num_planes = min(num_planes, remaining_slots)
                            self.cluster_info.append({
                                'start_time': None,
                                'generated_planes': 0,
                                'total_planes': num_planes
                            })

                    for cluster_idx, cluster in enumerate(self.cluster_info):
                        if cluster_idx == 0:
                            if cluster['start_time'] is None:
                                cluster['start_time'] = self.sim_t + 30
                            start_time = cluster['start_time']
                            generated_planes = cluster['generated_planes']
                            total_planes = cluster['total_planes']
                            if generated_planes < total_planes:
                                entry_time = start_time + generated_planes * 30
                                if self.sim_t == entry_time:
                                    cluster_gate = 4
                                    p = self._spawn_plane(gate=cluster_gate, noise=False)
                                    # 检查是否距离已有飞机过近
                                    too_close = False
                                    t_close = False
                                    for other in self.planes:
                                        if ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e) < 320.0:
                                            too_close = True
                                            break
                                    if too_close:
                                        for i in range(8):
                                            for other in self.planes:
                                                p = self._spawn_plane(gate=i, noise=False)
                                                if ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e) < 320.0:
                                                    t_close = True
                                                    break
                                    if t_close:
                                        print("无其他安全进近口可进，从邻近进近点进入")
                                        p = self._spawn_plane(gate=5, noise=False)
                                    p.id = self.unique_ids[self.id_counter]
                                    self.planes.append(p)
                                    self.id_counter += 1
                                    self.N_planes = len(self.planes)
                                    cluster['generated_planes'] += 1
                        else:
                            prev_cluster = self.cluster_info[cluster_idx - 1]
                            # 确保前一个集群完成
                            if prev_cluster['generated_planes'] == prev_cluster['total_planes']:
                                # 计算下一个集群的起始时间
                                if cluster['start_time'] is None:
                                    if cluster_idx == 1:
                                        # 第0波和第1波间隔45秒
                                        cluster['start_time'] = prev_cluster['start_time'] + prev_cluster[
                                            'total_planes'] * 30 + 59
                                    else:
                                        # 其他集群间隔60秒
                                        cluster['start_time'] = prev_cluster['start_time'] + prev_cluster[
                                            'total_planes'] * 30 + 60
                                    print(f"Cluster {cluster_idx} start time: {cluster['start_time']}")
                                start_time = cluster['start_time']
                                generated_planes = cluster['generated_planes']
                                total_planes = cluster['total_planes']
                                if generated_planes < total_planes:
                                    entry_time = start_time + generated_planes * 30
                                    if self.sim_t == entry_time:
                                        cluster_gate = 4
                                        p = self._spawn_plane(gate=cluster_gate, noise=False)
                                        # 检查是否距离已有飞机过近
                                        too_close = False
                                        t_close = False
                                        for other in self.planes:
                                            if ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e) < 320.0:
                                                too_close = True
                                                break
                                        if too_close:
                                            for i in range(8):
                                                for other in self.planes:
                                                    p = self._spawn_plane(gate=i, noise=False)
                                                    if ED(N0=p.n, E0=p.e, N1=other.n, E1=other.e) < 320.0:
                                                        t_close = True
                                                        break
                                        if t_close:
                                            print("无其他安全进近口可进，从邻近进近点进入")
                                            p = self._spawn_plane(gate=5, noise=False)
                                        p.id = self.unique_ids[self.id_counter]
                                        self.planes.append(p)
                                        self.id_counter += 1
                                        self.N_planes = len(self.planes)
                                        cluster['generated_planes'] += 1
                else :
                    if self.sim_t % 30 == 0:
                        for _ in range(4):
                            p = self._spawn_plane(gate=self.id_counter % 4, noise=False)
                            p.id = self.unique_ids[self.id_counter]
                            self.planes.append(p)
                            self.id_counter += 1
                            self.N_planes = len(self.planes)
        # ---- End Validation Specific Logic ----
        # high-level control: decide which plane should fly to the goal next
        if "Validation" in type(self).__name__:
             if just_opened:
                 self._high_level_control()
        else:
             self._high_level_control()
        # compute the next state based on the updated plane positions/states
        self._set_state() # This will set self.state based on plane 0 after updates
        # determine if the episode is done
        d = self._done()
        # logging (if logger is initialized)
        if hasattr(self, "logger"):
            P_info = {}
            # ... (rest of the logging code remains the same) ...
            for id in self.unique_ids:
                try:
                    # Find the index of the plane with the matching id
                    plane_indices = [idx for idx, p in enumerate(self.planes) if hasattr(p, 'id') and p.id == id]
                    if plane_indices:
                        i = plane_indices[0]
                        p = self.planes[i]
                        n = p.n
                        e = p.e
                        hdg = p.hdg
                        tas = p.tas
                        goal = int(p.fly_to_goal) if hasattr(p, 'fly_to_goal') else None
                    else: # Plane with this ID might have despawned
                         n, e, hdg, tas, goal = None, None, None, None, None
                except Exception as e: # Catch potential errors during logging
                    # print(f"Logging error for ID {id}: {e}") # Optional: print error
                    n, e, hdg, tas, goal = None, None, None, None, None
                P_info[f"P{id}_n"] = n
                P_info[f"P{id}_e"] = e
                P_info[f"P{id}_hdg"] = hdg
                P_info[f"P{id}_tas"] = tas
                P_info[f"P{id}_goal"] = goal
            self.logger.store(sim_t=self.sim_t, **P_info)
        # Return state of agent 0, reward of agent 0, done flag, and info dict
        # Ensure self.state and self.r[0] are valid even if plane 0 despawned in this step
        current_state = self.state if self.state is not None else np.zeros(self.obs_size) # Return zero state if no planes left
        current_reward = float(self.r[0]) if len(self.r) > 0 else 0.0 # Return zero reward if no planes left
        return current_state, current_reward, d, {}

    def _is_plane_spawned(self, cluster_idx: int, plane_idx: int) -> bool:
        """检查某集群的某飞机是否已生成"""
        key = f"cluster_{cluster_idx}_plane_{plane_idx}"
        return hasattr(self, key) and getattr(self, key)

    def _mark_plane_spawned(self, cluster_idx: int, plane_idx: int):
        """标记某集群的某飞机已生成"""
        key = f"cluster_{cluster_idx}_plane_{plane_idx}"
        setattr(self, key, True)


    def _handle_respawn(self, entered_open:np.ndarray):
        """Respawns planes when they left the map."""
        for i, p in enumerate(self.planes):
            r = False

            # check conditions
            if p.D_dest >= self.dest.respawn_radius:
                r = True
            elif p.role == "CUT" and i != 0:
                p0 = self.planes[0]
                _, TCPA = cpa(NOS=p0.n, EOS=p0.e, NTS=p.n, ETS=p.e, chiOS=np.radians(p0.hdg), 
                              chiTS=dtr(p.hdg), VOS=p0.tas, VTS=p.tas)
                d = ED(N0=p0.n, E0=p0.e, N1=p.n, E1=p.e)
                if TCPA < 0 and d > (self.c * 400.0):
                    r = True

            # perform respawn
            if r:
                new_plane = self._spawn_plane(role=p.role)  # 自动分配新ID
                new_plane.fly_to_goal = p.fly_to_goal
                self.planes[i] = new_plane

    def _calculate_reward(self, a0:float):
        r_coll = np.zeros((self.N_planes, 1), dtype=np.float32)
        r_goal = np.zeros((self.N_planes, 1), dtype=np.float32)
        r_comf = np.zeros((self.N_planes, 1), dtype=np.float32)

        # ------ collision reward ------
        D_matrix = np.ones((len(self.planes), len(self.planes))) * np.inf
        for i, pi in enumerate(self.planes):
            for j, pj in enumerate(self.planes):
                if i != j and i == 0:
                    D_matrix[i][j] = ED(N0=pi.n, E0=pi.e, N1=pj.n, E1=pj.e)

        for i, pi in enumerate(self.planes):
            if i != 0:
                continue

            D = float(np.min(D_matrix[i]))

            if D <= self.accident_dist:
                r_coll[i] -= 10.0

            elif D <= self.incident_dist:
                r_coll[i] -= 10.0

            else:
                r_coll[i] -= 5*np.exp(-(D-self.incident_dist)**2/(160.4549)**2) 
                # approximately yields reward of -5 at 100m and -0.01 at 500m
                # b = function(x, y){
                # return(sqrt(-(x-100)^2/log(y/-5)))
                #}

            # off-map
            if pi.D_dest > self.dest.spawn_radius: 
                r_coll[i] -= 5.0

        # ------ goal reward ------
        for i, p in enumerate(self.planes):

            if i != 0:
                continue
            
            # goal-approach reward for the one who should fly toward the goal
            if p.fly_to_goal == 1.0:
                r_goal[i] = (p.D_dest_old - p.D_dest)/self.r_goal_norm
            
            # punish others for getting into the restricted area
            elif p.D_dest <= self.dest.restricted_area:
                r_goal[i] = -5.0

        #--------------- comfort reward --------------------
        r_comf[0] = -(a0)**4

        # aggregate reward components
        if self.w == 0.0:
            r = np.zeros((self.N_planes, 1), dtype=np.float32)
        else:
            r = (self.w_coll*r_coll + self.w_goal*r_goal + self.w_comf*r_comf)/self.w

        # store
        self.r = r
        self.r_coll = r_coll
        self.r_goal = r_goal
        self.r_comf = r_comf

    def _done(self):
        d = False

        # id-0 成功到达目标区域
        if self.planes[0].D_dest <= self.dest.radius and self.planes[0].fly_to_goal == 1:
            d = True

        # 人工终止信号
        if self.step_cnt >= self._max_episode_steps:
            d = True

        if d:
            # 释放视频写入器
            if hasattr(self, "video_writer"):
                self.video_writer.release()
        return d

    def __str__(self):
        avg_time = np.mean(self.flight_durations) if self.flight_durations else 0
        return f"Step: {self.step_cnt}, Sim-Time [s]: {int(self.sim_t)}, Avg-Time [s]: {int(avg_time)}, Flight-Number: {self.N_planes}" + "\n" + \
            f"Time-to-open [s]: {int(self.dest.t_nxt_open)}, Time-since-open [s]: {int(self.dest.t_open_since)}" + "\n" + \
            f"Incidents: {len(self.planes_with_incidents)}, Accidents: {len(self.planes_with_accidents)}, Enter-Closed: {len(self.planes_entered_closed)}"

    def render(self, mode=None):
        """Renders the current environment."""

        # Plot every nth timestep
        if self.step_cnt % 1 == 0:
            
            # Initialize the figure
            if len(plt.get_fignums()) == 0:
                self.initialize_figure()

            # Storage
            if self.plot_reward:
                self.store_reward()

            if self.plot_state:
                self.store_state()

            # Periodically clear and initialize
            if self.step_cnt % 19 == 0:
                self.clear_and_initialize_plots()

            else:
                # Restore the background
                self.restore_background()

                # Update animated artists
                self.update_animated_artists()

                # Show on screen
                self.show_on_screen()

            # Update video
            if self.prod_video:
                self.update_video()
            else:
                plt.pause(0.01)

    def initialize_figure(self):
        """Initialize the figure and subplots."""
        if len(plt.get_fignums()) == 0:
            sns.set()

            if self.plot_reward and self.plot_state:
                self.f = plt.figure(figsize=(WIDTH/MY_DPI, HEIGTH/MY_DPI), dpi=MY_DPI)
                self.gs  = self.f.add_gridspec(2, 2)
                self.ax1 = self.f.add_subplot(self.gs[:, 0]) # ship
                self.ax2 = self.f.add_subplot(self.gs[0, 1]) # reward
                self.ax3 = self.f.add_subplot(self.gs[1, 1]) # state

            elif self.plot_reward:
                self.f = plt.figure(figsize=(WIDTH/MY_DPI, HEIGTH/MY_DPI), dpi=MY_DPI)
                self.gs  = self.f.add_gridspec(1, 2)
                self.ax1 = self.f.add_subplot(self.gs[0, 0]) # ship
                self.ax2 = self.f.add_subplot(self.gs[0, 1]) # reward

            elif self.plot_state:
                self.f = plt.figure(figsize=(WIDTH/MY_DPI, HEIGTH/MY_DPI), dpi=MY_DPI)
                self.gs  = self.f.add_gridspec(1, 2)
                self.ax1 = self.f.add_subplot(self.gs[0, 0]) # ship
                self.ax3 = self.f.add_subplot(self.gs[0, 1]) # state

            else:
                self.f, self.ax1 = plt.subplots(1, 1, figsize=(WIDTH/MY_DPI, HEIGTH/MY_DPI), dpi=MY_DPI)
            plt.ion()
            #plt.show()
            if not self.prod_video:
                plt.show()

        # Initialize video writer
        if self.prod_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fps = 10.0
            self.video_writer = cv2.VideoWriter("output.avi", fourcc, fps, (WIDTH, HEIGTH))

    def store_reward(self):
        """Update the reward plot."""
        if self.step_cnt == 0:
            self.ax2.r      = np.zeros(self._max_episode_steps)
            self.ax2.r_coll = np.zeros(self._max_episode_steps)
            self.ax2.r_goal = np.zeros(self._max_episode_steps)
            self.ax2.r_comf = np.zeros(self._max_episode_steps)
        else:
            self.ax2.r[self.step_cnt] = self.r if isinstance(self.r, float) else float(self.r[0])
            self.ax2.r_coll[self.step_cnt] = self.r_coll if isinstance(self.r_coll, float) else float(self.r_coll[0])
            self.ax2.r_goal[self.step_cnt] = self.r_goal if isinstance(self.r_goal, float) else float(self.r_goal[0])
            self.ax2.r_comf[self.step_cnt] = self.r_comf if isinstance(self.r_comf, float) else float(self.r_comf[0])
    
    def store_state(self):
        """Update the state plot."""
        if self.step_cnt == 0:
            self.ax3.s = np.zeros((self.obs_size, self._max_episode_steps))
        else:
            self.ax3.s[:, self.step_cnt] = self.state

    def clear_and_initialize_plots(self):
        """Clear and initialize the plots."""
        # clearance
        self.ax1.clear()
        if self.plot_reward:
            self.ax2.clear()
        if self.plot_state:
            self.ax3.clear()

        # appearance
        self.ax1.set_title("Urban Air Mobility", fontdict={"size" : 16})
        self.ax1.set_xlabel("East", fontdict={"size" : 16})
        self.ax1.set_ylabel("North", fontdict={"size" : 16})
        self.ax1.set_ylim(6650087.934312163, 6652735.055772923 + 150)
        self.ax1.set_xticks([])
        self.ax1.set_yticks([])
        # 设置子图 self.ax1 的背景颜色为白色
        self.ax1.set_facecolor('white')
        # 设置子图 self.ax1 的边框颜色为黑色，宽度为 2（可根据需要调整宽度）
        self.ax1.spines['top'].set_edgecolor('gray')
        self.ax1.spines['bottom'].set_edgecolor('gray')
        self.ax1.spines['left'].set_edgecolor('gray')
        self.ax1.spines['right'].set_edgecolor('gray')
        self.ax1.spines['top'].set_linewidth(1.5)
        self.ax1.spines['bottom'].set_linewidth(1.5)
        self.ax1.spines['left'].set_linewidth(1.5)
        self.ax1.spines['right'].set_linewidth(1.5)


        if self.plot_reward:
            self.ax2.set_xlabel("Timestep in episode")
            self.ax2.set_ylabel("Reward of ID0")
            self.ax2.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
            self.ax2.set_ylim(-7, 7)

        if self.plot_state:
            self.ax3.set_xlabel("Timestep in episode")
            self.ax3.set_ylabel("State of Agent 0")
            self.ax3.set_xlim(0, 50*(np.ceil(self.step_cnt/50)+1))
            self.ax3.set_ylim(-2, 5)

        # ---------------- non-animated artists ----------------
        # spawning area
        lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.spawn_radius))\
            for deg in self.clock_degs]))
        ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
        self.ax1.plot(es, ns, color="purple", alpha=0.75)

        # restricted area
        lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.restricted_area))\
            for deg in self.clock_degs]))
        ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
        self.ax1.plot(es, ns, color="purple", alpha=0.75)

        # respawn area
        lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.respawn_radius))\
            for deg in self.clock_degs]))
        ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
        self.ax1.plot(es, ns, color="black", alpha=0.3)

        # vertiport text
        self.ax1.text(x=0.4625, y=0.48, s="Vertiport", fontdict={"size" : 12}, transform = self.ax1.transAxes)
        self.ax1.text(x=0.4475, y=0.45, s="VTOL Zone", fontdict={"size" : 12}, transform = self.ax1.transAxes)

        # ---------- animated artists: initial drawing ---------
        # step info
        self.ax1.info_txt = self.ax1.text(x=0.0, y=0.925, s="", fontdict={"size" : 14}, animated=True,
                                          transform = self.ax1.transAxes)

        # destination
        lats, lons = map(list, zip(*[qdrpos(latd1=self.dest.lat, lond1=self.dest.lon, qdr=deg, dist=meter_to_NM(self.dest.radius))\
            for deg in self.clock_degs]))
        ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
        self.ax1.dest_ln = self.ax1.plot(es, ns, color=self.dest.color, animated=True)[0]

        # aircraft information
        self.ax1.scs   = []
        self.ax1.lns   = []
        self.ax1.paths = []
        self.ax1.pts1  = []
        self.ax1.pts2  = []
        self.ax1.pts3  = []
        self.ax1.txts  = []

        for i in range(self.N_agents_max):
            try:
                hdg = self.planes[i].hdg
            except:
                hdg = 0

            color = COLORS[i]

            # show aircraft
            self.ax1.scs.append(self.ax1.scatter([], [], marker=(3, 0, -hdg), color=color, animated=True))

            # incident area
            self.ax1.lns.append(self.ax1.plot([], [], color=color, animated=True, zorder=10)[0])

            # information
            self.ax1.txts.append(self.ax1.text(x=0.0, y=0.0, s="", color=color, fontdict={"size" : 12}, animated=True))

        if self.plot_reward:
            self.ax2.lns_agg  = []
            self.ax2.lns_coll = []
            self.ax2.lns_goal = []
            self.ax2.lns_comf = []

            self.ax2.lns_agg.append(self.ax2.plot([], [], color=COLORS[0], label=f"Agg", animated=True)[0])
            self.ax2.lns_coll.append(self.ax2.plot([], [], color=COLORS[1], label=f"Collision", animated=True)[0])
            self.ax2.lns_goal.append(self.ax2.plot([], [], color=COLORS[2], label=f"Goal", animated=True)[0])
            self.ax2.lns_comf.append(self.ax2.plot([], [], color=COLORS[3], label=f"Comfort", animated=True)[0])
            self.ax2.legend()

        if self.plot_state:
            self.ax3.lns = []
            for i in range(self.obs_size):
                self.ax3.lns.append(self.ax3.plot([], [], label=self.obs_names[i], color=COLORS[i], animated=True)[0])
            self.ax3.legend()

        # ----------------- store background -------------------
        self.f.canvas.draw()
        self.ax1.bg = self.f.canvas.copy_from_bbox(self.ax1.bbox)
        if self.plot_reward:
            self.ax2.bg = self.f.canvas.copy_from_bbox(self.ax2.bbox)
        if self.plot_state:
            self.ax3.bg = self.f.canvas.copy_from_bbox(self.ax3.bbox)

    def update_video(self):
        # Capture the current figure as an image
        # Capture the current figure as an image
        width, height = self.f.canvas.get_width_height()
        buf = self.f.canvas.tostring_argb()
        img_arr = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)

        # Convert ARGB to RGB
        img_arr = img_arr[:, :, 1:]  # Remove the alpha channel

        img = Image.fromarray(img_arr, mode="RGB")

        # 假设你有一个视频写入器 self.video_writer
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.video_writer.write(img)

    def restore_background(self):
        """Restore the background."""
        self.f.canvas.restore_region(self.ax1.bg)
        if self.plot_reward:
            self.f.canvas.restore_region(self.ax2.bg)
        if self.plot_state:
            self.f.canvas.restore_region(self.ax3.bg)

    def update_animated_artists(self):
        """Update animated artists."""
        # step info
        self.ax1.info_txt.set_text(self.__str__())
        self.ax1.draw_artist(self.ax1.info_txt)

        # destination
        self.ax1.dest_ln.set_color(self.dest.color)
        self.ax1.draw_artist(self.ax1.dest_ln)

        for i, p in enumerate(self.planes):

            # show aircraft
            self.ax1.scs[i].set_offsets(np.array([p.e, p.n]))
            self.ax1.draw_artist(self.ax1.scs[i])

            # incident area
            lats, lons = map(list, zip(*[qdrpos(latd1=p.lat, lond1=p.lon, qdr=deg, dist=meter_to_NM(self.incident_dist/2))\
                for deg in self.clock_degs]))
            ns, es, _ = to_utm(lat=np.array(lats), lon=np.array(lons))
            self.ax1.lns[i].set_data(es, ns) 
            self.ax1.draw_artist(self.ax1.lns[i])

            # information
            #s = f"id: {i}" + "\n" + f"hdg: {p.hdg:.1f}" + "\n" + f"alt: {p.alt:.1f}" + "\n" + f"tas: {p.tas:.1f}"
            #if hasattr(p, "t_alive"):
            #    s+= "\n" + f"t_alive: {int(p.t_alive)}" 
            #if p.fly_to_goal == 1.0:
            #    s += "\n" + "Go!!!"
            
            if p.fly_to_goal == 1.0:
                s = "Go"
            else:
                s = ""
            self.ax1.txts[i].set_text(s)
            self.ax1.txts[i].set_position((p.e + 50, p.n))
            self.ax1.draw_artist(self.ax1.txts[i])

        # reward
        if self.plot_reward:
            self.ax2.lns_agg[0].set_data(np.arange(self.step_cnt+1), self.ax2.r[:self.step_cnt+1])
            self.ax2.lns_coll[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_coll[:self.step_cnt+1])
            self.ax2.lns_goal[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_goal[:self.step_cnt+1])
            self.ax2.lns_comf[0].set_data(np.arange(self.step_cnt+1), self.ax2.r_comf[:self.step_cnt+1])
                
            self.ax2.draw_artist(self.ax2.lns_agg[0])
            self.ax2.draw_artist(self.ax2.lns_coll[0])
            self.ax2.draw_artist(self.ax2.lns_goal[0])
            self.ax2.draw_artist(self.ax2.lns_comf[0])

        # state
        if self.plot_state:
            for i in range(self.obs_size):
                self.ax3.lns[i].set_data(np.arange(self.step_cnt+1), self.ax3.s[i][:self.step_cnt+1])
                self.ax3.draw_artist(self.ax3.lns[i])

    def show_on_screen(self):
        """Show the plots on the screen."""
        self.f.canvas.blit(self.ax1.bbox)
        if self.plot_reward:
            self.f.canvas.blit(self.ax2.bbox)
        if self.plot_state:
            self.f.canvas.blit(self.ax3.bbox)

    def save_screenshot(self):
        # 定义截图保存的文件夹
        screenshot_folder = "screenshots"
        if not os.path.exists(screenshot_folder):
            os.makedirs(screenshot_folder)

        buffer = self.f.canvas.buffer_rgba()
        # 将数据转换为 numpy 数组
        rgba_array = np.asarray(buffer)
        # 从 RGBA 数组中提取 RGB 通道
        rgb_array = rgba_array[:, :, :3]
        # 创建 PIL 图像对象
        img = Image.fromarray(rgb_array)
        frame = np.array(img)

        # Convert RGB to BGR (required for OpenCV)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # 生成截图文件名，使用当前时间步作为文件名的一部分
        screenshot_filename = f"screenshot_step_{self.step_cnt}.png"
        screenshot_path = os.path.join(screenshot_folder, screenshot_filename)
        cv2.imwrite(screenshot_path, frame)
        print(f"Saved screenshot at step {self.step_cnt} as {screenshot_path}")