import uuid

from uam.envs._envs.UAM_Modular import *


def dtr(angle):
    return angle * math.pi / 180
class UAMLogger:
    """实施轨迹存储，以实现城市空中交通项目的验证绘图."""
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, [value])

    def store(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                eval(f"self.{key}.append({value})")
            else:
                setattr(self, key, [value])

    def dump(self, name):
        data = vars(self)
        L = max([len(e) for e in data.values()])
        for key, value in data.items():
            if len(value) != L:
                data[key] += [None] * (L - len(value))

        df = pd.DataFrame(vars(self))
        df.replace(to_replace=[None], value=0.0, inplace=True) # clear None
        df.to_pickle(f"{name}.pkl")


class UAM_Modular_Validation(UAM_Modular):
    """城市空中交通代理的验证方案."""
    def __init__(self, situation:int, sim_study:bool, sim_study_N:int, safe_number:int):

        self.situation   = situation
        self.sim_study   = sim_study
        self.sim_study_N = sim_study_N
        self.safe_number = safe_number
        self.cluster_count = 0
        assert not (not sim_study and sim_study_N is not None), "Specify sim_study = True if you give number of agents for it."
        #assert situation == 1, "We have only situation 1 at the moment."

        if sim_study:
            N_agents_max = sim_study_N

        elif situation == 1:
            N_agents_max = 26 # 6波泊松分布，对应模拟研究部分，无法使用6*4
        elif situation == 2:
            N_agents_max = 26 # 6波泊松分布，对应模拟研究部分，无法使用6*4
        else:
            N_agents_max = 12

        super().__init__(N_agents_max=N_agents_max, N_cutters_max=0, w_coll=0.0, w_goal=0.0, w_comf=0.0, r_goal_norm=1.0, c=1.0)
        self._max_episode_steps = 100_000 if sim_study else 2000

        # viz
        self.plot_reward = False
        self.plot_state  = False

    def reset(self):
        """将环境重置为初始状态"""
        self.step_cnt = 0           # 模拟步骤计数器
        self.sim_t    = 0           # overall passed simulation time (in s)

        # 创建一些飞行器
        self.planes:List[Plane] = []

        if self.sim_study:
            self.N_planes = 1       # 仿真验证设置的飞行器数量
            noise = True
        elif self.situation == 1:
            self.N_planes = 1
            noise = False
        elif self.situation == 2:
            self.N_planes = 1
            noise = False
        elif self.situation == 3:
            self.N_planes = 4
            noise = False

        # 创建标识以进行日志记录
        self.unique_ids = [str(uuid.uuid4()).replace("-","_") for _ in range(self.N_agents_max)]
        self.id_counter = 0

        # 生成飞机
        for _ in range(self.N_planes):

            # create plane
            if self.sim_study:
                gate = np.random.choice(8)
            elif self.situation == 1:
                gate = (int(self.sim_t/60) )% 8
            elif self.situation == 2:
                gate = 4
            else:
                gate = len(self.planes) % 4
            p = self._spawn_plane(gate=gate, noise=noise)
            # 分配唯一id
            p.id = self.unique_ids[self.id_counter]
            self.planes.append(p)
            self.id_counter += 1

        # 与高级模块的接口，包括目标决策
        self._high_level_control()

        # reset dest
        self.dest.reset()

        # init state
        self._set_state()
        self.state_init = self.state

        # logging
        P_info = {}
        for id in self.unique_ids:
            try:
                i = np.nonzero([p.id == id for p in self.planes])[0][0]
                p = self.planes[i]
                n = p.n
                e = p.e 
                hdg = p.hdg
                tas = p.tas
                goal = int(p.fly_to_goal)
            except:
                n, e, hdg, tas, goal = None, None, None, None, None

            P_info[f"P{id}_n"] = n
            P_info[f"P{id}_e"] = e
            P_info[f"P{id}_hdg"] = hdg
            P_info[f"P{id}_tas"] = tas
            P_info[f"P{id}_goal"] = goal
        self.logger = UAMLogger(sim_t=self.sim_t, **P_info)
        return self.state

    def _high_level_control(self):
        """决定哪个飞行器应该飞向目标."""
        if len(self.planes) > 0:

            # 检查飞行器是否有进入信号
            if all([p.fly_to_goal == -1.0 for p in self.planes]):

                idx = np.argmin([p.D_dest for p in self.planes])
                for i, _ in enumerate(self.planes):
                    if i == idx:
                        self.planes[i].fly_to_goal = 1.0
                    else:
                        self.planes[i].fly_to_goal = -1.0

    def _spawn_plane(self, gate:int=None, noise:bool=True, role:str= "RL" ):
        # 基础设置
        if self.situation == 3:
            qdr = [0.0,90.0,180.0,270.0][gate]
        else :
            qdr = [0.0,45.0,90.0,135.0,180.0,225.0,270.0,315.0][gate]
        hdg = (qdr + 180) % 360
        tas = self.actas
        dist = self.dest.spawn_radius

        if noise:
            hdg = (hdg + np.random.uniform(low=-20.0, high=20.0)) % 360
            tas += np.random.uniform(low=-self.delta_tas, high=self.delta_tas)

        # 确定初始位置
        E_add, N_add = xy_from_polar(r=dist, angle=dtr(qdr))
        lat, lon = to_latlon(north=self.dest.N+N_add, east=self.dest.E+E_add, number=32)

        # consider behavior type
        p = Plane(role=role, dt=self.dt, actype=self.actype, lat=lat, lon=lon, alt=self.acalt, hdg=hdg, tas=tas,plane_id=self.unique_id_counter)

        # 设置utm的坐标
        p.n, p.e, _ = to_utm(lat=lat, lon=lon)

        # 计算到目的地的初始距离
        p.D_dest     = ED(N0=self.dest.N, E0=self.dest.E, N1=p.n, E1=p.e)
        p.D_dest_old = copy(p.D_dest)

        # 在默认情况下不要飞向目标
        p.fly_to_goal = -1.0
        return p

    def _handle_respawn(self, entered_open:np.ndarray):
        """当飞机正确进入开放的目的地区域或离开整个地图时，重置它们."""
        for i, p in enumerate(self.planes):
            if (entered_open[i] and p.fly_to_goal == 1.0) or p.D_dest >= self.dest.respawn_radius:
                self.planes.pop(i)
                self.N_planes = len(self.planes)

    def _done(self):
        d = False
        # 1.人工终止信号
        if self.step_cnt >= self._max_episode_steps:
            d = True
        # 2.所有飞机都消失了
        elif len(self.planes) == 0:
            d = True
        # 3.只剩下非RL控制的飞机
        elif all([p.role != "RL" for p in self.planes]):
            d = True

        if d:
            for p in self.planes:
                if p.land_time is None and p.spawn_time is not None:
                    # 如果飞机未降落但环境已终止，记录当前时间作为降落时间
                    p.land_time = self.sim_t
                    self.flight_durations.append(p.land_time - p.spawn_time)
            avg_time = np.mean(self.flight_durations) if self.flight_durations else 0
            print(avg_time)
            # Release the video writer
            if hasattr(self, "video_writer"):
                self.video_writer.release()

            # Dump episode details
            if self.sim_study:
                self.logger.dump(name="UAM_SimStudy_" + str(self.N_agents_max) + "_" + str(self.safe_number))
            else:
                self.logger.dump(name="UAM_ValScene_" + str(self.situation) + "_" + str(self.N_agents_max))
        return d

    def render(self, mode=None):
        super().render(mode=mode)
