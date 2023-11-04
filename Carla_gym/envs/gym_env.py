import math
import matplotlib.pyplot as plt
import gym
import numpy as np
import pandas as pd
from gym import spaces
from tools.modules import *
from agents.local_planner.frenet_optimal_trajectory_lon import FrenetPlanner as MotionPlanner
from agents.low_level_controller.controller import VehiclePIDController
from agents.low_level_controller.controller import PIDLongitudinalController
from agents.low_level_controller.controller import PIDLateralController
from agents.tools.misc import get_speed
from agents.low_level_controller.controller import IntelligentDriverModel
# from MPC.MPC_controller_yundongxue import MPC_controller_yundongxue
# from MPC.parameter_config import MPC_lon_Config
from MPC.parameter_config_0 import MPC_Config_0
# from MPC.MPC_controller_lon import MPC_controller_lon
# from MPC.MPC_controller_lat import MPC_controller_lat
# from MPC.MPC_controller_lon_lat import MPC_controller_lon_lat
from MPC.MPC_controller_lon_lat_ipopt_nonlinear_terminal import MPC_controller_lon_lat_ipopt_nonlinear_terminal
from MPC.MPC_controller_lon_lat_acados_nonlinear_terminal_without_constrain import MPC_controller_lon_lat_acados_nonlinear_terminal
from MPC.MPC_controller_lon_lat_ipopt_nonlinear_sequence import MPC_controller_lon_lat_ipopt_nonlinear_sequence
from MPC.MPC_controller_lon_lat_ipopt_nonlinear_opt import MPC_controller_lon_lat_ipopt_nonlinear_opt
from MPC.parameter_config import MPC_lon_lat_Config
from MPC.parameter_config import MPC_lon_Config
from MPC.parameter_config import MPC_lat_Config
from datas.data_log import data_collection

MPC_Config = MPC_Config_0()
MPC_lon_Config = MPC_lon_Config()
MPC_lon_lat_Config = MPC_lon_lat_Config()
MPC_lat_Config = MPC_lat_Config()
from agents.local_planner.frenet_optimal_trajectory_lon import velocity_inertial_to_frenet, \
    get_obj_S_yaw

MODULE_WORLD = 'WORLD'
MODULE_HUD = 'HUD'
MODULE_INPUT = 'INPUT'
MODULE_TRAFFIC = 'TRAFFIC'
TENSOR_ROW_NAMES = ['EGO', 'LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN',
                    'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN']


def closest(lst, K):
    """
    Find closes value in a list
    """
    return lst[min(range(len(lst)), key=lambda i: abs(lst[i] - K))]


def frenet_to_inertial(s, d, csp):
    """
    transform a point from frenet frame to inertial frame
    input: frenet s and d variable and the instance of global cubic spline class
    output: x and y in global frame
    """
    ix, iy, iz = csp.calc_position(s)

    iyaw = csp.calc_yaw(s)
    x = ix + d * math.cos(iyaw + math.pi / 2.0)
    y = iy + d * math.sin(iyaw + math.pi / 2.0)

    return x, y, iz, iyaw


def inertial_to_body_frame(ego_location, xi, yi, psi):
    Xi = np.array([xi, yi])  # inertial frame
    R_psi_T = np.array([[np.cos(psi), np.sin(psi)],  # Rotation matrix transpose
                        [-np.sin(psi), np.cos(psi)]])
    Xt = np.array([ego_location[0],  # Translation from inertial to body frame
                   ego_location[1]])
    Xb = np.matmul(R_psi_T, Xi - Xt)
    return Xb


def closest_wp_idx(ego_state, fpath, f_idx, w_size=10):
    """
    given the ego_state and frenet_path this function returns the closest WP in front of the vehicle that is within the w_size
    """

    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[2])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    return f_idx + closest_wp_index


def closest_wp_idx_ref(ego_state, fpath, f_idx, w_size=10):
    """
    given the ego_state and frenet_path this function returns the closest WP in front of the vehicle that is within the w_size
    """

    min_dist = 300  # in meters (Max 100km/h /3.6) * 2 sn
    ego_location = [ego_state[0], ego_state[1]]
    closest_wp_index = 0  # default WP
    w_size = w_size if w_size <= len(fpath.t) - 2 - f_idx else len(fpath.t) - 2 - f_idx
    for i in range(w_size):
        temp_wp = [fpath.x[f_idx + i], fpath.y[f_idx + i]]
        temp_dist = euclidean_distance(ego_location, temp_wp)
        if temp_dist <= min_dist \
                and inertial_to_body_frame(ego_location, temp_wp[0], temp_wp[1], ego_state[4])[0] > 0.0:
            closest_wp_index = i
            min_dist = temp_dist

    return f_idx + closest_wp_index


class CarlagymEnv(gym.Env):

    # metadata = {'render.modes': ['human']}
    def __init__(self):

        self.__version__ = "9.9.2"
        self.lon_lat_param = MPC_lon_lat_Config
        self.lon_lat_controller_ipopt = MPC_controller_lon_lat_ipopt_nonlinear_terminal(self.lon_lat_param)
        # self.lon_lat_controller_acados = MPC_controller_lon_lat_acados_nonlinear_terminal(self.lon_lat_param)
        # self.lon_lat_controller_ipopt = MPC_controller_lon_lat_ipopt_nonlinear_sequence(self.lon_lat_param)
        # self.lon_lat_controller_ipopt = MPC_controller_lon_lat_ipopt_nonlinear_opt(self.lon_lat_param)

        # simulation
        self.verbosity = 0
        self.auto_render = False  # automatically render the environment
        self.n_step = 0
        try:
            self.global_route = np.load(
                'road_maps/global_route_town04.npy')  # track waypoints (center lane of the second lane from left)
            # 1520 *  3

        except IOError:
            self.global_route = None

        self.ref_path = xlrd.open_workbook(os.path.abspath('.') + '/tools/second_global_path.xlsx')
        self.ref_path = self.ref_path.sheets()[0]
        self.ref_path_x = self.ref_path.col_values(0)
        self.ref_path_y = self.ref_path.col_values(1)
        self.ref_path_phi = self.ref_path.col_values(2)
        self.ref_path_x = self.ref_path_x[0:1520]
        self.ref_path_y = self.ref_path_y[0:1520]
        self.ref_path_phi = self.ref_path_phi[0:1520]

        self.ref_path_left = xlrd.open_workbook(os.path.abspath('.') + '/tools/left_global_path.xlsx')
        self.ref_path_left = self.ref_path_left.sheets()[0]
        self.ref_path_left_x = self.ref_path_left.col_values(0)
        self.ref_path_left_y = self.ref_path_left.col_values(1)
        self.ref_path_left_phi = self.ref_path_left.col_values(2)
        self.ref_path_left_x = self.ref_path_left_x[0:1520]  # 700-451    1800-1200   2135-1434
        self.ref_path_left_y = self.ref_path_left_y[0:1520]
        self.ref_path_left_phi = self.ref_path_left_phi[0:1520]

        # self.ref_path = xlrd.open_workbook(os.path.abspath('.') + '/tools/ref_global_path.xlsx')
        # self.ref_path = self.ref_path.sheets()[0]
        # self.ref_path_x = self.ref_path.col_values(0)
        # self.ref_path_y = self.ref_path.col_values(1)
        # self.ref_path_phi = self.ref_path.col_values(2)

        # constraints
        self.targetSpeed = float(cfg.GYM_ENV.TARGET_SPEED)
        self.maxSpeed = float(cfg.GYM_ENV.MAX_SPEED)
        self.minSpeed = float(cfg.GYM_ENV.MIN_SPEED)
        self.LANE_WIDTH = float(cfg.CARLA.LANE_WIDTH)
        self.N_SPAWN_CARS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_CARS)
        self.N_SPAWN_PEDESTRAINS = int(cfg.TRAFFIC_MANAGER.N_SPAWN_PEDESTRAINS)
        self.obj_max_vs = int(cfg.TRAFFIC_MANAGER.MAX_SPEED)
        self.walker_max_vd = int(cfg.TRAFFIC_MANAGER.Walker_MAX_SPEED)
        self.d_max_s = int(cfg.CARLA.D_MAX_S)

        # frenet
        self.f_idx = 0
        self.init_s = None  # initial frenet s value - will be updated in reset function
        self.max_s = int(cfg.CARLA.MAX_S)
        self.effective_distance_from_vehicle_ahead = int(cfg.GYM_ENV.DISTN_FRM_VHCL_AHD)
        self.lanechange = False
        self.is_first_path = True

        # RL
        self.collision_penalty = int(cfg.RL.COLLISION)
        self.low_speed_reward = float(cfg.RL.Low_SPEED_REWARD)
        self.middle_speed_reward = float(cfg.RL.Middle_SPEED_REWARD)
        self.high_speed_reward = float(cfg.RL.High_SPEED_REWARD)
        # self.lane_change_reward = float(cfg.RL.LANE_CHANGE_REWARD)
        # self.lane_change_penalty = float(cfg.RL.LANE_CHANGE_PENALTY)
        # self.off_the_road_penalty = int(cfg.RL.OFF_THE_ROAD)

        # instances
        self.ego = None
        self.ego_los_sensor = None
        self.module_manager = None
        self.world_module = None
        self.traffic_module = None
        self.hud_module = None
        self.input_module = None
        self.control_module = None
        self.init_transform = None  # ego initial transform to recover at each episode
        self.Input = None
        self.fpath = None
        self.eps_rew = 0
        self.u_last = [0.0,0.0]
        self.lane_last = 0
        self.ref_left_idx = 0
        self.fig, self.ax = plt.subplots()
        self.x = []
        self.y = []
        self.actor_enumerated_dict = {}
        self.actor_enumeration = []
        self.side_window = 5
        self.look_back = int(cfg.GYM_ENV.LOOK_BACK)
        self.x_m = None

        self.motionPlanner = None
        self.vehicleController = None
        self.PIDLongitudinalController = None
        self.PIDLateralController = None

        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

        action_low = -1
        action_high = 1
        self.acton_dim = (1, 1)
        self.action_space = spaces.Box(action_low, action_high, shape=self.acton_dim, dtype='float32')
        self.obs_dim = (1, (self.N_SPAWN_CARS + self.N_SPAWN_PEDESTRAINS + 1) * 4)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.obs_dim, dtype='float32')
        self.state = np.zeros_like(self.observation_space.sample())
        # data_record

        self.log_flag = cfg.GYM_ENV.LOG_FLAG
        if self.log_flag:
            self.log = data_collection()
        else:
            pass

    def get_vehicle_ahead(self, ego_s, ego_d, ego_init_d, ego_target_d):
        """
        This function returns the values for the leading actor in front of the ego vehicle. When there is lane-change
        it is important to consider actor in the current lane and target lane. If leading actor in the current lane is
        too close than it is considered to be vehicle_ahead other wise target lane is prioritized.
        """
        distance = self.effective_distance_from_vehicle_ahead
        others_s = [0 for _ in range(self.N_SPAWN_CARS)]
        others_d = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d = actor['Frenet State'][0][-1], actor['Frenet State'][1]
            others_s[i] = act_s
            others_d[i] = act_d

        init_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 1.75) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        init_lane_strict_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 0.4) * (abs(np.array(others_d) - ego_init_d) < 1))[0]

        target_lane_d_idx = \
            np.where((abs(np.array(others_d) - ego_d) < 3.3) * (abs(np.array(others_d) - ego_target_d) < 1))[0]

        if len(init_lane_d_idx) and len(target_lane_d_idx) == 0:
            return None  # no vehicle ahead
        else:
            init_lane_s = np.array(others_s)[init_lane_d_idx]
            init_s_idx = np.concatenate(
                (np.array(init_lane_d_idx).reshape(-1, 1), (init_lane_s - ego_s).reshape(-1, 1),)
                , axis=1)
            sorted_init_s_idx = init_s_idx[init_s_idx[:, 1].argsort()]

            init_lane_strict_s = np.array(others_s)[init_lane_strict_d_idx]
            init_strict_s_idx = np.concatenate(
                (np.array(init_lane_strict_d_idx).reshape(-1, 1), (init_lane_strict_s - ego_s).reshape(-1, 1),)
                , axis=1)
            sorted_init_strict_s_idx = init_strict_s_idx[init_strict_s_idx[:, 1].argsort()]

            target_lane_s = np.array(others_s)[target_lane_d_idx]
            target_s_idx = np.concatenate((np.array(target_lane_d_idx).reshape(-1, 1),
                                           (target_lane_s - ego_s).reshape(-1, 1),), axis=1)
            sorted_target_s_idx = target_s_idx[target_s_idx[:, 1].argsort()]

            if any(sorted_init_s_idx[:, 1][sorted_init_s_idx[:, 1] <= 10] > 0):
                vehicle_ahead_idx = int(sorted_init_s_idx[:, 0][sorted_init_s_idx[:, 1] > 0][0])
            elif any(sorted_init_strict_s_idx[:, 1][sorted_init_strict_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_init_strict_s_idx[:, 0][sorted_init_strict_s_idx[:, 1] > 0][0])
            elif any(sorted_target_s_idx[:, 1][sorted_target_s_idx[:, 1] <= distance] > 0):
                vehicle_ahead_idx = int(sorted_target_s_idx[:, 0][sorted_target_s_idx[:, 1] > 0][0])
            else:
                return None

            return self.traffic_module.actors_batch[vehicle_ahead_idx]

    def obj_info_simple(self):
        """
        Actor:  [actor_id]
        Frenet:  [s,d,v_s, v_d, phi_Frenet, K_Frenet]
        Cartesian:  [x, y, v_x, v_y, phi, speed, delta_f]
        """
        obj_info = {}
        # obj_actor = [0 for _ in range(self.N_SPAWN_CARS)]
        # obj_frenet = [0 for _ in range(self.N_SPAWN_CARS)]
        # obj_cartesian = [0 for _ in range(self.N_SPAWN_CARS)]
        obj_actor = [0 for _ in range(self.N_SPAWN_CARS)]
        obj_frenet = [0 for _ in range(self.N_SPAWN_CARS)]
        obj_cartesian = [0 for _ in range(self.N_SPAWN_CARS)]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            obj_idx = i
            obj_actor[obj_idx] = actor['Actor']
            obj_frenet[obj_idx] = actor['Obj_Frenet_state']
            obj_cartesian[obj_idx] = actor['Obj_Cartesian_state']
        obj_dict = ({'Obj_actor': obj_actor, 'Obj_frenet': obj_frenet, 'Obj_cartesian': obj_cartesian})

        others_s = np.zeros(self.N_SPAWN_CARS)
        others_d = np.zeros(self.N_SPAWN_CARS)
        others_v_S = np.zeros(self.N_SPAWN_CARS)
        others_v_D = np.zeros(self.N_SPAWN_CARS)
        others_phi_Frenet = np.zeros(self.N_SPAWN_CARS)

        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s, act_d, act_v_S, act_v_D, act_psi_Frenet, act_K_Frenet = actor['Obj_Frenet_state']
            others_s[i] = act_s
            others_d[i] = act_d
            others_v_S[i] = act_v_S
            others_v_D[i] = act_v_D
            others_phi_Frenet[i] = act_psi_Frenet
        obj_info_Mux = np.vstack((others_s, others_d, others_v_S, others_v_D, others_phi_Frenet))

        return obj_dict, obj_info_Mux

    def obj_info(self):
        """
        Actor:  [actor_id]
        Frenet:  [s,d,v_s, v_d, phi_Frenet, K_Frenet]
        Cartesian:  [x, y, v_x, v_y, phi, speed, delta_f]
        """
        obj_info = {}
        obj_actor = [0 for _ in range(self.N_SPAWN_CARS)]
        obj_frenet = [0 for _ in range(self.N_SPAWN_CARS)]
        obj_cartesian = [0 for _ in range(self.N_SPAWN_CARS)]
        obj_targetSpeed = [0 for _ in range(self.N_SPAWN_CARS)]

        for i, actor in enumerate(self.traffic_module.actors_batch):
            obj_idx = i
            obj_actor[obj_idx] = actor['Actor']
            obj_frenet[obj_idx] = actor['Obj_Frenet_state']
            obj_cartesian[obj_idx] = actor['Obj_Cartesian_state']
            obj_targetSpeed[obj_idx] = actor['Obj_Cartesian_state'][7]

        self.enumerate_actors()

        self.ego_lane_preceding_idx = 0
        self.ego_lane_following_idx = 0
        self.left_lane_idx = 0
        self.left_lane_preceding_idx = 0
        self.left_lane_following_idx = 0
        self.left_left_lane_idx = 0
        self.left_left_lane_preceding_idx = 0
        self.left_left_lane_following_idx = 0
        self.right_lane_idx = 0
        self.right_lane_preceding_idx = 0
        self.right_lane_following_idx = 0
        self.right__rightlane_idx = 0
        self.right_right_lane_preceding_idx = 0
        self.right_right_lane_following_idx = 0

        others_id = [0 for _ in range(self.N_SPAWN_CARS)]
        for i in range(self.N_SPAWN_CARS):
            others_id[i] = obj_actor[i].id
            if others_id[i] == self.actor_enumeration[0]:
                self.ego_lane_preceding_idx = i + 1
            elif others_id[i] == self.actor_enumeration[1]:
                self.ego_lane_following_idx = i + 1
            elif others_id[i] == self.actor_enumeration[2]:
                self.left_lane_idx = i + 1
            elif others_id[i] == self.actor_enumeration[3]:
                self.left_lane_preceding_idx = i + 1
            elif others_id[i] == self.actor_enumeration[4]:
                self.left_lane_following_idx = i + 1
            elif others_id[i] == self.actor_enumeration[5]:
                self.left_left_lane_idx = i + 1
            elif others_id[i] == self.actor_enumeration[6]:
                self.left_left_lane_preceding_idx = i + 1
            elif others_id[i] == self.actor_enumeration[7]:
                self.left_left_lane_following_idx = i + 1
            elif others_id[i] == self.actor_enumeration[8]:
                self.right_lane_idx = i + 1
            elif others_id[i] == self.actor_enumeration[9]:
                self.right_lane_preceding_idx = i + 1
            elif others_id[i] == self.actor_enumeration[10]:
                self.right_lane_following_idx = i + 1
            elif others_id[i] == self.actor_enumeration[11]:
                self.right_right_lane_idx = i + 1
            elif others_id[i] == self.actor_enumeration[12]:
                self.right_right_lane_preceding_idx = i + 1
            elif others_id[i] == self.actor_enumeration[13]:
                self.right_right_lane_following_idx = i + 1
            else:
                pass

        if self.ego_lane_preceding_idx != 0:
            ego_preceding_actor = obj_actor[self.ego_lane_preceding_idx - 1]
            ego_preceding_actor_frenet = obj_frenet[self.ego_lane_preceding_idx - 1]
            ego_preceding_actor_cartesian = obj_cartesian[self.ego_lane_preceding_idx - 1]
        else:
            ego_preceding_actor = None
            ego_preceding_actor_frenet = None
            ego_preceding_actor_cartesian = None

        if self.ego_lane_following_idx != 0:
            ego_following_actor = obj_actor[self.ego_lane_following_idx - 1]
            ego_following_actor_frenet = obj_frenet[self.ego_lane_following_idx - 1]
            ego_following_actor_cartesian = obj_cartesian[self.ego_lane_following_idx - 1]

        else:
            ego_following_actor = None
            ego_following_actor_frenet = None
            ego_following_actor_cartesian = None

        if self.left_lane_preceding_idx != 0:
            left_preceding_actor = obj_actor[self.left_lane_preceding_idx - 1]
            left_preceding_actor_frenet = obj_frenet[self.left_lane_preceding_idx - 1]
            left_preceding_actor_cartesian = obj_cartesian[self.left_lane_preceding_idx - 1]
        else:
            left_preceding_actor = None
            left_preceding_actor_frenet = None
            left_preceding_actor_cartesian = None

        if self.left_left_lane_preceding_idx != 0:
            left_left_preceding_actor = obj_actor[self.left_left_lane_preceding_idx - 1]
            left_left_preceding_actor_frenet = obj_frenet[self.left_left_lane_preceding_idx - 1]
            left_left_preceding_actor_cartesian = obj_cartesian[self.left_left_lane_preceding_idx - 1]
        else:
            left_left_preceding_actor = None
            left_left_preceding_actor_frenet = None
            left_left_preceding_actor_cartesian = None

        if self.left_lane_following_idx != 0:
            left_following_actor = obj_actor[self.left_lane_following_idx - 1]
            left_following_actor_frenet = obj_frenet[self.left_lane_following_idx - 1]
            left_following_actor_cartesian = obj_cartesian[self.left_lane_following_idx - 1]

        else:
            left_following_actor = None
            left_following_actor_frenet = None
            left_following_actor_cartesian = None

        if self.right_lane_preceding_idx != 0:
            right_preceding_actor = obj_actor[self.right_lane_preceding_idx - 1]
            right_preceding_actor_frenet = obj_frenet[self.right_lane_preceding_idx - 1]
            right_preceding_actor_cartesian = obj_cartesian[self.right_lane_preceding_idx - 1]
        else:
            right_preceding_actor = None
            right_preceding_actor_frenet = None
            right_preceding_actor_cartesian = None

        if self.right_right_lane_preceding_idx != 0:
            right_right_preceding_actor = obj_actor[self.right_right_lane_preceding_idx - 1]
            right_right_preceding_actor_frenet = obj_frenet[self.right_right_lane_preceding_idx - 1]
            right_right_preceding_actor_cartesian = obj_cartesian[self.right_right_lane_preceding_idx - 1]
        else:
            right_right_preceding_actor = None
            right_right_preceding_actor_frenet = None
            right_right_preceding_actor_cartesian = None

        if self.right_lane_following_idx != 0:
            right_following_actor = obj_actor[self.right_lane_following_idx - 1]
            right_following_actor_frenet = obj_frenet[self.right_lane_following_idx - 1]
            right_following_actor_cartesian = obj_cartesian[self.right_lane_following_idx - 1]
        else:
            right_following_actor = None
            right_following_actor_frenet = None
            right_following_actor_cartesian = None

        if self.left_lane_idx != 0:
            left_actor = obj_actor[self.left_lane_idx - 1]
            left_actor_frenet = obj_frenet[self.left_lane_idx - 1]
            left_actor_cartesian = obj_cartesian[self.left_lane_idx - 1]
        else:
            left_actor = None
            left_actor_frenet = None
            left_actor_cartesian = None

        if self.right_lane_idx != 0:
            right_actor = obj_actor[self.right_lane_idx - 1]
            right_actor_frenet = obj_frenet[self.right_lane_idx - 1]
            right_actor_cartesian = obj_cartesian[self.right_lane_idx - 1]
        else:
            right_actor = None
            right_actor_frenet = None
            right_actor_cartesian = None

        obj_info = ({'Obj_actor': obj_actor, 'Obj_frenet': obj_frenet, 'Obj_cartesian': obj_cartesian,
                     'Ego_preceding': [ego_preceding_actor, ego_preceding_actor_frenet,
                                       ego_preceding_actor_cartesian],
                     'Ego_following': [ego_following_actor, ego_following_actor_frenet,
                                       ego_following_actor_cartesian],
                     'Left_preceding': [left_preceding_actor, left_preceding_actor_frenet,
                                        left_preceding_actor_cartesian],
                     'Left_Left_preceding': [left_left_preceding_actor, left_left_preceding_actor_frenet,
                                             left_left_preceding_actor_cartesian],
                     'Left_following': [left_following_actor, left_following_actor_frenet,
                                        left_following_actor_cartesian],
                     'Right_preceding': [right_preceding_actor, right_preceding_actor_frenet,
                                         right_preceding_actor_cartesian],
                     'Right_Right_preceding': [right_right_preceding_actor, right_right_preceding_actor_frenet,
                                               right_right_preceding_actor_cartesian],
                     'Right_following': [right_following_actor, right_following_actor_frenet,
                                         right_following_actor_cartesian],
                     'Left': [left_actor, left_actor_frenet, left_actor_cartesian],
                     'Right': [right_actor, right_actor_frenet, right_actor_cartesian]
                     })

        return obj_info

    def enumerate_actors(self):
        """
        Given the traffic actors and ego_state this fucntion enumerate actors, calculates their relative positions with
        to ego and assign them to actor_enumerated_dict.
        Keys to be updated: ['LEADING', 'FOLLOWING', 'LEFT', 'LEFT_UP', 'LEFT_DOWN', 'LLEFT', 'LLEFT_UP',
        'LLEFT_DOWN', 'RIGHT', 'RIGHT_UP', 'RIGHT_DOWN', 'RRIGHT', 'RRIGHT_UP', 'RRIGHT_DOWN']
        """

        self.actor_enumeration = []
        ego_s = self.actor_enumerated_dict['EGO']['S'][-1]
        ego_d = self.actor_enumerated_dict['EGO']['D'][-1]

        others_s = [0 for _ in range(len(self.traffic_module.actors_batch))]
        others_d = [0 for _ in range(len(self.traffic_module.actors_batch))]
        others_id = [0 for _ in range(len(self.traffic_module.actors_batch))]
        for i, actor in enumerate(self.traffic_module.actors_batch):
            act_s = actor['Obj_Frenet_state'][0]
            act_d = actor['Obj_Frenet_state'][1]
            others_s[i] = act_s
            others_d[i] = act_d
            others_id[i] = actor['Actor'].id

        def append_actor(x_lane_d_idx, actor_names=None):
            # actor names example: ['left', 'leftUp', 'leftDown']
            x_lane_s = np.array(others_s)[x_lane_d_idx]
            x_lane_id = np.array(others_id)[x_lane_d_idx]
            s_idx = np.concatenate(
                (np.array(x_lane_d_idx).reshape(-1, 1), (x_lane_s - ego_s).reshape(-1, 1), x_lane_id.reshape(-1, 1)),
                axis=1)
            sorted_s_idx = s_idx[s_idx[:, 1].argsort()]

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][abs(sorted_s_idx[:, 1]) < self.side_window][0])] if (
                    any(abs(
                        sorted_s_idx[:, 1][abs(sorted_s_idx[:, 1]) <= self.side_window]) >= -self.side_window)) else -1)

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] > self.side_window][0])] if (
                    any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] > 0] > self.side_window)) else -1)

            self.actor_enumeration.append(
                others_id[int(sorted_s_idx[:, 0][sorted_s_idx[:, 1] < -self.side_window][-1])] if (
                    any(sorted_s_idx[:, 1][sorted_s_idx[:, 1] < 0] < -self.side_window)) else -1)

        # --------------------------------------------- ego lane -------------------------------------------------
        same_lane_d_idx = np.where(abs(np.array(others_d) - ego_d) <= 1.75)[0]
        if len(same_lane_d_idx) == 0:
            self.actor_enumeration.append(-2)
            self.actor_enumeration.append(-2)

        else:
            same_lane_s = np.array(others_s)[same_lane_d_idx]
            same_lane_id = np.array(others_id)[same_lane_d_idx]
            same_s_idx = np.concatenate((np.array(same_lane_d_idx).reshape(-1, 1), (same_lane_s - ego_s).reshape(-1, 1),
                                         same_lane_id.reshape(-1, 1)), axis=1)
            sorted_same_s_idx = same_s_idx[same_s_idx[:, 1].argsort()]
            self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] > 0][0])]
                                          if (any(sorted_same_s_idx[:, 1] > 0)) else -1)
            self.actor_enumeration.append(others_id[int(sorted_same_s_idx[:, 0][sorted_same_s_idx[:, 1] < 0][-1])]
                                          if (any(sorted_same_s_idx[:, 1] < 0)) else -1)

        # --------------------------------------------- left lane -------------------------------------------------
        left_lane_d_idx = np.where(((np.array(others_d) - ego_d) < -1.75) * ((np.array(others_d) - ego_d) > -5.25))[0]
        if ego_d < -1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(left_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(left_lane_d_idx)

        # ------------------------------------------- two left lane -----------------------------------------------
        lleft_lane_d_idx = np.where(((np.array(others_d) - ego_d) > -8.75) * ((np.array(others_d) - ego_d) < -5.25))[0]

        if ego_d < 1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(lleft_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(lleft_lane_d_idx)

            # ---------------------------------------------- rigth lane --------------------------------------------------
        right_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 1.75) * ((np.array(others_d) - ego_d) < 5.25))[0]
        if ego_d > 5.25:
            self.actor_enumeration += [-2, -2, -2]

        elif len(right_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(right_lane_d_idx)

        # ------------------------------------------- two rigth lane --------------------------------------------------
        rright_lane_d_idx = np.where(((np.array(others_d) - ego_d) > 5.25) * ((np.array(others_d) - ego_d) < 8.75))[0]
        if ego_d > 1.75:
            self.actor_enumeration += [-2, -2, -2]

        elif len(rright_lane_d_idx) == 0:
            self.actor_enumeration += [-1, -1, -1]

        else:
            append_actor(rright_lane_d_idx)

        # Fill enumerated actor values

        actor_id_s_d = {}  # all actor_sd
        norm_s = []
        # norm_d = []
        for actor in self.traffic_module.actors_batch:
            actor_id_s_d[actor['Actor'].id] = actor['Frenet State']

        for i, actor_id in enumerate(self.actor_enumeration):
            if actor_id >= 0:
                actor_norm_s = []
                act_s_hist = actor_id_s_d[actor_id][0]
                act_d = actor_id_s_d[actor_id][1]  # act_s_hist:list act_d:float
                for act_s, ego_s in zip(list(act_s_hist)[-self.look_back:],
                                        self.actor_enumerated_dict['EGO']['S'][-self.look_back:]):
                    actor_norm_s.append((act_s - ego_s) / self.max_s)
                norm_s.append(actor_norm_s)
            #    norm_d[i] = (act_d - ego_d) / (3 * self.LANE_WIDTH)
            # -1:empty lane, -2:no lane
            else:
                norm_s.append(actor_id)

        # How to fill actor_s when there is no lane or lane is empty. relative_norm_s to ego vehicle
        emp_ln_max = 0.03
        emp_ln_min = -0.03
        no_ln_down = -0.03
        no_ln_up = 0.004
        no_ln = 0.001

        if norm_s[0] not in (-1, -2):
            self.actor_enumerated_dict['LEADING'] = {'S': norm_s[0]}
        else:
            self.actor_enumerated_dict['LEADING'] = {'S': [emp_ln_max]}

        if norm_s[1] not in (-1, -2):
            self.actor_enumerated_dict['FOLLOWING'] = {'S': norm_s[1]}
        else:
            self.actor_enumerated_dict['FOLLOWING'] = {'S': [emp_ln_min]}

        if norm_s[2] not in (-1, -2):
            self.actor_enumerated_dict['LEFT'] = {'S': norm_s[2]}
        else:
            self.actor_enumerated_dict['LEFT'] = {'S': [emp_ln_min] if norm_s[2] == -1 else [no_ln]}

        if norm_s[3] not in (-1, -2):
            self.actor_enumerated_dict['LEFT_UP'] = {'S': norm_s[3]}
        else:
            self.actor_enumerated_dict['LEFT_UP'] = {'S': [emp_ln_max] if norm_s[3] == -1 else [no_ln_up]}

        if norm_s[4] not in (-1, -2):
            self.actor_enumerated_dict['LEFT_DOWN'] = {'S': norm_s[4]}
        else:
            self.actor_enumerated_dict['LEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[4] == -1 else [no_ln_down]}

        if norm_s[5] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT'] = {'S': norm_s[5]}
        else:
            self.actor_enumerated_dict['LLEFT'] = {'S': [emp_ln_min] if norm_s[5] == -1 else [no_ln]}

        if norm_s[6] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT_UP'] = {'S': norm_s[6]}
        else:
            self.actor_enumerated_dict['LLEFT_UP'] = {'S': [emp_ln_max] if norm_s[6] == -1 else [no_ln_up]}

        if norm_s[7] not in (-1, -2):
            self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': norm_s[7]}
        else:
            self.actor_enumerated_dict['LLEFT_DOWN'] = {'S': [emp_ln_min] if norm_s[7] == -1 else [no_ln_down]}

        if norm_s[8] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT'] = {'S': norm_s[8]}
        else:
            self.actor_enumerated_dict['RIGHT'] = {'S': [emp_ln_min] if norm_s[8] == -1 else [no_ln]}

        if norm_s[9] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT_UP'] = {'S': norm_s[9]}
        else:
            self.actor_enumerated_dict['RIGHT_UP'] = {'S': [emp_ln_max] if norm_s[9] == -1 else [no_ln_up]}

        if norm_s[10] not in (-1, -2):
            self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': norm_s[10]}
        else:
            self.actor_enumerated_dict['RIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[10] == -1 else [no_ln_down]}

        if norm_s[11] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT'] = {'S': norm_s[11]}
        else:
            self.actor_enumerated_dict['RRIGHT'] = {'S': [emp_ln_min] if norm_s[11] == -1 else [no_ln]}

        if norm_s[12] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT_UP'] = {'S': norm_s[12]}
        else:
            self.actor_enumerated_dict['RRIGHT_UP'] = {'S': [emp_ln_max] if norm_s[12] == -1 else [no_ln_up]}

        if norm_s[13] not in (-1, -2):
            self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': norm_s[13]}
        else:
            self.actor_enumerated_dict['RRIGHT_DOWN'] = {'S': [emp_ln_min] if norm_s[13] == -1 else [no_ln_down]}

    def walker_info_simple(self):
        """
        Actor:  [actor_id]
        Frenet:  [s,d,v_s, v_d, phi_Frenet]
        Cartesian:  [x, y, v_x, v_y, phi, speed, delta_f]
        """
        walker_info = {}
        walker_actor = [0 for _ in range(self.N_SPAWN_PEDESTRAINS)]
        walker_frenet = [0 for _ in range(self.N_SPAWN_PEDESTRAINS)]
        walker_cartesian = [0 for _ in range(self.N_SPAWN_PEDESTRAINS)]
        for i, walker in enumerate(self.traffic_module.walkers_batch):
            walker_idx = i
            walker_actor[walker_idx] = walker['Actor']
            walker_frenet[walker_idx] = walker['Walker_Frenet_state']
            walker_cartesian[walker_idx] = walker['Walker_Cartesian_state']
        walker_dict = ({'Walker_actor': walker_actor, 'Walker_frenet': walker_frenet, 'Walker_cartesian': walker_cartesian})

        others_s = np.zeros(self.N_SPAWN_PEDESTRAINS)
        others_d = np.zeros(self.N_SPAWN_PEDESTRAINS)
        others_v_S = np.zeros(self.N_SPAWN_PEDESTRAINS)
        others_v_D = np.zeros(self.N_SPAWN_PEDESTRAINS)
        others_phi_Frenet = np.zeros(self.N_SPAWN_PEDESTRAINS)

        for i, walker in enumerate(self.traffic_module.walkers_batch):
            act_s, act_d, act_v_S, act_v_D, act_psi_Frenet = walker['Walker_Frenet_state']
            others_s[i] = act_s
            others_d[i] = act_d
            others_v_S[i] = act_v_S
            others_v_D[i] = act_v_D
            others_phi_Frenet[i] = act_psi_Frenet
        walker_info_Mux = np.vstack((others_s, others_d, others_v_S, others_v_D, others_phi_Frenet))

        return walker_dict, walker_info_Mux

    def not_zero(self, x, eps: float = 1e-2) -> float:
        if abs(x) > eps:
            return x
        elif x >= 0:
            return eps
        else:
            return -eps

    def state_input_vector(self, v_S, ego_s, ego_d):
        # Paper: Automated Speed and Lane Change Decision Making using Deep Reinforcement Learning
        state_vector = np.zeros(4 * (1 + self.N_SPAWN_CARS + self.N_SPAWN_PEDESTRAINS))
        state_vector[0] = v_S / self.maxSpeed

        df_ego = closest([self.LANE_WIDTH * lane_n for lane_n in range(-1, 3)], ego_d)

        if df_ego == -3.5:
            lane_num = 'Lane_1'
        elif df_ego == 0:
            lane_num = 'Lane_2'
        elif df_ego == 3.5:
            lane_num = 'Lane_3'
        elif df_ego == 7:
            lane_num = 'Lane_4'

        Nearby_lane_info = {'Lane_1': [0, 1], 'Lane_2': [1, 0], 'Lane_3': [1, 1], 'Lane_4': [1, 0]}
        state_vector[1] = Nearby_lane_info[lane_num][0]
        state_vector[2] = Nearby_lane_info[lane_num][1]
        state_vector[3] = ego_d

        obj_mat = self.obj_info_simple()[1]

        obj_mat[0, :] = obj_mat[0, :] - ego_s
        obj_sorted_id = np.argsort(abs(obj_mat[0, :]))
        obj_mat_surr = obj_mat[:, obj_sorted_id][:, 0:8]

        for i in range(np.shape(obj_mat_surr)[1]):
            state_vector[(i + 1) * 4] = obj_mat_surr[0][i] / self.d_max_s
            state_vector[(i + 1) * 4 + 1] = obj_mat_surr[2][i] / self.obj_max_vs
            df_obj = closest([self.LANE_WIDTH * lane_n for lane_n in range(-1, 3)], obj_mat_surr[1][i])
            state_vector[(i + 1) * 4 + 2] = (df_obj - df_ego) / 3.5  # * (1 / 3)
            state_vector[(i + 1) * 4 + 3] = obj_mat_surr[1][i]

        walker_mat = self.walker_info_simple()[1]

        walker_mat[0, :] = walker_mat[0, :] - ego_s
        walker_sorted_id = np.argsort(abs(walker_mat[0, :]))
        walker_mat_surr = walker_mat[:, walker_sorted_id][:, 0:8]

        for i in range(np.shape(walker_mat_surr)[1]):
            state_vector[(i + 1 + self.N_SPAWN_CARS) * 4] = walker_mat_surr[0][i] / self.d_max_s
            state_vector[(i + 1 + self.N_SPAWN_CARS) * 4 + 1] = walker_mat_surr[3][i] / self.walker_max_vd
            df_walker = closest([self.LANE_WIDTH * lane_n for lane_n in range(-1, 3)], walker_mat_surr[1][i])
            state_vector[(i + 1 + self.N_SPAWN_CARS) * 4 + 2] = (df_walker - df_ego) / 3.5  # * (1 / 3)
            state_vector[(i + 1 + self.N_SPAWN_CARS) * 4 + 3] = walker_mat_surr[1][i]

        return state_vector

    def step(self, action):
        self.n_step += 1
        self.actor_enumerated_dict['EGO'] = {'NORM_S': [], 'NORM_D': [], 'S': [], 'D': [], 'SPEED': []}
        # normalized
        action = np.array(action)
        if len(action.shape) == 1:
            for i in action:
                q = i
                pass
        elif len(action.shape) == 2:
            for i in range(action.shape[0]):
                for j in range(action.shape[1]):
                    q = action[i][j]
                    pass
        else:
            print("Array dimensions greater than 2 are not handled.")
        # q = (q + 1.0) * 0.5
        # birds-eye view
        spectator = self.world_module.world.get_spectator()
        transform = self.ego.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(x=-10,z=20), carla.Rotation(pitch=-45)))
        # # third person view
        # spectator = self.world.get_spectator()
        # transform = ego_vehicle.get_transform()
        # spectator.set_transform(carla.Transform(transform.location + carla.Location(x=0, y=7, z=5), carla.Rotation(pitch=-20, yaw=-90)))

        """
                **********************************************************************************************************************
                *********************************************** Motion Planner *******************************************************
                **********************************************************************************************************************
        """
        temp = [self.ego.get_velocity(), self.ego.get_acceleration()]
        speed = get_speed(self.ego)  # Compute speed of a vehicle in Kmh
        acc_vec = self.ego.get_acceleration()
        acc = math.sqrt(acc_vec.x ** 2 + acc_vec.y ** 2)
        psi = math.radians(self.ego.get_transform().rotation.yaw)
        angular_velocity = self.ego.get_angular_velocity()
        acc_angular = math.sqrt(angular_velocity.x ** 2 + angular_velocity.y ** 2 + angular_velocity.z ** 2)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y, speed, acc, psi, temp, self.max_s]

        if self.n_step == 1:
            fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(ego_state, self.f_idx,
                                                                                           df_n=0, Tf=3, Vf_n=0)
        else:
            self.ref_idx = closest_wp_idx_ref(ego_state, self.fpath, self.f_idx)
            ego_state_ref = [self.fpath.x[self.ref_idx], self.fpath.y[self.ref_idx], speed, acc,
                             self.fpath.yaw[self.ref_idx], temp, self.max_s]
            fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(ego_state_ref, self.f_idx,
                                                                                           df_n=0, Tf=3, Vf_n=0)

        self.fpath = fpath
        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        self.f_idx = 0
        collision = False
        vx_ego = self.ego.get_velocity().x
        vy_ego = self.ego.get_velocity().y
        ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]
        # ego_d = fpath.d[self.f_idx]
        ego_d = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[3]
        v_S, v_D = velocity_inertial_to_frenet(ego_s, vx_ego, vy_ego, self.motionPlanner.csp)
        psi_Frenet = get_obj_S_yaw(psi, ego_s, self.motionPlanner.csp)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                     math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]
        self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
        ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]
        vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)
        norm_d = round((ego_d + self.LANE_WIDTH) / (3 * self.LANE_WIDTH), 2)
        ego_s_list = [ego_s for _ in range(self.look_back)]
        ego_d_list = [ego_d for _ in range(self.look_back)]
        self.actor_enumerated_dict['EGO'] = {'NORM_S': [0], 'NORM_D': [norm_d], 'S': ego_s_list, 'D': ego_d_list,
                                             'SPEED': [speed]}
        obj_info = self.obj_info()
        walker_dict, walker_info_Mux = self.walker_info_simple()
        # if self.n_step == 180:
        #     print("180")
        # print(self.n_step)
        ref_left = frenet_to_inertial(self.fpath.s[29], self.fpath.d[29] - 3.5, self.motionPlanner.csp)
        ref_right = frenet_to_inertial(self.fpath.s[29], self.fpath.d[29] + 7, self.motionPlanner.csp)

        # terminal
        self.Input, MPC_unsolved, x_m = self.lon_lat_controller_ipopt.calc_input(
            x_current=[ego_state[0],ego_state[1], ego_state[2]],
            obj_info=obj_info,
            walker_info=walker_dict,
            ref=np.array(
                [self.fpath.x[29], self.fpath.y[29], self.fpath.yaw[29], self.fpath.s[29], self.fpath.d[29]]),
            ref_left=np.array([ref_left[0], ref_left[1], ref_left[3]]),
            ref_right=np.array([ref_right[0], ref_right[1], ref_right[3]]),
            u_last=self.u_last, csp=self.motionPlanner.csp, fpath=fpath,
            q=q, ru=1, rdu=1)

        self.u_last = self.Input
        self.x_m = x_m

        delta_f = self.Input[1]
        target_speed = self.Input[0]
        cmdSpeed = target_speed * np.cos(psi_Frenet)
        steer = delta_f * 180.0 / 100.0 / np.pi  ##origin:70
        throttle_and_brake = self.PIDLongitudinalController.run_step(cmdSpeed)  # calculate control
        throttle_and_brake = throttle_and_brake[0]
        throttle = max(throttle_and_brake, 0)
        brake = min(throttle_and_brake, 0)
        # print('steer =', steer)
        # print('speed =', speed)
        vehicle_control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )

        self.ego.apply_control(vehicle_control)

        '''******   Lane change judgement    ******'''
        # lanechange should be set true if there is a lane change
        if -4.35 <= ego_d < -1.75:
            lane = -1
        elif -1.75 <= ego_d < 1.75:
            lane = 0
        elif 1.75 <= ego_d < 5.25:
            lane = 1
        elif 5.25 <= ego_d < 8.75:
            lane = 2
        else:
            lane = -2
            off_the_road = True

        lanechange = True if self.lane_last != lane else False
        self.lane_last = lane

        '''******   Update carla world    ******'''
        self.module_manager.tick()  # Update carla world

        if self.auto_render:
            self.render()
        self.world_module.collision_sensor.reset()
        collision_hist = self.world_module.get_collision_history()
        if any(collision_hist):
            collision = True

        """
                **********************************************************************************************************************
                *********************************************** Draw Waypoints *******************************************************
                **********************************************************************************************************************
        """

        if self.world_module.args.play_mode != 0:
            for i in range(len(fpath.x)):
                self.world_module.points_to_draw['fpath wp {}'.format(i)] = [
                    carla.Location(x=fpath.x[i], y=fpath.y[i]),
                    'COLOR_ALUMINIUM_0']
        #
        #     self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
        #     self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
        #     self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])

        if self.world_module.args.play_mode != 0:
            for i in range(len(self.x_m)):
                self.world_module.points_to_draw['x_m wp {}'.format(i)] = [
                    carla.Location(x=self.x_m[i, 0], y=self.x_m[i, 1]),
                    'COLOR_SCARLET_RED_0']

        self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
        # self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
        # self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])

        """
                **********************************************************************************************************************
                *********************************************** Data Log *******************************************************
                **********************************************************************************************************************
        """
        '''******   Data Record   ******'''

        if self.log_flag:
            # 保存数据
            info_ego = ('ego_s', ego_s, 'ego_d', ego_d)
            self.log.data_record(info_ego, 'ego_info')
            info_vehicle_1 = (
                'info_vehicle_1_s', obj_info['Obj_frenet'][0][0], 'info_vehicle_1_d', obj_info['Obj_frenet'][0][1])
            self.log.data_record(info_vehicle_1, 'vehicle_info_1')
            info_vehicle_2 = (
                'info_vehicle_2_s', obj_info['Obj_frenet'][1][0], 'info_vehicle_2_d', obj_info['Obj_frenet'][1][1])
            self.log.data_record(info_vehicle_2, 'vehicle_info_2')
            info_vehicle_3 = (
                'info_vehicle_3_s', obj_info['Obj_frenet'][2][0], 'info_vehicle_3_d', obj_info['Obj_frenet'][2][1])
            self.log.data_record(info_vehicle_3, 'vehicle_info_3')
            info_vehicle_4 = (
                'info_vehicle_4_s', obj_info['Obj_frenet'][3][0], 'info_vehicle_4_d', obj_info['Obj_frenet'][3][1])
            self.log.data_record(info_vehicle_4, 'vehicle_info_4')
            info_vehicle_5 = (
                'info_vehicle_5_s', obj_info['Obj_frenet'][4][0], 'info_vehicle_5_d', obj_info['Obj_frenet'][4][1])
            self.log.data_record(info_vehicle_5, 'vehicle_info_5')
        else:
            pass

        ##画动态曲线
        # self.x.append(self.n_step)
        # self.y.append(self.Input_lon[0])        # self.ax.cla()  # clear plot
        # self.ax.plot(self.x, self.y, 'r', lw=1)  # draw line chart
        # # ax.bar(y, height=y, width=0.3) # draw bar chart
        # plt.pause(0.1)
        """
                **********************************************************************************************************************
                *********************************************** Reinforcement Learning ***********************************************
                **********************************************************************************************************************
        """

        '''******   State Design    ******'''
        state_vector = self.state_input_vector(v_S, ego_s, ego_d)
        for i in range(len(state_vector)):
            self.state[0][i] = state_vector[i]

        '''******   Reward Design   ******'''
        # # 碰撞惩罚
        if collision:
            reward_cl = -20  ## -100.0
        else:
            reward_cl = 1

        if lanechange:
            reward_lanechange = -2
        else:
            reward_lanechange = 0

        if off_the_road:
            reward_offTheRoad = -2
        else:
            reward_offTheRoad = 0

        if MPC_unsolved == True:
            reward_mpcNoResult = 0
        else:
            reward_mpcNoResult = -2

        reward_dis_vehicle = 0
        for i in range(self.N_SPAWN_CARS):
            d_s = self.state[0, (i + 1) * 4] * self.d_max_s
            d_d_f = self.state[0, (i + 1) * 4 + 2]
            d_d = abs(self.state[0, (i + 1) * 4 + 3] - self.state[0, 3])
            reward_dis_vehicle -= 30 / (d_s ** 2 + d_d ** 2)

        reward_dis_walker = 0
        for i in range(self.N_SPAWN_PEDESTRAINS):
            d_s = self.state[0, (i + 1 + self.N_SPAWN_CARS) * 4] * self.d_max_s
            d_d_f = self.state[0, (i + 1 + self.N_SPAWN_CARS) * 4 + 2]
            d_d = abs(self.state[0, (i + 1 + self.N_SPAWN_CARS) * 4 + 3] - self.state[0, 3])
            reward_dis_walker -= 30 / (d_s ** 2 + d_d ** 2)

        reward_speed = v_S * 3 / self.maxSpeed

        reward = reward_cl + reward_dis_vehicle + reward_dis_walker + reward_speed + reward_lanechange +reward_mpcNoResult

        obj_S_Mux = []
        for j in range(np.size(obj_info['Obj_actor'])):
            obj_S = obj_info['Obj_frenet'][j][0]
            obj_S_Mux.append(obj_S)

        if MPC_unsolved:
            self.u_last = np.zeros(self.u_last.shape)

        done = False
        if collision or self.n_step >= 400: #or ego_s > max(obj_S_Mux) + 10:
            self.u_last = np.zeros(self.u_last.shape)
            done = True

        info = {'reserved': 0}
        obs = state_vector
        return obs, reward, done, info

    def reset(self):
        self.vehicleController.reset()
        self.PIDLongitudinalController.reset()
        self.PIDLateralController.reset()
        self.world_module.reset()
        self.init_s = self.world_module.init_s
        self.init_d = self.world_module.init_d
        self.traffic_module.reset(self.init_s, self.init_d)
        self.motionPlanner.reset(self.init_s, self.world_module.init_d, df_n=0, Tf=3, Vf_n=0, optimal_path=False)
        self.f_idx = 0
        self.n_step = 0  # initialize episode steps count
        self.eps_rew = 0
        self.is_first_path = True

        # Ego starts to move slightly after being relocated when a new episode starts. Probably, ego keeps a fraction of previous acceleration after
        # being relocated. To solve this, the following procedure is needed.
        self.ego.set_simulate_physics(enabled=True)

        self.module_manager.tick()
        self.ego.set_simulate_physics(enabled=True)
        return np.zeros_like(self.observation_space.sample()[0, :])

    def begin_modules(self, args):
        # define and register module instances
        self.module_manager = ModuleManager()
        width, height = [int(x) for x in args.carla_res.split('x')]
        self.world_module = ModuleWorld(MODULE_WORLD, args, timeout=10.0, module_manager=self.module_manager,
                                        width=width, height=height)
        self.traffic_module = TrafficManager(MODULE_TRAFFIC, module_manager=self.module_manager)
        self.module_manager.register_module(self.world_module)
        self.module_manager.register_module(self.traffic_module)

        if args.play_mode:
            self.hud_module = ModuleHUD(MODULE_HUD, width, height, module_manager=self.module_manager)
            self.module_manager.register_module(self.hud_module)
            self.input_module = ModuleInput(MODULE_INPUT, module_manager=self.module_manager)
            self.module_manager.register_module(self.input_module)

        # generate and save global route if it does not exist in the road_maps folder
        if self.global_route is None:
            self.global_route = np.empty((0, 3))
            distance = 1
            for i in range(1520):
                wp = self.world_module.town_map.get_waypoint(carla.Location(x=406, y=-100, z=0.1),
                                                             project_to_road=True).next(distance=distance)[0]
                distance += 2

                self.global_route = np.append(self.global_route,
                                              [[wp.transform.location.x, wp.transform.location.y,
                                                wp.transform.location.z]], axis=0)
                # To visualize point clouds
                self.world_module.points_to_draw['wp {}'.format(wp.id)] = [wp.transform.location, 'COLOR_CHAMELEON_0']
            self.global_route = np.vstack([self.global_route, self.global_route[0, :]])
            np.save('road_maps/global_route_town04', self.global_route)
            # plt.plot(self.global_route[:, 0], self.global_route[:, 1])
            # plt.show()

        self.motionPlanner = MotionPlanner()

        # Start Modules
        self.motionPlanner.start(self.global_route)
        # solve Spline
        self.world_module.update_global_route_csp(self.motionPlanner.csp)
        self.traffic_module.update_global_route_csp(self.motionPlanner.csp)
        self.module_manager.start_modules()
        # self.motionPlanner.reset(self.world_module.init_s, self.world_module.init_d)

        self.ego = self.world_module.hero_actor

        self.ego_los_sensor = self.world_module.los_sensor
        self.vehicleController = VehiclePIDController(self.ego, args_lateral={'K_P': 1.5, 'K_D': 0.0, 'K_I': 0.0})
        self.PIDLongitudinalController = PIDLongitudinalController(self.ego, K_P=40.0, K_D=0.1, K_I=4.0)
        self.PIDLateralController = PIDLateralController(self.ego, K_P=1.5, K_D=0.0, K_I=0.0)
        self.IDM = IntelligentDriverModel(self.ego)

        self.module_manager.tick()  # Update carla world
        self.init_transform = self.ego.get_transform()
        # print(self.ego.get_velocity())

    def enable_auto_render(self):
        self.auto_render = True

    def render(self, mode='human', close=False):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        # print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
            self.traffic_module.destroy()
