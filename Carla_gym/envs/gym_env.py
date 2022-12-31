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
from MPC.MPC_controller_yundongxue import MPC_controller_yundongxue
from MPC.parameter_config import MPC_lon_Config
from MPC.parameter_config_0 import MPC_Config_0
from MPC.MPC_controller_lon import MPC_controller_lon
from MPC.MPC_controller_lat import MPC_controller_lat
from MPC.MPC_controller_lon_lat import MPC_controller_lon_lat
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


def lamp(v, x, y):
    return y[0] + (v - x[0]) * (y[1] - y[0]) / (x[1] - x[0] + 1e-10)


def cal_lat_error(waypoint1, waypoint2, vehicle_transform):
    """
    Estimate the steering angle of the vehicle based on the PID equations

    :param waypoint: target waypoint [x, y]
    :param vehicle_transform: current transform of the vehicle
    :return: lat_error
    """
    v_begin = vehicle_transform.location
    v_end = v_begin + carla.Location(x=math.cos(math.radians(vehicle_transform.rotation.yaw)),
                                     y=math.sin(math.radians(vehicle_transform.rotation.yaw)))
    v_vec_0 = np.array(
        [math.cos(math.radians(vehicle_transform.rotation.yaw)), math.sin(math.radians(vehicle_transform.rotation.yaw)),
         0.0])
    v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
    w_vec = np.array([waypoint2[0] -
                      waypoint1[0], waypoint2[1] -
                      waypoint1[1], 0.0])
    lat_error = math.acos(np.clip(np.dot(w_vec, v_vec) /
                                  (np.linalg.norm(w_vec) * np.linalg.norm(v_vec)), -1.0, 1.0))

    return lat_error


class CarlagymEnv(gym.Env):

    # metadata = {'render.modes': ['human']}
    def __init__(self):

        self.Input = None
        self.du_lon_last = None
        self.Input_lon = None
        self.Input_lt = None
        self.lat_error = None
        self.__version__ = "9.9.2"
        self.lon_param = MPC_lon_Config
        self.lat_param = MPC_lat_Config
        self.lon_lat_param = MPC_lon_lat_Config
        self.lon_controller = MPC_controller_lon(self.lon_param)
        self.lat_controller = MPC_controller_lat(self.lat_param)
        self.lon_lat_controller = MPC_controller_lon_lat(self.lon_lat_param)
        self.mpc_param = MPC_Config
        self.mpc_controller = MPC_controller_yundongxue(self.mpc_param)

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

        self.ref_path = xlrd.open_workbook(os.path.abspath('.') + '/tools/global_path_15.xlsx')
        self.ref_path = self.ref_path.sheets()[0]
        self.ref_path_x = self.ref_path.col_values(0)
        self.ref_path_y = self.ref_path.col_values(1)
        self.ref_path_phi = self.ref_path.col_values(2)
        self.ref_path_x = self.ref_path_x[1430:2028]  # 700-451    1800-1200   2135-1434
        self.ref_path_y = self.ref_path_y[1430:2028]
        self.ref_path_phi = self.ref_path_phi[1430:2028]

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
        self.acceleration_ = 0
        self.eps_rew = 0
        self.u_last = np.array([[0.0], [0.0]])
        self.u_lon_last = 0.0
        self.u_lon_llast = 0.0
        self.u_lat_last = 0.0
        self.u_lat_llast = 0.0
        self.fig, self.ax = plt.subplots()
        self.x = []
        self.y = []
        self.obj_frenet_history = np.zeros([1, 6])
        self.obj_cartesian_history = np.zeros([1, 8])

        self.motionPlanner = None
        self.vehicleController = None
        self.PIDLongitudinalController = None
        self.PIDLateralController = None

        if float(cfg.CARLA.DT) > 0:
            self.dt = float(cfg.CARLA.DT)
        else:
            self.dt = 0.05

        action_low = 0.1
        action_high = 0.9
        self.acton_dim = (1, 1)
        self.action_space = spaces.Box(action_low, action_high, shape=self.acton_dim, dtype='float32')
        self.obs_dim = (1, 3)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=self.obs_dim, dtype='float32')

        # data_record
        # self.log = data_collection()

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

    # def obj_info(self):
    #     """
    #     Frenet:  [s,d,v_s, v_d, phi_Frenet]
    #     """
    #     # others_s = np.zeros(self.N_SPAWN_CARS)
    #     # others_d = np.zeros(self.N_SPAWN_CARS)
    #     # others_v_S = np.zeros(self.N_SPAWN_CARS)
    #     # others_v_D = np.zeros(self.N_SPAWN_CARS)
    #     # others_phi_Frenet = np.zeros(self.N_SPAWN_CARS)
    #     #
    #     # for i, actor in enumerate(self.traffic_module.actors_batch):
    #     #     act_s, act_d, act_v_S, act_v_D, act_psi_Frenet = actor['Obj_Frenet_state']
    #     #     others_s[i] = act_s
    #     #     others_d[i] = act_d
    #     #     others_v_S[i] = act_v_S
    #     #     others_v_D[i] = act_v_D
    #     #     others_phi_Frenet[i] = act_psi_Frenet
    #     # obj_info_Mux = np.vstack((others_s, others_d, others_v_S, others_v_D, others_phi_Frenet))
    #     others_s = np.zeros(self.N_SPAWN_CARS)
    #     others_d = np.zeros(self.N_SPAWN_CARS)
    #     others_v_S = np.zeros(self.N_SPAWN_CARS)
    #     others_v_D = np.zeros(self.N_SPAWN_CARS)
    #     others_phi_Frenet = np.zeros(self.N_SPAWN_CARS)
    #     others_x = np.zeros(self.N_SPAWN_CARS)
    #     others_y = np.zeros(self.N_SPAWN_CARS)
    #     others_z = np.zeros(self.N_SPAWN_CARS)
    #     others_psi = np.zeros(self.N_SPAWN_CARS)
    #     others_delta_f = np.zeros(self.N_SPAWN_CARS)
    #     others_speed = np.zeros(self.N_SPAWN_CARS)
    #     for i, actor in enumerate(self.traffic_module.actors_batch):
    #         act_s, act_d, act_v_S, act_v_D, act_psi_Frenet, x, y, z, psi, delta_f, speed = actor['Obj_Frenet_state']
    #         others_s[i] = act_s
    #         others_d[i] = act_d
    #         others_v_S[i] = act_v_S
    #         others_v_D[i] = act_v_D
    #         others_phi_Frenet[i] = act_psi_Frenet
    #         others_x[i] = x
    #         others_y[i] = y
    #         others_z[i] = z
    #         others_psi[i] = psi
    #         others_delta_f[i] = delta_f
    #         others_speed[i] = speed
    #     obj_info_Mux = np.vstack((others_s, others_d, others_v_S, others_v_D, others_phi_Frenet, others_x, others_y,
    #                               others_z, others_psi, others_delta_f, others_speed))

    # return obj_info_Mux

    def state_input_vector(self, v_S, ego_s):
        # Paper: Automated Speed and Lane Change Decision Making using Deep Reinforcement Learning
        obj_mat = self.obj_info()
        state_vector = np.zeros(3)

        # No normalized
        state_vector[0] = obj_mat[0] - ego_s
        state_vector[1] = obj_mat[2] - v_S
        state_vector[2] = v_S

        # #Normalized
        # state_vector[0] = (obj_mat[0] - ego_s) / 100
        # state_vector[1] = np.clip(lamp(obj_mat[2] - v_S, [-20, 10], [0, 1]), 0, 1)
        # state_vector[2] = np.clip(v_S / 20, 0, 1)

        return state_vector

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

        for i, actor in enumerate(self.traffic_module.actors_batch):
            obj_idx = i
            obj_actor[obj_idx] = actor['Actor']
            obj_frenet[obj_idx] = actor['Obj_Frenet_state']
            obj_cartesian[obj_idx] = actor['Obj_Cartesian_state']
        # if self.n_step <= 30:
        #     self.obj_frenet_history.append(obj_frenet)
        #     self.obj_cartesian_history.append(obj_cartesian)
        # else:
        #     j = self.n_step % 30 - 1
        #     self.obj_frenet_history[j] = obj_frenet
        #     self.obj_cartesian_history[j] = obj_cartesian

        obj_dict = ({'Obj_actor': obj_actor, 'Obj_frenet': obj_frenet, 'Obj_cartesian': obj_cartesian,
                     # 'Obj_frenet_hist': self.obj_frenet_history, 'Obj_cartesian_hist': self.obj_cartesian_history
                     })

        return obj_dict

    def not_zero(self, x, eps: float = 1e-2) -> float:
        if abs(x) > eps:
            return x
        elif x >= 0:
            return eps
        else:
            return -eps

    def step(self, action):
        self.n_step += 1
        self.u_lon_llast = self.u_lon_last
        self.u_lat_llast = self.u_lat_last
        # birds-eye view
        spectator = self.world_module.world.get_spectator()
        transform = self.ego.get_transform()
        spectator.set_transform(carla.Transform(transform.location + carla.Location(z=50), carla.Rotation(pitch=-90)))
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

        if 100 <= self.n_step <= 173:
            df_n = 3.5
        elif 250 <= self.n_step <= 273:
            df_n = 0
        else:
            df_n = 0

        # df_n = 0
        if self.n_step == 1:
            fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(ego_state, self.f_idx,
                                                                                           # df_n=0,
                                                                                           df_n=df_n,
                                                                                           Tf=3,
                                                                                           Vf_n=0)
        else:
            self.ref_idx = closest_wp_idx_ref(ego_state, self.fpath, self.f_idx)
            ego_state_ref = [self.fpath.x[self.ref_idx], self.fpath.y[self.ref_idx], speed, acc,
                             self.fpath.yaw[self.ref_idx], temp, self.max_s]
            fpath, self.lanechange, off_the_road = self.motionPlanner.run_step_single_path(ego_state_ref, self.f_idx,
                                                                                           # df_n=0,
                                                                                           df_n=df_n,
                                                                                           Tf=3, Vf_n=0)

        self.fpath = fpath
        """
                **********************************************************************************************************************
                ************************************************* Controller *********************************************************
                **********************************************************************************************************************
        """
        self.f_idx = 1
        collision = False
        vx_ego = self.ego.get_velocity().x
        vy_ego = self.ego.get_velocity().y
        ego_s = self.motionPlanner.estimate_frenet_state(ego_state, self.f_idx)[0]
        ego_d = fpath.d[self.f_idx]
        v_S, v_D = velocity_inertial_to_frenet(ego_s, vx_ego, vy_ego, self.motionPlanner.csp)
        psi_Frenet = get_obj_S_yaw(psi, ego_s, self.motionPlanner.csp)
        K_Frenet = get_calc_curvature(ego_s, self.motionPlanner.csp)
        ego_state = [self.ego.get_location().x, self.ego.get_location().y,
                     math.radians(self.ego.get_transform().rotation.yaw), 0, 0, temp, self.max_s]
        self.f_idx = closest_wp_idx(ego_state, fpath, self.f_idx)
        cmdWP = [fpath.x[self.f_idx], fpath.y[self.f_idx]]
        cmdWP2 = [fpath.x[self.f_idx + 1], fpath.y[self.f_idx + 1]]
        obj_info_Mux = self.obj_info()
        ego_init_d, ego_target_d = fpath.d[0], fpath.d[-1]
        vehicle_ahead = self.get_vehicle_ahead(ego_s, ego_d, ego_init_d, ego_target_d)

        print(self.n_step)

        ##MPC_LAT+PID
        # q = action[0] / 10
        # ru = 1
        # rdu = 1
        # self.lat_error = cal_lat_error(cmdWP, cmdWP2, transform)
        # self.Input_lon = self.lon_controller.calc_input(S_obj=obj_info_Mux['Obj_frenet'][0][0], v_obj=obj_info_Mux['Obj_frenet'][0][2],
        #                                                 x_current_lon=np.array([[ego_s], [v_S]]),
        # #                                                 u_lon_last=self.u_lon_last, q=q, ru=ru, rdu=rdu)
        # self.Input_lat = self.lat_controller.calc_input(D_ref=obj_info_Mux['Obj_frenet'][0][1],
        #                                                 vx=v_S,
        #                                                 vy=v_D,
        #                                                 x_current_lat=np.array([[psi], [ego_d]]),
        #                                                 u_lat_last=self.u_lat_last,
        #                                                 cur=K_Frenet)
        #
        # self.u_lat_last = self.Input_lat
        # vehicle_ahead = obj_info_Mux['Obj_actor'][0]
        # cmdSpeed = self.IDM.run_step(vd=self.targetSpeed, vehicle_ahead=vehicle_ahead)
        #
        # control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
        # throttle = control.throttle
        # brake = control.brake
        # steer = self.Input_lat * np.pi / 35
        # if self.n_step == 30:
        #     print('2202020')
        # vehicle_control = carla.VehicleControl(
        #     throttle=float(throttle),
        #     steer=float(steer),
        #     brake=float(brake),
        #     hand_brake=False,
        #     reverse=False,
        #     manual_gear_shift=False
        # )
        #
        # self.ego.apply_control(vehicle_control)

        # direct_x = self.ego.get_transform().get_forward_vector().x
        # direct_y = self.ego.get_transform().get_forward_vector().y
        #
        # normal_dir_x = direct_x / math.sqrt(direct_x ** 2 + direct_y ** 2)
        # normal_dir_y = direct_y / math.sqrt(direct_x ** 2 + direct_y ** 2)
        #
        # current_acc = acc_vec.x * normal_dir_x + acc_vec.y * normal_dir_y
        #
        # target_acc = self.Input_lon[0]
        # cmdSpeed = get_speed(self.ego) + float(target_acc) * self.dt
        #
        # self.u_lon_last = target_acc
        # self.u_lat_last = self.Input_lat[0]
        #
        # self.du_lon_last = (self.u_lon_last - self.u_lon_llast) / self.dt
        # MPC_no_answer = self.Input_lon[1]
        #
        # control = self.vehicleController.run_step_acc_2_wp(self.Input_lon[0], current_acc, cmdWP,
        #                                                    cmdWP2)  # calculate control
        # # control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
        #
        # self.ego.apply_control(control)  # apply control

        # ##MPC_LON+PID
        # q = action[0] / 10
        # ru = 1
        # rdu = 1
        # self.lat_error = cal_lat_error(cmdWP, cmdWP2, transform)
        # self.Input_lon = self.lon_controller.calc_input(S_obj=obj_info_Mux['Obj_frenet'][0][0], v_obj=obj_info_Mux['Obj_frenet'][0][2],
        #                                                 x_current_lon=np.array([[ego_s], [v_S]]),
        #                                                 u_lon_last=self.u_lon_last, q=q, ru=ru, rdu=rdu)
        #
        # direct_x = self.ego.get_transform().get_forward_vector().x
        # direct_y = self.ego.get_transform().get_forward_vector().y
        #
        # normal_dir_x = direct_x / math.sqrt(direct_x ** 2 + direct_y ** 2)
        # normal_dir_y = direct_y / math.sqrt(direct_x ** 2 + direct_y ** 2)
        #
        # current_acc = acc_vec.x * normal_dir_x + acc_vec.y * normal_dir_y
        #
        # target_acc = self.Input_lon[0]
        # cmdSpeed = get_speed(self.ego) + float(target_acc) * self.dt
        #
        # self.u_lon_last = target_acc
        #
        # self.du_lon_last = (self.u_lon_last - self.u_lon_llast) / self.dt
        # MPC_no_answer = self.Input_lon[1]
        #
        # control = self.vehicleController.run_step_acc_2_wp(self.Input_lon[0], current_acc, cmdWP,
        #                                                    cmdWP2)  # calculate control
        # # control = self.vehicleController.run_step_2_wp(cmdSpeed, cmdWP, cmdWP2)  # calculate control
        #
        # self.ego.apply_control(control)  # apply control

        ##MPCALL

        # obj_x = obj_info_Mux['Obj_cartesian'][0][0]
        # obj_y = obj_info_Mux['Obj_cartesian'][0][1]
        # obj_phi = obj_info_Mux['Obj_cartesian'][0][4]
        # obj_s = obj_info_Mux['Obj_frenet'][0][0]
        # obj_d = obj_info_Mux['Obj_frenet'][0][1]
        # obj_phi_Frenet = obj_info_Mux['Obj_frenet'][0][4]
        # obj_speed = obj_info_Mux['Obj_cartesian'][0][5]
        # obj_delta_f = obj_info_Mux['Obj_cartesian'][0][6]
        obj_info = self.obj_info()
        obj_frenet = obj_info['Obj_frenet']
        obj_cartesian = obj_info['Obj_cartesian']
        if self.n_step == 204:
            print("180")

        # self.f_ref_idx = closest_wp_idx_ref(ego_state, self.ref_path_x,self.ref_path_y, self.f_idx)
        # ref_path_x = self.ref_path_x[self.n_step:self.n_step + 30]
        # ref_path_y = self.ref_path_y[self.n_step:self.n_step + 30]
        # ref_path_phi = self.ref_path_phi[self.n_step:self.n_step + 30]
        ref_path_x = self.ref_path_x[self.n_step:self.n_step + 30]
        ref_path_y = self.ref_path_y[self.n_step:self.n_step + 30]
        ref_path_phi = self.ref_path_phi[self.n_step:self.n_step + 30]
        # if self.n_step%30 == 1:
        #     self.fpath = fpath
        # ref_path_x = self.fpath.x
        # ref_path_y = self.fpath.y
        # ref_path_phi = self.fpath.yaw
        # i = self.n_step%30
        # ref_path_x = ref_path_x[i:i+30]
        # ref_path_y = ref_path_y[i:i + 30]
        # ref_path_phi = ref_path_phi[i:i + 30]
        ####MPC_lon_lat
        # plt.plot(self.n_step*0.1, speed,'o')
        # plt.plot(fpath.t,fpath.s_d)
        plt.plot(fpath.x, fpath.y, 'o')
        plt.plot(ego_state[0], ego_state[1], '*')
        plt.pause(0.001)
        plt.cla()

        obj_x = obj_info['Obj_cartesian'][0][0]
        obj_y = obj_info['Obj_cartesian'][0][1]
        obj_phi = obj_info['Obj_cartesian'][0][4]
        obj_speed = obj_info['Obj_cartesian'][0][5]
        obj_delta_f = obj_info['Obj_cartesian'][0][6]

        self.Input, MPC_unsolved = self.lon_lat_controller.calc_input(
            x_current=np.array([[ego_state[0]], [ego_state[1]], [ego_state[2]]]),
            ref=np.array([obj_x, obj_y, obj_phi, obj_speed, obj_delta_f]),
            fpath=np.array([fpath.x[0:40], fpath.y[0:40], fpath.yaw[0:40]]),
            # fpath=np.array([ref_path_x, ref_path_y, ref_path_phi]),
            u_last=self.u_last,
            q=100, ru=1, rdu=1)

        self.u_last = self.Input
        target_speed = self.Input[0]
        cmdSpeed = target_speed * np.cos(psi_Frenet)
        # steer = self.Input[0][1] * np.pi / 35
        steer = self.Input[1] * 180.0 / 70.0 / np.pi
        # steer = self.Input[0][1] * 180.0 / 500.0 / np.pi
        throttle_and_brake = self.PIDLongitudinalController.run_step(cmdSpeed)  # calculate control
        throttle_and_brake = throttle_and_brake[0]
        throttle = max(throttle_and_brake, 0)
        brake = min(throttle_and_brake, 0)
        print('steer =', steer)
        print('speed =', speed)
        vehicle_control = carla.VehicleControl(
            throttle=float(throttle),
            steer=float(steer),
            brake=float(brake),
            hand_brake=False,
            reverse=False,
            manual_gear_shift=False
        )

        self.ego.apply_control(vehicle_control)

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
                self.world_module.points_to_draw['path wp {}'.format(i)] = [
                    carla.Location(x=fpath.x[i], y=fpath.y[i]),
                    'COLOR_ALUMINIUM_0']

            # if self.world_module.args.play_mode != 0:
            #     for i in range(len(self.ref_path_x)):
            #         self.world_module.points_to_draw['path wp {}'.format(i)] = [
            #             carla.Location(x=self.ref_path_x[i], y=self.ref_path_y[i]),
            #             'COLOR_ALUMINIUM_0']
            self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
        #     self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
        #     self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])

        # if self.world_module.args.play_mode != 0:
        #     for i in range(len(self.ref_path_x)):
        #         self.world_module.points_to_draw['path wp {}'.format(i)] = [
        #             carla.Location(x=self.ref_path_x[i], y=self.ref_path_y[i]),
        #             'COLOR_ALUMINIUM_0']

        # self.world_module.points_to_draw['ego'] = [self.ego.get_location(), 'COLOR_SCARLET_RED_0']
        # self.world_module.points_to_draw['waypoint ahead'] = carla.Location(x=cmdWP[0], y=cmdWP[1])
        # self.world_module.points_to_draw['waypoint ahead 2'] = carla.Location(x=cmdWP2[0], y=cmdWP2[1])

        """
                **********************************************************************************************************************
                *********************************************** Data Log *******************************************************
                **********************************************************************************************************************
        """
        # 保存数据
        # info = ('u', target_acc, 'du', self.du_lon_last, 'Cur_acc', current_acc, 'q', q, 'ru', ru, 'rdu', rdu)
        # self.log.data_record(info, 'mpc')
        # info_vehicle = ('vs', v_S, 'vs_cmd', cmdSpeed, 'throttle', control.throttle)
        # self.log.data_record(info_vehicle, 'vehicle_info')
        # info_vehicle_2_v = ('ego_s', ego_s, 's_ahead', obj_info_Mux[0][0], 'vs', v_S, 'v_ahead', obj_info_Mux[2][0])
        # self.log.data_record(info_vehicle_2_v, 'vehicle_info_2_v')

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
        # state_vector = self.state_input_vector(v_S, ego_s)
        state_vector = np.zeros(3)
        # state_vector[0] = obj_info_Mux[0] - ego_s
        # state_vector[1] = obj_info_Mux[2] - v_S
        # state_vector[2] = v_S

        '''******   Reward Design   ******'''
        # # 碰撞惩罚
        if collision:
            # reward_cl = np.array([[-15.0]])
            reward_cl = self.collision_penalty  ## -10.0
        else:
            # reward_cl = np.array([[0.0]])
            reward_cl = 0.0

            # 速度优先在某范围
        # scaled_speed_l = lamp(state_vector[2], [0, self.minSpeed], [0, 1])  # 0-13.5
        # scaled_speed_m = lamp(state_vector[2], [self.minSpeed, 0.75 * self.maxSpeed], [0, 1])  # 13.5-14.5
        # scaled_speed_h = lamp(state_vector[2], [0.75 * self.maxSpeed, self.maxSpeed], [0, 1])  # 14.5-19.5
        # reward_hs_l = self.low_speed_reward  # 0.3
        # reward_hs_m = self.middle_speed_reward  # 4
        # reward_hs_h = self.high_speed_reward  # 0.3

        # 加速度
        # if state_vector[0] > 30 or state_vector[1] > 0:
        #     reward_acc = 1 * 1 / (abs(self.u_lon_last) + 0.1)
        # elif state_vector[0] <= 30 and state_vector[1] < 0:
        #     reward_acc = 2 * 1 * (-self.u_lon_last + 0.1)
        # else:
        #     reward_acc = 0

        # if state_vector[0] > 30
        #     reward_acc = 1 * 1 / (abs(self.u_lon_last)*0.05 + 0.1)
        # elif state_vector[0] <= 30 and state_vector[1] < 0:
        #     reward_acc = 2 * 1 * (-self.u_lon_last + 0.1)
        # else:
        #     reward_acc = 0
        # grid search

        # if state_vector[0] > 20 + 5:
        #     if abs(self.u_lon_last) <= 2:
        #         reward_acc = 10 * 1 / (abs(self.u_lon_last) + 0.1)
        #     elif abs(self.u_lon_last) <= 3:
        #         reward_acc = 1
        #     else:
        #         reward_acc = -50
        # elif state_vector[0] <= 20 + 5:
        #     reward_acc = 30 * (-(self.u_lon_last))
        # else:
        #     reward_acc = 0

        # if self.u_lon_last <= 0:
        #     reward_acc = 5
        # else:
        #     reward_acc = -2.0

        if -4 <= self.u_lon_last <= 3:
            if state_vector[0] > 20 + 2 or state_vector[1] >= 0:
                reward_acc = 30 - 10 * abs(self.u_lon_last)
            elif state_vector[0] < 20 - 2 and state_vector[1] <= -1:
                reward_acc = -10 * self.u_lon_last
            else:
                reward_acc = 0
        else:
            reward_acc = -100

        # # 加加速度
        # if abs(self.du_lon_last) <= 3:
        #     reward_d_acc = 0
        # else:
        #     reward_d_acc = -30.0

        # # 加加速度
        # if abs(self.du_lon_last) <= 1:
        #     reward_d_acc = 10 - 10 * abs(self.du_lon_last)
        # elif abs(self.du_lon_last) <= 3:
        #     reward_d_acc = 0
        # else:
        #     reward_d_acc = -100.0

        # if -3 <= self.du_lon_last <= 3:
        #     if state_vector[0] > 20 + 2 or state_vector[1] >= 0:
        #         reward_d_acc = 30 - 10 * abs(self.du_lon_last)
        #     elif state_vector[0] < 20 - 2 and state_vector[1] <= -1:
        #         reward_d_acc = -10 * self.du_lon_last
        #     else:
        #         reward_d_acc = 0
        # else:
        #     reward_d_acc = -100
        # if -3 <= self.du_lon_last <= 3:
        #     if state_vector[0] > 20 + 2:
        #         reward_d_acc = 150 - 50 * abs(self.du_lon_last)
        #     elif state_vector[0] < 20 - 2:
        #         reward_d_acc = -10 * self.du_lon_last
        #     else:
        #         reward_d_acc = 0
        # else:
        #     reward_d_acc = -500

        # if abs(self.du_lon_last) <= 1:
        #     reward_d_acc = 1.6666
        # elif abs(self.du_lon_last) <= 3:
        #     reward_d_acc = 0
        # else:
        #     reward_d_acc = -20.0

        # if abs(self.du_lon_last) <= 3:
        #     reward_d_acc = 0
        # else:
        #     reward_d_acc = -20.0

        # 跟车 ++++
        s_d = 60
        if abs(state_vector[0]) > s_d:
            reward_dis = -41.0
        else:
            reward_dis = 0.0

        # s_d = 20
        # if abs(state_vector[0]) < s_d:
        #     reward_dis = 8.0 * (float((state_vector[0] - s_d) / s_d))
        # elif abs(state_vector[0]) < (2 * s_d):
        #     reward_dis = 3.0 / (float((state_vector[0] - s_d) / s_d) + 1)
        # else:
        #     reward_dis = 0.0

        # reward = reward_cl + reward_hs_l * np.clip(scaled_speed_l, 0, 1) + reward_hs_h * np.clip(scaled_speed_h, 0, 1) \
        #          + reward_hs_m * np.clip(scaled_speed_m, 0, 1) + reward_dis + reward_d_acc + reward_acc
        reward = reward_cl + reward_acc + reward_dis
        done = False
        if collision or MPC_unsolved or self.n_step >= 2000 or state_vector[0] < -0.1 or state_vector[0] >= 100:
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
        print(self.ego.get_velocity())
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
        print(self.ego.get_velocity())

    def enable_auto_render(self):
        self.auto_render = True

    def render(self, mode='human', close=False):
        self.module_manager.render(self.world_module.display)

    def destroy(self):
        print('Destroying environment...')
        if self.world_module is not None:
            self.world_module.destroy()
            self.traffic_module.destroy()
