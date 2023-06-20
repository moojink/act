from copy import deepcopy

import gym
import numpy as np

from r2d2.calibration.calibration_utils import load_calibration_info
from r2d2.camera_utils.info import camera_type_dict
from r2d2.camera_utils.wrappers.multi_camera_wrapper import MultiCameraWrapper
from r2d2.misc.parameters import hand_camera_id, nuc_ip
from r2d2.misc.server_interface import ServerInterface
from r2d2.misc.time import time_ms
from r2d2.misc.transformations import change_pose_frame


class DummyRobotEnv(gym.Env):
    def __init__(self, action_space="cartesian_velocity", camera_kwargs={}):
        # Initialize Gym Environment
        super().__init__()
        self.control_hz = 15

    def step(self, action):
        pass

    def reset(self, randomize=False):
        pass

    def get_observation(self):
        obs_dict = dict(
            timestamp=dict(),
            robot_state=dict(
                joint_positions=np.zeros((7,)),
                gripper_position=np.zeros((1,)),
            ),
            image=dict()
        )
        cam_serial_num = '138422074005'
        obs_dict['image'][cam_serial_num] = np.zeros((256,256,3))
        return obs_dict
