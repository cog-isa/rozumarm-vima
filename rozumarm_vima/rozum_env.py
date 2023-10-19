from typing import Callable

import numpy as np
import gym

from rozumarm_vima_utils.transform import rf_tf_c2r, map_tf_repr_c2r
from rozumarm_vima_utils.scene_renderer import VIMASceneRenderer
from rozumarm_vima_utils.robot import RozumArm
from rozumarm_vima_utils.scripts.detect_cubes import mock_detect_cubes

# from rozumarm_vima_utils.cv.test import detector


class RozumEnv(gym.Env):
    def __init__(self, detector):
        self.robot = RozumArm(use_mock_api=False)
        self.renderer = VIMASceneRenderer('sweep_without_exceeding')
        self.detector: Callable = detector
    
    def _get_obs(self):
        obj_posquats = self.detector()

        # map from cam to rozum
        obj_posquats = [
            (rf_tf_c2r(pos), map_tf_repr_c2r(quat)) for pos, quat in obj_posquats
        ]

        front_img, top_img = self.renderer.render_scene(obj_posquats)
        obs = {
            'rgb': {
                'front': np.transpose(front_img, axes=(2, 0, 1)),
                'top': np.transpose(top_img, axes=(2, 0, 1))
            },
            'ee': 1,  # spatula
            'segm': self.renderer.env._get_obs()['segm']
        }
        return obs
    
    def render(self):
        return None
    
    def reset(self):
        # self.render = lambda: None
        self.meta_info = self.renderer.env.meta_info
        self.prompt = self.renderer.env.prompt
        self.prompt_assets = self.renderer.env.prompt_assets

        self.renderer.reset(exact_num_swept_objects=1)
        obs = self._get_obs()
        return obs

    def step(self, action):
        pos_0 = action["pose0_position"]
        pos_1 = action["pose1_position"]
        eef_quat = self.robot.get_swipe_quat(pos_0, pos_1)
        
        x_compensation_bias = 0.03
        pos_0[0] += x_compensation_bias
        pos_1[0] += x_compensation_bias
        
        posquat_0 = (pos_0, eef_quat)
        posquat_1 = (pos_1, eef_quat)
        self.robot.swipe(posquat_0, posquat_1)

        obs = self._get_obs()
        return obs


class RozumSegmentationEnv(gym.Env):
    def __init__(self, segmentator, front_cam, top_cam):
        self.robot = RozumArm(use_mock_api=False)
        self.segmentator: Callable = segmentator
    
    def _get_obs(self):
        seg_mask = self.segmentator()


        front_img, top_img = self.renderer.render_scene(obj_posquats)
        obs = {
            'rgb': {
                'front': np.transpose(front_img, axes=(2, 0, 1)),
                'top': np.transpose(top_img, axes=(2, 0, 1))
            },
            'ee': 1,  # spatula
            'segm': self.renderer.env._get_obs()['segm']
        }
        return obs
    
    def render(self):
        return None
    
    def reset(self):
        # self.render = lambda: None
        self.meta_info = self.renderer.env.meta_info
        self.prompt = self.renderer.env.prompt
        self.prompt_assets = self.renderer.env.prompt_assets

        self.renderer.reset(exact_num_swept_objects=1)
        obs = self._get_obs()
        return obs

    def step(self, action):
        pos_0 = action["pose0_position"]
        pos_1 = action["pose1_position"]
        eef_quat = self.robot.get_swipe_quat(pos_0, pos_1)
        
        x_compensation_bias = 0.03
        pos_0[0] += x_compensation_bias
        pos_1[0] += x_compensation_bias
        
        posquat_0 = (pos_0, eef_quat)
        posquat_1 = (pos_1, eef_quat)
        self.robot.swipe(posquat_0, posquat_1)

        obs = self._get_obs()
        return obs