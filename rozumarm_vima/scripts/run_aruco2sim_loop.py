from typing import Callable

import numpy as np

from rozumarm_vima_utils.transform import rf_tf_c2r, map_tf_repr_c2r


N_SWEPT_OBJECTS = 2


def run_loop(r, robot, oracle, cubes_detector: Callable):
    """
    r: scene renderer
    """
    while True:
        obj_posquats = cubes_detector.detect()

        # map from cam to rozum
        obj_posquats = [
            (rf_tf_c2r(pos), map_tf_repr_c2r(quat)) for pos, quat in obj_posquats
        ]

        front_img, top_img = r.render_scene(obj_posquats)
        obs, *_ = r.env.step(action=None)

        action = oracle.act(obs)
        if action is None:
            print("ORACLE FAILED.")
            return

        clipped_action = {
            k: np.clip(v, r.env.action_space[k].low, r.env.action_space[k].high)
            for k, v in action.items()
        }

        pos_0 = clipped_action["pose0_position"]
        pos_1 = clipped_action["pose1_position"]
        eef_quat = robot.get_swipe_quat(pos_0, pos_1)
        
        # x_compensation_bias = 0.03
        # pos_0[0] += x_compensation_bias
        # pos_1[0] += x_compensation_bias
        
        posquat_0 = (pos_0, eef_quat)
        posquat_1 = (pos_1, eef_quat)
        robot.swipe(posquat_0, posquat_1)
        
        cmd = input("Press return for one more swipe, enter [s] to save videos, enter [q] to exit.\n")
        if cmd == "s":
            cubes_detector.release()
        if cmd == "s" or cmd == "q":
            print("Quitting...")
            break


class MockObjDetector:
    def __call__(self):
        obj_posquats = [
            ((0.3, -0.15), (0, 0, 0, 1)),
            ((0.0, -0.15), (0, 0, 0, 1))
        ]
        return obj_posquats


from rozumarm_vima_utils.scene_renderer import VIMASceneRenderer
def main():
    from rozumarm_vima_utils.robot import RozumArm

    r = VIMASceneRenderer('sweep_without_exceeding')
    oracle = r.env.task.oracle(r.env)
    robot = RozumArm(use_mock_api=False)
    
    r.reset(exact_num_swept_objects=N_SWEPT_OBJECTS)
    
    # detector = MockObjDetector()

    from rozumarm_vima.detectors import detector
    
    run_loop(r, robot, oracle, cubes_detector=detector)


if __name__ == '__main__':
    main()
