"""
Created by Indraneel on 11/02/2025

Teleoperate Lerobot

python scripts/lerobot_teleoperate.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=singles_inferno_dex_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=singles_inferno_dex_leader

"""

from dataclasses import dataclass
import time

from lerobot.utils.utils import init_logging
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config, Teleoperator, so101_leader
from lerobot.robots import RobotConfig, make_robot_from_config, Robot, so101_follower
from lerobot.configs import parser
from lerobot.processor import make_default_processors, RobotProcessorPipeline, RobotAction, RobotObservation
from lerobot.utils.robot_utils import busy_wait


@dataclass
class TeleoperateConfig:
    teleop: TeleoperatorConfig
    robot: RobotConfig
    fps: int = 60
    teleop_time_s: float |None = None
    display_data: bool = False


def teleop_loop(teleop: Teleoperator, 
                robot: Robot, 
                fps: int, 
                teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
                robot_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
                robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
                display_data: bool = False,
                duration: float | None = None):
    while True:
        loop_start = time.perf_counter()

        obs = robot.get_observation()

        raw_action = teleop.get_action()

        teleop_action = teleop_action_processor((raw_action, obs))

        robot_action_to_send = robot_action_processor((teleop_action, obs))

        _ = robot.send_action(robot_action_to_send)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)
        print(f"\ntime: {dt_s * 1e3:.2f}ms ({1 / dt_s:.0f} Hz)")


@parser.wrap()
def teleoperate(cfg: TeleoperateConfig):
    print("Starting teleoperate")
    init_logging()

    teleop = make_teleoperator_from_config(cfg.teleop)
    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    teleop.connect()
    robot.connect()

    try:
        teleop_loop(
            teleop=teleop,
            robot=robot,
            fps=cfg.fps,
            display_data=cfg.display_data,
            duration=cfg.teleop_time_s,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor
        )
    except KeyboardInterrupt:
        pass
    finally:
        teleop.disconnect()
        robot.disconnect()


def main():
    teleoperate()

if __name__ == "__main__":
    main()