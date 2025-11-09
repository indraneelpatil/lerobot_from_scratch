"""
Created by Indraneel on 11/03/2025

Record dataset from Lerobot

python scripts/lerobot_record.py \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=singles_inferno_dex_follower \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=singles_inferno_dex_leader \
    --display_data=true \
    --dataset.repo_id=indraneelpatil/lerobot-teleop-no-cameras \
    --dataset.num_episodes=5 \
    --dataset.single_task="Grab the white box" \
    --dataset.push_to_hub=true \
    --dataset.episode_time_s=30 \
    --dataset.reset_time_s=10 

"""
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any
import logging

from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.configs import parser
from lerobot.robots import RobotConfig, make_robot_from_config, Robot, so101_follower
from lerobot.teleoperators import TeleoperatorConfig, make_teleoperator_from_config, Teleoperator, so101_leader
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import combine_feature_dicts, build_dataset_frame
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.utils.utils import init_logging
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.processor import make_default_processors, RobotProcessorPipeline, RobotAction, RobotObservation, PolicyProcessorPipeline, PolicyAction
from lerobot.utils.control_utils import (
    sanity_check_dataset_robot_compatibility,
    init_keyboard_listener,
    sanity_check_dataset_name
)
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import busy_wait



@dataclass
class DatasetRecordConfig:
    # Dataset identifier
    repo_id: str
    # Description of the task
    single_task: str
    # root directory where dataset will be stored
    root: str | Path | None = None
    fps: int = 30
    # Seconds of data recording for each episode
    episode_time_s: int | float = 60
    # Number of seconds to reset the environment
    reset_time_s: int | float = 60
    # Number of episodes
    num_episodes: int = 50
    # Encode frames in dataset into video
    video: bool = True
    # Upload dataset to hugging face
    push_to_hub:bool = True
    private: bool = False
    tags: list[str] | None = None
    # Number of subprocesses in each thread
    num_image_writer_processes: int = 0
    # Number of threads per camera, too many threads can block main thread too few might cause low cameras fps
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    rename_map: dict[str,str] = field(default_factory=dict)

    def __post_init__(self):
        if self.single_task is None:
            raise ValueError("You need to provide a task")


@dataclass
class RecordConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    teleop: TeleoperatorConfig | None = None
    policy: PreTrainedConfig| None = None
    display_data: bool = False
    play_sounds: bool = True
    # Resume recording on existing dataset
    resume:bool = False

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")
        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")
            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.teleop is None and self.policy is None:
            raise ValueError("Choose a policy, a teleoperator or both to control the robot")
        
    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        return ["policy"]

@safe_stop_image_writer
def record_loop(
    robot: Robot,
    events: dict,
    fps: int,
    teleop_action_processor: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    robot_action_processor: RobotProcessorPipeline[tuple[RobotAction,RobotObservation],RobotAction],
    robot_observation_processor: RobotProcessorPipeline[RobotObservation, RobotObservation],
    dataset: LeRobotDataset | None = None,
    teleop: Teleoperator | list[Teleoperator] | None = None,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[ PolicyAction, PolicyAction] | None = None,
    control_time_s: int | None=None,
    single_task: str | None = None,
    display_data:bool = False
):
    if dataset is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset.fps} != {fps})")
    
    teleop_arm = teleop_keyboard = None
    
    # TODO: Not handled multi teleop

    # TODO Policy stuff
    
    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Applies a pipeline to the raw robot observation, default is IdentityProcessor
        obs_processed = robot_observation_processor(obs)

        if policy is not None or dataset is not None:
            observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)
        
        if policy is None and isinstance(teleop, Teleoperator):
            act = teleop.get_action()

            act_processed_teleop = teleop_action_processor((act, obs))
        else:
            logging.info("No policy or teleoperator provided, skipping action generation")
            continue

        action_values = act_processed_teleop
        robot_action_to_send = robot_action_processor((act_processed_teleop, obs))

        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        if dataset is not None:
            action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
            frame = {**observation_frame, **action_frame, "task": single_task}
            dataset.add_frame(frame)
        
        if display_data:
            log_rerun_data(observation=obs_processed, action= action_values)
        
        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t




@parser.wrap()
def record(cfg: RecordConfig) -> LeRobotDataset:
    init_logging()
    if cfg.display_data:
        init_rerun(session_name="neel_recording")
    
    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop) if cfg.teleop else None

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    # Combines multiple dicts into one dict
    dataset_features = combine_feature_dicts(
        # Gets features from robot and transforms them with pipeline
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(
                action=robot.action_features
            ),
            use_videos=cfg.dataset.video
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=cfg.dataset.video
        )

    )

    # TODO Reloading existing dataset

    sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera*len(robot.cameras),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size
    )

    # Load pretrained policy TODO

    robot.connect()
    if teleop is not None:
        teleop.connect()

    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            print(f"Recording episode {dataset.num_episodes}")
            record_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                teleop=teleop,
                policy=None,
                preprocessor=None,
                postprocessor=None,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data
            )
            print(f"Finished recording")

            # Sleep for 10 seconds
            time.sleep(cfg.dataset.reset_time_s)

            if events["rerecord_episode"]:
                print("Re recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()
            recorded_episodes += 1
    

    print('Stop recording')

    robot.disconnect()
    if teleop is not None:
        teleop.disconnect()

    if listener is not None:
        listener.stop()

    if cfg.dataset.push_to_hub:
        dataset.push_to_hub(tags=cfg.dataset.tags, private=cfg.dataset.private)

    print('Exiting')

    return dataset

def main():
    record()


if __name__ == "__main__":
    main()