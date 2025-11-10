"""
Created by Indraneel on 11/08/2025

python scripts/lerobot_inference.py \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=singles_inferno_dex_follower \
  --robot.cameras "{ wrist: {type: opencv, index_or_path: 2, width: 640, height: 480, fps: 30}, top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}}" \
  --dataset.single_task="Grasp popcorn and put it in the bin." \
  --dataset.repo_id=indraneelpatil/eval_lerobot_with_cameras_test \
  --dataset.episode_time_s=50 \
  --dataset.num_episodes=10 \
  --policy.path=lerobot/smolvla_base \
  --display_data=true

"""
from pathlib import Path
from dataclasses import dataclass
import time
from typing import Any

from lerobot.configs import parser
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.robots import RobotConfig, make_robot_from_config, Robot
from lerobot_record import DatasetRecordConfig, RecordConfig
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging, get_safe_torch_device
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data
from lerobot.utils.control_utils import sanity_check_dataset_name, init_keyboard_listener, predict_action
from lerobot.processor import make_default_processors, PolicyAction, PolicyProcessorPipeline, RobotAction, RobotProcessorPipeline, RobotObservation
from lerobot.processor.rename_processor import rename_stats
from lerobot.datasets.utils import combine_feature_dicts, build_dataset_frame
from lerobot.datasets.lerobot_dataset import LeRobotDataset 
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.utils.constants import OBS_STR, ACTION


@safe_stop_image_writer
def inference_loop(
    robot: Robot,
    events: dict,
    fps: int,
    robot_action_processor: RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ],
    robot_observation_processor: RobotProcessorPipeline[
        RobotObservation, RobotObservation
    ],
    policy: PreTrainedPolicy,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction],
    dataset: LeRobotDataset,
    control_time_s: int | None = None,
    single_task: str | None = None,
    display_data: bool = False,
):
    
    if dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps")

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if events["exit_early"]:
            events["exit_early"] = False
            break

        # Get robot observation
        obs = robot.get_observation()

        # Apply pipeline
        obs_processed = robot_observation_processor(obs)

        observation_frame = build_dataset_frame(dataset.features, obs_processed, prefix=OBS_STR)

        # Get action from policy
        action_values = predict_action(
            observation=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
        )

        action_names = dataset.features[ACTION]["names"]
        act_processed_policy: RobotAction = {
            f"{name}": float(action_values[i]) for i, name in enumerate(action_names)
        }

        action_values = act_processed_policy
        robot_action_to_send = robot_action_processor((act_processed_policy, obs))

        # Send action to robot
        _sent_action = robot.send_action(robot_action_to_send)

        # Write to dataset
        action_frame = build_dataset_frame(dataset.features, action_values, prefix=ACTION)
        frame = {**observation_frame, **action_frame, "task": single_task}
        dataset.add_frame(frame)

        if display_data:
            log_rerun_data(observation=obs_processed, action=action_values)

        dt_s = time.perf_counter() - start_loop_t
        busy_wait(1 / fps - dt_s)

        timestamp = time.perf_counter() - start_episode_t
        
    

@parser.wrap()
def inference(cfg: RecordConfig):
    init_logging()
    if cfg.display_data:
        init_rerun(session_name="inference")
    
    robot = make_robot_from_config(cfg.robot)

    teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

    dataset_features = combine_feature_dicts(
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

    sanity_check_dataset_name(cfg.dataset.repo_id, cfg.policy)
    dataset = LeRobotDataset.create(
        cfg.dataset.repo_id,
        cfg.dataset.fps,
        root=cfg.dataset.root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=cfg.dataset.video,
        image_writer_processes=cfg.dataset.num_image_writer_processes,
        image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
        batch_encoding_size=cfg.dataset.video_encoding_batch_size
    )

    # Load pretrained policy
    if cfg.policy is None:
        raise RuntimeError("Policy cannot be None")
    policy = make_policy(cfg.policy, ds_meta=dataset.meta)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
        }
    )

    robot.connect()

    listener, events = init_keyboard_listener()

    with VideoEncodingManager(dataset):
        recorded_episodes = 0
        while recorded_episodes < cfg.dataset.num_episodes and not events["stop_recording"]:
            print(f"Inference episode {dataset.num_episodes}")
            inference_loop(
                robot=robot,
                events=events,
                fps=cfg.dataset.fps,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                dataset=dataset,
                control_time_s=cfg.dataset.episode_time_s,
                single_task=cfg.dataset.single_task,
                display_data=cfg.display_data
            )
            print("Finished inference")

            time.sleep(10)


            if events["rerecord_episode"]:
                print("Rerecord episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue
                
            
            dataset.save_episode()
            recorded_episodes +=1
    
    print("Stop recording")

    robot.disconnect()


    if listener is not None:
        listener.stop()


def main():
    inference()

if __name__ == "__main__":
    main()