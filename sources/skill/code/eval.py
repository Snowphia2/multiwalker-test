from harl.runners.off_policy_base_runner import OffPolicyBaseRunner
import rich.pretty
import time
import wandb
import hydra
import rich
from harl.runners import RUNNER_REGISTRY
from datetime import datetime
from .types.task.eval_type import EvalConfig
from .train import _to_dict
from .train import _to_harl_dict as _train_to_harl_dict
import os
from harl.runners.on_policy_ma_runner import OnPolicyMARunner
import json
import atexit
from harl.envs.pettingzoo_mw.pettingzoo_mw_logger import PettingZooMWLogger
from moviepy.editor import VideoFileClip
import imageio
from typing import cast

os.environ["SDL_VIDEODRIVER"] = "dummy"


def _to_harl_dict(
    cfg: EvalConfig,
):
    (
        algo_dict,
        env_dict,
        basic_info,
        algorithm_name,
        env_name,
        scenario_name,
        run_group,
        save_group,
    ) = _train_to_harl_dict(cfg)

    rich.print(algo_dict)

    if (
        hasattr(cfg.eval_scenario, "env_tweak")
        and cfg.eval_scenario.env_tweak is not None
    ):
        eval_env_tweak = _to_dict(cfg.eval_scenario.env_tweak)
        for key in eval_env_tweak.keys():
            if not key.startswith("_") and key != "tweak_types":
                env_dict[key] = eval_env_tweak[key]
                print(f"eval_env_tweak: {key} = {eval_env_tweak[key]}")

    if hasattr(cfg.eval_scenario, "events") and cfg.eval_scenario.events is not None:
        env_dict["events"] = _to_dict(cfg.eval_scenario)["events"]

    return (
        algo_dict,
        env_dict,
        basic_info,
        algorithm_name,
        env_name,
        scenario_name,
        run_group,
        save_group,
    )


def export_gif(config_name, frames_arr, rewards_arr):
    # 文件夹

    rich.print(f"Exporting gif for {config_name}")
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos/"
    )
    gif_folder = os.path.join(gif_dir, f"{config_name}")
    os.makedirs(gif_folder, exist_ok=True)

    # rich.print(frames_arr)
    for i, frames in enumerate(frames_arr):
        # 1. gif生成
        rewards = rewards_arr[i]
        gif_path = os.path.join(
            gif_folder,
            f"{config_name}_[{rewards:.2f}]_{i}.gif",
        )
        imageio.mimwrite(
            gif_path,
            frames,
            duration=10,
        )

        # 3. 视频生成
        clip = VideoFileClip(gif_path)
        clip.write_videofile(
            os.path.join(
                gif_folder,
                f"{config_name}_[{rewards:.2f}]_{i}.mp4",
            ),
            codec="libx264",
            logger=None,
        )

        # 删除前面的gif
        os.remove(gif_path)


def eval(
    config: EvalConfig,
):
    rich.print(f"Evaluation started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 0. 处理参数
    (
        algo_dict,
        env_dict,
        basic_info,
        algorithm_name,
        env_name,
        scenario_name,
        run_group,
        save_group,
    ) = _to_harl_dict(config)

    # 1. 加载模型
    env_folder = ""
    if env_name == "pettingzoo_mw":
        env_folder = "multiwalker"
    elif env_name == "pettingzoo_mw_llm":
        env_folder = "multiwalker"
    elif env_name == "sumo":
        env_folder = "sumo"
    model_path = f"./results/models/{save_group}/{env_name}/{env_folder}/{algorithm_name}/[{algorithm_name}]<{scenario_name}>"
    rich.print(f"Loading model from {model_path}")

    # 2. 通用的env_tweak方法
    name_suffix = ""
    rich.print(config.environment.env_tweak.tweak_types)
    tweak_types = config.environment.env_tweak.tweak_types
    if env_name == "pettingzoo_mw":
        tweak_types = ["n_walkers", *sorted(config.environment.env_tweak.tweak_types)]
    env_tweaks = _to_dict(config.environment.env_tweak)
    for key in tweak_types:
        if not key.startswith("_"):
            name_suffix += f"<{key}={env_tweaks.get(key, None)}>"
    model_path += name_suffix

    seed_folder = next(
        folder for folder in os.listdir(model_path) if folder.startswith("seed-")
    )

    checkpoint_path = os.path.join(model_path, seed_folder, "models")
    rich.print(f"Loading model from {checkpoint_path}")

    # 2. 修改参数为eval可用的
    def _modify_algo_and_env_dict():
        algo_dict["train"]["model_dir"] = checkpoint_path  # 模型位置

        # eval thread
        algo_dict["eval"]["n_eval_rollout_threads"] = (
            config.eval_settings.general.eval_threads
        )
        algo_dict["eval"]["eval_episodes"] = config.eval_settings.general.eval_episodes

        # render
        algo_dict["render"]["use_render"] = config.eval_settings.functions.render
        algo_dict["render"]["render_episodes"] = (
            config.eval_settings.functions.render_episodes
        )

        # logger
        algo_dict["logger"]["log_dir"] = f"./results/logs/{save_group}"

        # FIXME: 为什么需要这个？
        if (
            env_name == "pettingzoo_mw" or env_name == "pettingzoo_mw_llm"
        ) and algo_dict["train"].get("num_env_steps") is not None:
            algo_dict["train"]["num_env_steps"] = 1  # FIXME: ???

        # disturbances的引入
        if env_name == "pettingzoo_mw" or env_name == "pettingzoo_mw_llm":
            env_dict["custom"]["is_eval"] = True
            env_dict["custom"]["eval_disturb"] = _to_dict(config.eval_scenario).get(
                "disturbances", []
            )

    _modify_algo_and_env_dict()

    rich.pretty.pprint(algo_dict, expand_all=True)
    rich.pretty.pprint(env_dict, expand_all=True)

    # 3. 初始化runner
    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

    @atexit.register
    def _cleanup():
        runner.close()
        # wandb.finish()

    start_time = time.time()
    # 4. render？还是eval？
    if config.eval_settings.functions.render:

        def _render():
            render_mode = "rgb_array"
            (
                rgb_array,
                rewards_arr,
                episode_obses_arr,
                lidar_obs_arr,
            ) = runner.render(render_mode)
            config_name = f"[{algorithm_name}]<{env_name}>_{scenario_name}{name_suffix}"
            # 保存episode_obses_arr到JSON文件
            if (
                episode_obses_arr is not None
                and config.eval_settings.functions.export_angle_data
            ):
                json_dir = os.path.join(
                    hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
                    "./data/",
                )
                os.makedirs(json_dir, exist_ok=True)

                json_path = os.path.join(json_dir, f"{config_name}_episode_obses.json")

                # # 将numpy数组转换为列表以便JSON序列化
                episode_obses_serializable = []
                for episode in episode_obses_arr:
                    episode_serializable = []
                    for agent_obses in episode:
                        episode_serializable.append(
                            [obs.tolist() for obs in agent_obses]
                        )
                    episode_obses_serializable.append(episode_serializable)

                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(episode_obses_serializable, f, ensure_ascii=False)

                # 和上面一样，也存一份lidar_obs_arr
                json_path = os.path.join(json_dir, f"{config_name}_lidar_obs.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(lidar_obs_arr, f, ensure_ascii=False)

                rich.print(f"Episode observations saved to: {json_path}")
            if rgb_array is not None:
                export_gif(
                    config_name=config_name,
                    frames_arr=rgb_array,
                    rewards_arr=rewards_arr,
                )

        _render()
        if hasattr(runner, "eval_envs") and runner.eval_envs is not None:
            runner.eval_envs.close()
        runner.close()
        end_time = time.time()
        print(f"Render time: {end_time - start_time} seconds")
    else:
        # 根据是否是off-policy，选择不同的eval方式
        has_logger = hasattr(runner, "logger")
        if has_logger:
            runner = cast(OnPolicyMARunner, runner)
            logger: PettingZooMWLogger = runner.logger
            logger.is_testing = (
                True  # 标识目前在eval；但是eval这个词被它用了，只能用test了。
            )
            runner.eval()
            assert runner.eval_envs is not None
            runner.eval_envs.reset()
            terminate_arr = logger.test_data["terminate_at"]
            angle_arr = logger.test_data["angle_data"]
        else:
            runner = cast(OffPolicyBaseRunner, runner)
            logger = None
            runner.eval(1)
            terminate_arr = runner.eval_episode_lens
            angle_arr = runner.eval_episode_angles

        # 开始计算
        # 2.1 计算提前摔倒的次数
        terminate_cnt = 0
        package_x = []
        for i in range(len(terminate_arr)):
            if (
                terminate_arr[i] + 2 < config.environment.env_tweak.max_cycles
            ):  # +2 去除一点边际问题
                terminate_cnt += 1
            package_x.append(
                logger.test_data["package_x"][i] if has_logger else runner.episode_xs[i]
            )
        # 关闭eval_envs和runner
        if hasattr(runner, "eval_envs") and runner.eval_envs is not None:
            runner.eval_envs.close()
        runner.close()

        angle_flatten = [
            angle for episode_angles in angle_arr for angle in episode_angles
        ]
        import numpy as np

        end_time = time.time()

        assert config.environment.env_tweak.max_cycles is not None

        return_result = {
            "desc": f"[{algorithm_name}]<{scenario_name}>_{config.eval_scenario.name}_{_to_dict(config.eval_scenario).get('desc', 'original')}",
            "algo": algorithm_name,
            "variant": scenario_name,
            "scenario": config.eval_scenario.name,
            "terminate_cnt": terminate_cnt,
            "terminate_cnt1": len(terminate_arr),
            "total_episodes": config.eval_settings.general.eval_episodes,
            "angle_data": angle_flatten,
            "angle_data_grouped": angle_arr,
            "angle_data_avg": sum(angle_flatten) / len(angle_flatten),
            "angle_data_std": np.std(angle_flatten),
            "angle_larger_than_5": sum([1 for angle in angle_flatten if angle > 5])
            / len(angle_flatten),
            "angle_larger_than_10": sum([1 for angle in angle_flatten if angle > 10])
            / len(angle_flatten),
            "angle_larger_than_15": sum([1 for angle in angle_flatten if angle > 15])
            / len(angle_flatten),
            "package_x": sum(package_x) / len(package_x),
            "total_time": end_time - start_time,
            "total_timesteps": sum(terminate_arr)
            + (config.eval_settings.general.eval_episodes - terminate_cnt)
            * config.environment.env_tweak.max_cycles,
            "total_timesteps1": sum(terminate_arr)
            + (config.eval_settings.general.eval_episodes - len(terminate_arr))
            * config.environment.env_tweak.max_cycles,
            "avg_terminate_at": sum(terminate_arr) / len(terminate_arr),
        }
        print(f"Evaluation time: {end_time - start_time} seconds")
        return return_result

    end_time = time.time()
    print(f"Evaluation time: {end_time - start_time} seconds")


@hydra.main(
    config_path="../1.config/task/eval", config_name="default", version_base=None
)
def main(cfg: EvalConfig):
    rich.pretty.pprint(_to_dict(cfg), expand_all=True)

    # 2. 整理参数，转换为dict以传导给harl
    (
        algo_dict,
        env_dict,
        basic_info,
        algorithm_name,
        env_name,
        scenario_name,
        run_group,
        save_group,
    ) = _to_harl_dict(cfg)

    # 3. 初始化wandb
    wandb.init(
        project=cfg.wandb.wandb_project,
        config={"original": _to_dict(cfg), "algo": algo_dict, "env": env_dict},
        sync_tensorboard=True,
        # name=run_name + f"_{ts}",
        group=run_group,
        job_type="eval",
        tags=[
            env_name,
            algorithm_name,
            scenario_name,
        ],
    )
    # 5. 启动训练
    result = eval(cfg)
    # 保存结果到JSON文件
    """
    将评估结果保存为JSON格式文件
    根据当前时间创建目录结构并保存结果
    """
    if result is not None:
        import json
        from datetime import datetime

        # 获取当前时间
        now = datetime.now()
        mmdd = now.strftime("%m%d")
        hhmm = now.strftime("%H%M%S")

        # 创建保存路径
        save_dir = f"./results/runs/{mmdd}/{hhmm}"
        os.makedirs(save_dir, exist_ok=True)

        # 保存为JSON文件
        json_path = os.path.join(save_dir, "result.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Results saved to: {json_path}")

        wandb.log(result)


if __name__ == "__main__":
    main()
