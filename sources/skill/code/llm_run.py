from __future__ import annotations

import os
import time
import asyncio
import hydra
import rich
from rich.panel import Panel
import omegaconf
import wandb

from harl.envs.pettingzoo_mw.pettingzoo_mw_logger import PettingZooMWLogger
import hydra_type_2
from hydra import initialize, compose
from hydra.core.global_hydra import GlobalHydra
from moviepy.editor import VideoFileClip
import imageio
from .llm_runner import SwitchableRunner

os.environ["SDL_VIDEODRIVER"] = "dummy"

# 设置字体路径
# font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"  # 确保路径正确
# font_prop = font_manager.FontProperties(fname=font_path)
# plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
# plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
wandb_results = []
max_cycles = 10000


def export_gif(
    config_name, frames_arr, rewards_arr, globalConfig: hydra_type_2.EntrypointConfig
):
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
        is_negative = rewards < 0  # {'fail' if is_negative else 'success'}_
        gif_path = os.path.join(
            gif_folder,
            f"{i}_[{rewards:.2f}]_{config_name}.gif",
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
                f"{i}_[{rewards:.2f}]_{config_name}.mp4",
            ),
            codec="libx264",
            logger=None,
        )

        # 删除前面的gif
        os.remove(gif_path)


def _to_dict(cfg1) -> dict:
    return omegaconf.OmegaConf.to_container(cfg1, resolve=True, throw_on_missing=True)  # pyright: ignore


async def run_evaluations(config: hydra_type_2.EntrypointConfig, algorithm: str):
    """执行baseline和扰动测试的评估"""
    gif_dir = os.path.join(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir, "./videos"
    )
    os.makedirs(gif_dir, exist_ok=True)

    for scenario_name, scenario in config.disturbances.items():
        scenario: hydra_type_2.ScenarioConfig = scenario
        await eval(
            config,
            algorithm=algorithm,
            checkpoint_type="raw",
            eval_scenario=scenario,
        )


async def eval(
    entrypointConfig: hydra_type_2.EntrypointConfig,
    algorithm: str,
    checkpoint_type: str,
    eval_scenario: hydra_type_2.ScenarioConfig,
):
    start_time = time.time()
    assert entrypointConfig.policy_sets is not None, "policy_sets is required"

    # 先取出所有的policy
    policies = {
        policy.name: policy.dir for policy in entrypointConfig.policy_sets.choices
    }
    default_policy_name = entrypointConfig.policy_sets.default
    default_policy_dir = policies[default_policy_name]
    policy_reversed_motors = {
        policy.name: policy.reverse_motor
        for policy in entrypointConfig.policy_sets.choices
    }
    base_checkpoint_path = (
        f"/pettingzoo_mw/multiwalker/{algorithm}/[{algorithm}]<{checkpoint_type}>"
    )
    for key in ["n_walkers", *entrypointConfig.env_tweak.tweak_types]:
        if not key.startswith("_"):
            base_checkpoint_path += f"<{key}={entrypointConfig.env_tweak[key]}>"  # pyright: ignore
    # rich.print(os.listdir(default_policy_dir + base_checkpoint_path))

    policy_args = hydra_type_2.PolicyArg(
        default_policy_name=default_policy_name,
        default_policy_dir=default_policy_dir,
        policies=policies,
        policy_reversed_motors=policy_reversed_motors,
        checkpoint_prefix=base_checkpoint_path,
    )

    seed_folder = next(
        folder
        for folder in os.listdir(default_policy_dir + base_checkpoint_path)
        if folder.startswith("seed-")
    )
    rich.print(default_policy_dir)
    rich.print(default_policy_name)
    rich.print(policies)
    checkpoint_path = os.path.join(
        default_policy_dir + base_checkpoint_path, seed_folder, "models"
    )
    # 1. 先读取对应的模型
    rich.print(
        Panel(
            f"Checkpoint Path: {checkpoint_path}\nScenario Name: {eval_scenario.name}",
            title="Evaluation Info",
        )
    )

    # 1.1. 从配置里读取参数，转换为harl使用的格式
    with initialize(version_base=None, config_path="./configs"):
        cfg = compose(
            config_name="train",
            overrides=[
                f"algorithm={algorithm}",
                f"environment={checkpoint_type}",
            ],
        )
        algo_args = cfg.algorithm
        env_args = cfg.environment

        algorithm_name = cfg.algorithm.name
        env_name = cfg.environment.name
        scenario_name = cfg.environment.scenario
        basic_info = {
            "env": env_name,
            "algo": algorithm_name,
            "exp_name": f"testing_<{algorithm_name}>_{scenario_name}",
        }
        # 特殊处理max_cycles
        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("episode_length") is not None
        ):
            algo_args.train.episode_length = entrypointConfig.env_tweak.max_cycles
        env_args.max_cycles = entrypointConfig.env_tweak.max_cycles

        algo_args.train.model_dir = checkpoint_path  # 读取模型！
        rich.print(algo_args.train.model_dir)

        # 配置eval遍数
        algo_args.eval.n_eval_rollout_threads = (
            entrypointConfig.basic_config.eval_threads
        )
        algo_args.eval.eval_episodes = entrypointConfig.basic_config.eval_episodes

        # gpu
        algo_args.device.cuda = entrypointConfig.basic_config.use_gpu

        # 配置render
        if entrypointConfig.basic_config.render:
            algo_args.render.use_render = True
            algo_args.render.render_episodes = (
                entrypointConfig.basic_config.eval_episodes
            )

        # 配置num_env_steps
        if (
            env_name == "pettingzoo_mw"
            and algo_args.train.get("num_env_steps") is not None
        ):
            algo_args.train.num_env_steps = 1  # FIXME: ???

        algo_dict = _to_dict(algo_args)
        algo_dict.pop("name")

        env_dict = _to_dict(env_args)
        del env_dict["name"]
        del env_dict["scenario"]

        # 处理env_tweaks
        for key, value in _to_dict(entrypointConfig.env_tweak).items():
            if value is not None:
                env_dict[key] = value
        env_dict["custom"]["eval_disturb"] = _to_dict(eval_scenario)["disturbances"]
        env_dict["custom"]["is_eval"] = True
        del env_dict["tweak_types"]

        # ============== 处理完毕config，启动runner实例 ==============
        runner = SwitchableRunner(basic_info, algo_dict, env_dict, policy_args)

        if entrypointConfig.basic_config.render:
            render_mode = "rgb_array"
            (
                rgb_array,
                rewards_arr,
                episode_obses_arr,
                lidar_obs_arr,
            ) = await runner.exec(render_mode)
            config_name = f"[{algorithm}]<{checkpoint_type}>_{eval_scenario.name}"
            for key in ["n_walkers", *entrypointConfig.env_tweak.tweak_types]:
                if not key.startswith("_"):
                    config_name += f"<{key}={entrypointConfig.env_tweak[key]}>"
            export_gif(
                config_name=config_name,
                frames_arr=rgb_array,
                rewards_arr=rewards_arr,
                globalConfig=entrypointConfig,
            )

            exit()
        else:
            # 根据是否是off-policy，选择不同的eval方式
            has_logger = hasattr(runner, "logger")
            if has_logger:
                logger: PettingZooMWLogger = runner.logger
                logger.is_testing = (
                    True  # 标识目前在eval；但是eval这个词被它用了，只能用test了。
                )
                runner.eval()
                terminate_arr = logger.test_data["terminate_at"]
                angle_arr = logger.test_data["angle_data"]
            else:
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
                    terminate_arr[i] + 2 < entrypointConfig.env_tweak.max_cycles
                ):  # +2 去除一点边际问题
                    terminate_cnt += 1
                package_x.append(
                    logger.test_data["package_x"][i]
                    if has_logger
                    else runner.episode_xs[i]
                )
            # 关闭eval_envs和runner
            if hasattr(runner, "eval_envs") and runner.eval_envs is not None:
                runner.eval_envs.close()
            runner.close()

            end_time = time.time()
            print(
                f"处理[{algorithm}]<{checkpoint_type}>_{eval_scenario.name} 耗时: {end_time - start_time:.2f}秒"
            )
            return_result = {
                "desc": f"[{algorithm}]<{checkpoint_type}>_{eval_scenario.name}_{_to_dict(eval_scenario).get('desc', 'original')}",
                "algo": algorithm,
                "variant": checkpoint_type,
                "scenario": eval_scenario.name,
                "terminate_cnt": terminate_cnt,
                "angle_data": [
                    angle for episode_angles in angle_arr for angle in episode_angles
                ],
                "angle_data_grouped": angle_arr,
                "package_x": sum(package_x) / len(package_x),
            }
            return return_result


@hydra.main(
    config_path="./configs/evaluation",
    config_name="rend",
    version_base=None,
)
def main(cfg: hydra_type_2.EntrypointConfig):
    # 用于json存储的目录
    timestamp = time.strftime("%m%d-%H:%M")
    GlobalHydra.instance().clear()

    # 初始化wandb
    run = wandb.init(
        project=cfg.basic_config.wandb_project,
        name=cfg.algorithm + "_" + timestamp,
        config=_to_dict(cfg),
        save_code=True,
        group=cfg.basic_config.run_group,
        job_type="eval" if not cfg.basic_config.render else "render",
    )

    asyncio.run(run_evaluations(cfg, cfg.algorithm))

    run.finish()


if __name__ == "__main__":
    main()
