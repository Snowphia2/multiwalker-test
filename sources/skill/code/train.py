import rich.pretty
import wandb
import hydra
import omegaconf
import rich
from harl.runners import RUNNER_REGISTRY
from datetime import datetime
from .types.task.train_type import TrainConfig
from .types.algorithm.mappo_type import MappoConfig
from typing import Any


def _to_dict(cfg1) -> dict:
    dict_result = omegaconf.OmegaConf.to_container(
        cfg1, resolve=True, throw_on_missing=True
    )
    if type(dict_result) is not dict:
        raise ValueError("dict_result is not a dict")
    return dict_result


def _to_harl_dict(
    env_name: str,
    algorithm_name: str,
    algo_args: MappoConfig,
    env_args: Any,
    cfg: TrainConfig,
    run_name: str,
    save_group: str,
):
    algo_args.logger.log_dir = f"./results/models/{save_group}"

    algo_dict = _to_dict(algo_args)
    env_dict = _to_dict(env_args)

    env_tweak = _to_dict(cfg.environment.env_tweak)
    for key in env_tweak.keys():
        if not key.startswith("_"):
            env_dict[key] = env_tweak[key]

    if (
        env_name == "pettingzoo_mw"
        and algo_dict["train"].get("episode_length") is not None
    ):
        algo_dict["train"]["episode_length"] = env_dict["max_cycles"]

    basic_info = {
        "env": env_name,
        "algo": algorithm_name,
        "exp_name": run_name,
    }
    return algo_dict, env_dict, basic_info


@hydra.main(config_path="../1.config/task", config_name="0.train", version_base=None)
def main(cfg: TrainConfig):
    rich.pretty.pprint(_to_dict(cfg), expand_all=True)

    # 1. 从配置里读取参数
    algorithm_name = cfg.algorithm.name
    env_name = cfg.environment.name
    scenario_name = cfg.environment.scenario

    algo_args = cfg.algorithm_parameters
    env_args = cfg.environment_parameters

    # 1.1 生成run_name
    run_name = f"[{algorithm_name}]<{scenario_name}>"
    env_tweaks = _to_dict(cfg.environment.env_tweak)
    for key in env_tweaks.keys():
        if not key.startswith("_"):
            run_name += f"<{key}={env_tweaks[key]}>"

    # 1.2 生成wandb_group 和 save_group
    now_time = datetime.now().strftime("%m%d/%H%M")

    run_group = cfg.wandb.wandb_group
    if run_group == "latest":
        run_group = now_time

    save_group = cfg.model.save_group
    if save_group == "latest":
        save_group = now_time

    # 2. 整理参数，转换为dict以传导给harl
    algo_dict, env_dict, basic_info = _to_harl_dict(
        env_name, algorithm_name, algo_args, env_args, cfg, run_name, save_group
    )

    # 3. 初始化runner
    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

    # 4. 初始化wandb
    wandb.init(
        project=cfg.wandb.wandb_project,
        config={"original": _to_dict(cfg), "algo": algo_dict, "env": env_dict},
        sync_tensorboard=True,
        # name=run_name + f"_{ts}",
        group=run_group,
        job_type="train",
        tags=[
            env_name,
            algorithm_name,
            scenario_name,
        ],
    )
    wandb.define_metric(
        "logs/eval_average_episode_rewards/eval_average_episode_rewards/eval_average_episode_rewards",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_episode_rewards",
        summary="max",
    )

    # 5. 启动训练
    runner.run()

    runner.close()
    wandb.finish()


if __name__ == "__main__":
    main()
