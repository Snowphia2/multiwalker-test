import rich.pretty
import wandb
import hydra
import omegaconf
import rich
from harl.runners import RUNNER_REGISTRY
from datetime import datetime
from .types.task.train_type import TrainConfig
import atexit


def _to_dict(cfg1) -> dict:
    dict_result = omegaconf.OmegaConf.to_container(
        cfg1, resolve=True, throw_on_missing=True
    )
    if type(dict_result) is not dict:
        raise ValueError("dict_result is not a dict")
    return dict_result


def _to_harl_dict(
    cfg: TrainConfig,
):
    # 1. 从配置里读取参数
    algorithm_name = cfg.algorithm.name
    env_name = cfg.environment.name
    scenario_name = cfg.scenario.name

    algo_args = cfg.algorithm_parameters
    env_args = cfg.environment_parameters

    # 1.1 生成run_name
    run_name = f"[{algorithm_name}]<{scenario_name}>"
    env_tweaks = cfg.environment.env_tweak
    for key in env_tweaks.tweak_types:
        if not key.startswith("_"):
            run_name += f"<{key}={_to_dict(env_tweaks)[key]}>"

    # 1.2 生成wandb_group 和 save_group
    now_time = datetime.now().strftime("%m%d/%H%M")

    run_group = cfg.wandb.wandb_group
    if run_group == "latest":
        run_group = now_time

    save_group = cfg.model.save_group
    if save_group == "latest":
        save_group = now_time

    algo_args.logger.log_dir = f"./results/models/{save_group}"

    # 1.3 转换为dict
    algo_dict = _to_dict(algo_args)
    env_dict = _to_dict(env_args)

    # ----每个env可以在这里做特殊操作---
    # 至少要修改episode_length

    if env_name == "pettingzoo_mw":
        from .types.environment.type_multiwalker import multiwalker_customize_dict

        algo_dict, env_dict = multiwalker_customize_dict(cfg, algo_dict, env_dict)
    elif env_name == "sumo":
        from .types.environment.type_sumo import sumo_customize_dict

        algo_dict, env_dict = sumo_customize_dict(cfg, algo_dict, env_dict, save_group)

    # 1.4 执行env_tweak
    # 1.5 执行scenario
    env_tweak = _to_dict(cfg.environment.env_tweak)
    for key in env_tweak.keys():
        if not key.startswith("_") and key != "tweak_types":
            env_dict[key] = env_tweak[key]
    if cfg.environment_scenario is not None:
        env_dict.update(_to_dict(cfg.environment_scenario))

    # 1.7 生成basic_info
    basic_info = {
        "env": env_name,
        "algo": algorithm_name,
        "exp_name": run_name,
    }
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


@hydra.main(config_path="../1.config/task/train", config_name="sumo", version_base=None)
def main(cfg: TrainConfig):
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

    # 3. 初始化runner
    print("ENV_DICT!!")
    rich.print(env_dict)
    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

    @atexit.register
    def _cleanup():
        runner.close()
        wandb.finish()

    # 4. 初始化wandb
    wandb.tensorboard.patch(root_logdir=runner.log_dir)  # type: ignore
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
        "logs/eval_average_steps/eval_average_steps/eval_average_steps",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_steps/eval_average_steps/eval_average_steps",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_episode_rewards",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_episode_rewards",
        summary="last",
    )
    wandb.define_metric(
        "eval_average_steps",
        summary="last",
    )
    wandb.define_metric(
        "eval_average_steps",
        summary="max",
    )
    wandb.define_metric(
        "eval_terminate_x",
        summary="last",
    )
    wandb.define_metric(
        "eval_terminate_x",
        summary="max",
    )
    wandb.define_metric(
        "eval_average_v_deviation",
        summary="last",
    )
    wandb.define_metric(
        "eval_average_v_deviation",
        summary="min",
    )

    # 5. 启动训练
    runner.run()

    runner.close()
    wandb.finish()


if __name__ == "__main__":
    main()
