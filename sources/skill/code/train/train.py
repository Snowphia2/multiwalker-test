import rich.pretty
import wandb
import hydra
from omegaconf import DictConfig
import omegaconf
import rich
from harl.runners import RUNNER_REGISTRY
from datetime import datetime


def _to_dict(cfg1: DictConfig) -> dict:
    dict_result = omegaconf.OmegaConf.to_container(
        cfg1, resolve=True, throw_on_missing=True
    )
    if type(dict_result) is not dict:
        raise ValueError("dict_result is not a dict")
    return dict_result


def _to_harl_dict(algo_args, env_args, cfg, run_name):
    algorithm_name = algo_args.name
    env_name = env_args.name
    scenario_name = env_args.scenario

    algo_dict = _to_dict(algo_args)
    del algo_dict["name"]

    env_dict = _to_dict(env_args)
    del env_dict["name"]
    del env_dict["scenario"]
    for key in cfg.env_tweak:
        if not key.startswith("_"):
            env_dict[key] = cfg.env_tweak[key]

    if (
        env_name == "pettingzoo_mw"
        and algo_args.train.get("episode_length") is not None
    ):
        algo_dict["train"]["episode_length"] = env_dict["max_cycles"]

    basic_info = {
        "env": env_name,
        "algo": algorithm_name,
        "exp_name": run_name,
    }
    return algo_dict, env_dict, basic_info


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg: DictConfig):
    rich.pretty.pprint(cfg, expand_all=True)

    # 1. 从配置里读取参数
    algo_args = cfg.algorithm
    env_args = cfg.environment

    algorithm_name = cfg.algorithm.name
    env_name = cfg.environment.name
    scenario_name = cfg.environment.scenario

    # 1.1 生成run_name
    run_name = f"[{algorithm_name}]<{scenario_name}>"
    for key in cfg.env_tweak:
        if not key.startswith("_"):
            run_name += f"<{key}={cfg.env_tweak[key]}>"

    # 1.2 生成wandb_group
    run_group = cfg.wandb.wandb_group

    # 1.3 生成save_group
    save_group = cfg.model.save_group
    now_time = datetime.now().strftime("%m%d/%H%M")
    if run_group == "latest":
        run_group = now_time
    if save_group == "latest":
        save_group = now_time

    # 1.4 把save_group写入到algo_args.logger.log_dir
    algo_args.logger.log_dir = f"./results/models/{save_group}"

    # 2. 整理参数，转换为dict以传导给harl
    algo_dict, env_dict, basic_info = _to_harl_dict(algo_args, env_args, cfg, run_name)

    # 3. 初始化runner
    runner = RUNNER_REGISTRY[algorithm_name](basic_info, algo_dict, env_dict)

    # 4. 初始化wandb
    wandb.init(
        project=cfg.wandb_project,
        config={"original": _to_dict(cfg), "algo": algo_dict, "env": env_dict},
        sync_tensorboard=True,
        # name=run_name + f"_{ts}",
        group=run_group,
        job_type="train",
        tags=[
            env_name,
            algorithm_name,
            scenario_name,
            f"wker-{cfg.environment.n_walkers}",
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

    # 6. 训练完成
    runner.close()

    wandb.finish()


if __name__ == "__main__":
    main()
