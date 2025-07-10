from dataclasses import dataclass
from typing import List, Any, Optional


@dataclass
class DisturbanceConfig:
    name: str
    start_at: int
    end_at: int
    disturbance_args: Any


@dataclass
class ScenarioConfig:
    name: str
    desc: Optional[str]
    disturbances: Optional[List[DisturbanceConfig]]
    is_raw: bool = False


@dataclass
class EnvTweakConfig:
    tweak_types: Any

    terminate_reward: Optional[float] = -100.0
    n_walkers: Optional[int] = 3
    forward_reward: Optional[float] = 1.0
    max_cycles: Optional[int] = 500


@dataclass
class BasicConfig:
    run_group: str
    save_group: str

    wandb_project: str

    seed: int = 42
    eval_episodes: int = 100
    eval_threads: int = 50

    load_results: bool = False
    result_file_name: str = "latest"

    render: bool = False

    ablation: bool = False

    use_gpu: bool = False


@dataclass
class CherryPickConfig:
    export_angle_data: bool = False


@dataclass
class EntrypointConfig:
    algorithm: str
    disturbances: Any
    basic_config: BasicConfig
    env_tweak: EnvTweakConfig
    cherry_pick: CherryPickConfig
