from dataclasses import dataclass
from typing import Any, List, Optional
from ..algorithm.mappo_type import MappoConfig
from ..environment.type_multiwalker import MultiWalkerTweakConfig, MultiWalkerConfig


@dataclass
class WandbConfig:
    """wandb实验跟踪配置"""

    wandb_name: str
    wandb_group: str = "latest"
    wandb_project: str = "mw_skill"


@dataclass
class ModelConfig:
    """模型保存配置"""

    should_load_model: bool = False
    load_group: Optional[str] = None
    save_group: str = "latest"

    def __post_init__(self):
        """验证配置的有效性"""
        if self.should_load_model and self.load_group is None:
            raise ValueError("当should_load_model=True时，load_group不能为None")


@dataclass
class AlgorithmConfig:
    name: str


@dataclass
class EnvironmentConfig:
    name: str
    scenario: str
    env_tweak: MultiWalkerTweakConfig


@dataclass
class ScenarioConfig:
    name: str


@dataclass
class DisturbanceConfig:
    name: str
    start_at: int
    end_at: int
    disturbance_args: Any


@dataclass
class EvalScenarioConfig:
    name: str
    desc: str
    is_raw: Optional[bool] = False
    disturbances: Optional[List[DisturbanceConfig]] = None


@dataclass
class TrainConfig:
    """训练配置主类.

    wandb: wandb配置
    model: 模型保存配置
    experiment: 实验配置
    algorithm_parameters: 算法参数
    environment_parameters: 环境参数
    """

    wandb: WandbConfig
    model: ModelConfig

    algorithm: AlgorithmConfig
    algorithm_parameters: MappoConfig

    environment: EnvironmentConfig
    environment_parameters: MultiWalkerConfig

    scenario: ScenarioConfig
    environment_scenario: Optional[dict]

    eval_scenario: EvalScenarioConfig
