from dataclasses import dataclass
from typing import Any, Optional
from ..algorithm.mappo_type import MappoConfig


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
class EnvTweakConfig:
    """环境微调配置"""

    n_walkers: int = 3
    max_cycles: int = 500


@dataclass
class AlgorithmConfig:
    name: str


@dataclass
class EnvironmentConfig:
    name: str
    scenario: str
    env_tweak: EnvTweakConfig


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
    environment_parameters: Any
