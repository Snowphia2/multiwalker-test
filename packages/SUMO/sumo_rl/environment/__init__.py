"""SUMO Environment for Traffic Signal Control."""

from gymnasium.envs.registration import register

from . import observations
from . import traffic_signal
from . import env
from . import resco_envs

register(
    id="sumo-rl-v0",
    entry_point="sumo_rl.environment.env:SumoEnvironment",
    kwargs={"single_agent": True},
)
