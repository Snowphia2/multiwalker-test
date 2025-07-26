import logging
from typing import Any, final, Union

# from pettingzoo.sisl import multiwalker_v9
from ..pettingzoo_mw.pettingzoo_mw_env import (
    PettingZooMWEnv,
    ActionType,
    ObsType,
    StateType,
    AllAgentActionAvailableType,
)
from ..harl_env_llm import HarlEnvWithLLM
from .llm.manager import PettingZooMWLLMManager

logging.basicConfig()
# logging.getLogger().setLevel(logging.DEBUG)


@final
class PettingZooMWLLMEnv(PettingZooMWEnv, HarlEnvWithLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm_manager = self._init_llm_manager()
        self.llm_frequency = 50

    def _init_llm_manager(self) -> PettingZooMWLLMManager:
        return PettingZooMWLLMManager(self.env, self.raw_env.env, None)

    def step(
        self, actions: ActionType
    ) -> tuple[
        list[ObsType],
        list[StateType],
        list[list[float]],
        list[bool],
        list[dict[str, Any]],
        Union[AllAgentActionAvailableType, None],
    ]:
        obs, state, reward, terminated, info, available_actions = super().step(actions)
        if self.cur_step % self.llm_frequency == 0:
            self.llm_manager.execute_llm(obs, state[0])
        return obs, state, reward, terminated, info, available_actions
