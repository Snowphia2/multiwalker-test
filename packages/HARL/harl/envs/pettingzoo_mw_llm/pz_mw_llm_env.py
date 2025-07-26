import logging
from typing import final

# from pettingzoo.sisl import multiwalker_v9
from ..pettingzoo_mw.pettingzoo_mw_env import PettingZooMWEnv
from .llm.manager import PettingZooMWLLMManager

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)


@final
class PettingZooMWLLMEnv(PettingZooMWEnv):
    def __init__(self, args):
        super().__init__(args)
        self.llm_frequency = 50

    def _init_llm_manager(self) -> PettingZooMWLLMManager:
        return PettingZooMWLLMManager(self.env, self.raw_env.env, None)
