from typing import Any
from sumo_rl.environment.env import SumoEnvironment
from ..pettingzoo_sumo_env import SumoEnvironmentPZWithGlobalState
from ...harl_env_with_events import EventManager
import traci

from pettingzoo.utils.conversions import aec_to_parallel_wrapper


class LaneCloseEventManager(EventManager[aec_to_parallel_wrapper, SumoEnvironment]):
    def _get_real_env(self) -> SumoEnvironment:
        real_env_pz_wrapper: SumoEnvironmentPZWithGlobalState = self.env.aec_env.env.env
        real_env_sumo = real_env_pz_wrapper.env
        self.sumo_env = traci
        self.sumo_env = real_env_sumo.sumo
        assert self.sumo_env is not None
        return real_env_sumo

    def _event_start(self, args: Any) -> None:
        print("[LaneCloseEventManager] starting!!!")
        self._get_real_env()
        assert self.sumo_env is not None
        print(self.sumo_env.lane.getIDList())
        self.sumo_env.lane.setDisallowed("B2C2_1", ["passenger"])

        pass

    def _event_stop(self) -> None:
        self._get_real_env()
        assert self.sumo_env is not None
        self.sumo_env.lane.setDisallowed("B2C2_1", [])
        pass

    def _event_random_value(self) -> Any:
        pass
