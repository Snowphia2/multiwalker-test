from typing import Any

from .sumo_event_manager import SumoEventManager


class LaneCloseEventManager(SumoEventManager):
    def _event_start(self, args: Any) -> None:
        print("[LaneCloseEventManager] starting!!!")
        self.real_env = self._extract_real_env()
        assert self.sumo_env is not None
        print(self.sumo_env.lane.getIDList())
        self.sumo_env.lane.setDisallowed("B2C2_1", ["passenger"])

        pass

    def _event_stop(self) -> None:
        self.real_env = self._extract_real_env()
        assert self.sumo_env is not None
        self.sumo_env.lane.setDisallowed("B2C2_1", [])
        pass

    def _event_random_value(self) -> Any:
        pass
