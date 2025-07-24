from typing import Any
from ...harl_env_with_events import EventManager


class LaneCloseEventManager(EventManager):
    def _event_start(self, args: Any) -> None:
        print("[LaneCloseEventManager] starting!!!")
        pass

    def _event_stop(self) -> None:
        pass

    def _event_random_value(self) -> Any:
        pass
