from typing import Any

from .sumo_event_manager import SumoEventManager


class LaneCloseEventManager(SumoEventManager):
    def _event_start(self, args: Any) -> None:
        print("[LaneCloseEventManager] starting!!!")
        self.real_env = self._extract_real_env()
        assert self.sumo_env is not None
        # self.sumo_env.lane.setDisallowed("A2B2_1", ["passenger"])

        edgeId = "A2B2_1"
        vehicle_ids = self.sumo_env.vehicle.getIDList()  # 获取所有车辆ID
        for vehId in vehicle_ids:
            self.sumo_env.vehicle.setAdaptedTraveltime(vehId, edgeId, float("inf"))
            self.sumo_env.vehicle.rerouteTraveltime(vehId)

        pass

    def _event_stop(self) -> None:
        self.real_env = self._extract_real_env()
        assert self.sumo_env is not None
        # self.sumo_env.lane.setDisallowed("A2B2_1", [])

        edgeId = "A2B2_1"
        vehicle_ids = self.sumo_env.vehicle.getIDList()  # 获取所有车辆ID
        for vehId in vehicle_ids:
            self.sumo_env.vehicle.setAdaptedTraveltime(vehId, edgeId)
            self.sumo_env.vehicle.rerouteTraveltime(vehId)
        pass

    def _event_random_value(self) -> Any:
        pass
