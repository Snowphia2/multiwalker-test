from dataclasses import dataclass
from typing import Generic, Literal, Protocol, TypeVar, Union, Any

import gymnasium as gym
import numpy as np


@dataclass
class EventData_GivenValue:
    type: Literal["given"]
    given_value: float


@dataclass
class EventData_RandomValue:
    type: Literal["random"]
    random_upper: float
    random_lower: float
    random_method: Literal["uniform"] = "uniform"


@dataclass
class GivenTimestepsTriggerArgs:
    trigger_at_timestep: int
    stop_at_timestep: int
    event_value: Union[EventData_GivenValue, EventData_RandomValue]


@dataclass
class RandomTriggerArgs:
    trigger_frequency: float
    event_value: Union[EventData_GivenValue, EventData_RandomValue]


@dataclass
class Event:
    event_id: str
    should_trigger_by_given_timestep: bool
    given_timestep_trigger_args: GivenTimestepsTriggerArgs
    should_trigger_by_random: bool
    random_trigger_args: RandomTriggerArgs


class EnvProtocol(Protocol):
    def reset(self, *args, **kwargs) -> Any: ...
    def step(self, *args, **kwargs) -> Any: ...
    def close(self, *args, **kwargs) -> Any: ...


TEnv = TypeVar("TEnv", bound=EnvProtocol)

TAgentId = TypeVar("TAgentId")
TArgs = TypeVar("TArgs")
T = TypeVar("T")
ObsType = TypeVar("ObsType", covariant=True)
ActionType = TypeVar("ActionType")
StateType = TypeVar("StateType")

ActionAvailableType = list[bool]
AllAgentActionAvailableType = list[ActionAvailableType]


class HarlEnvWithEvents(
    Generic[TAgentId, TEnv, TArgs, ObsType, ActionType, StateType], Protocol
):
    events: list[Event]
    max_cycles: int
    discrete: bool
    n_agents: int
    share_observation_space: list[gym.Space[np.float32]]
    observation_space: list[gym.Space[np.float32]]
    action_space: list[gym.Space[np.float32]]
    cur_step: int
    agents: list[TAgentId]
    env: TEnv
    args: TArgs

    def __init__(self, args):
        # 检查子类是否设置了 self.n_agents
        some_must_set = [
            "n_agents",
            "agents",
            "share_observation_space",
            "observation_space",
            "action_space",
            "max_cycles",
            "discrete",
            "env",
            "cur_step",
            "args",
        ]
        for attr in some_must_set:
            if not hasattr(self, attr):
                raise NotImplementedError(f"子类必须在 __init__ 里设置 {attr}")
        # 其他初始化逻辑
        _ = self.reset()
        self.cur_step = 0

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
        raise NotImplementedError("[Step] Should be implemented by subclass")

    def reset(
        self,
    ) -> tuple[
        list[ObsType], list[StateType], Union[AllAgentActionAvailableType, None]
    ]:
        raise NotImplementedError("[Reset] Should be implemented by subclass")

    def close(self) -> None:
        self.env.close()
        raise NotImplementedError("[Close] Should be implemented by subclass")

    def render(self) -> Union[np.ndarray[Any, np.dtype[np.uint8]], None]:
        """
        渲染环境并返回RGB图像（如支持），否则返回None。

        Returns:
            np.ndarray: 形状为(H, W, 3)的uint8类型RGB图像，或None。
        Raises:
            NotImplementedError: 如果子类未实现渲染方法。
        """
        # 默认抛出异常，子类应实现具体渲染逻辑
        raise NotImplementedError("[Render] Should be implemented by subclass")

    def get_avail_actions(self) -> Union[AllAgentActionAvailableType, None]:
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_idx: int) -> ActionAvailableType:
        # Same for everyone: return [1] * self.action_space[agent_id].n
        raise NotImplementedError(
            "[GetAvailAgentActions] Should be implemented by subclass"
        )

    def get_events(self) -> list[Event]:
        return self.events

    def start_event(self, event: Event):
        raise NotImplementedError("[StartEvent] Should be implemented by subclass")

    def stop_event(self, event: Event):
        raise NotImplementedError("[StopEvent] Should be implemented by subclass")

    def wrap(self, lam: list[T]) -> dict[TAgentId, T]:
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = lam[i]
        return d

    def unwrap(self, d: dict[TAgentId, T]) -> list[T]:
        _tmp = []
        for agent in self.agents:
            _tmp.append(d[agent])
        return _tmp

    def repeat(self, a: T) -> list[T]:
        return [a for _ in range(self.n_agents)]

    def seed(self, seed):
        self._seed = seed
        self.env.reset(seed=self._seed)
