import copy
import logging

from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Union
from gymnasium import spaces
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from sumo_rl.environment.env import SumoEnvironment
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
import sumo_rl
import numpy as np

import gymnasium as gym

# from pettingzoo.sisl import multiwalker_v9
from ..harl_env_with_events import (
    HarlEnvWithEvents,
    Event,
    EnvProtocol,
)

# from pettingzoo.sisl import multiwalker_v9

# 设置日志级别为WARNING，避免输出debug信息
logging.basicConfig(level=logging.WARNING)


class SumoEnvironmentPZWithGlobalState(sumo_rl.SumoEnvironmentPZ, EnvProtocol):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state_space = self.get_state_space()

    def get_state_space(self):
        _a = self.env.ts_ids[0]
        ts = self.env.traffic_signals[_a]
        low = ts.num_green_phases + 1 + 2 * len(ts.lanes)
        return spaces.Box(
            low=np.zeros(low * self.num_agents, dtype=np.float32),
            high=np.ones(low * self.num_agents, dtype=np.float32),
        )

    def state(self):
        obs = []
        for agent in self.agents:
            obs.append(self.observe(agent))
        global_state = np.array(obs).flatten().astype(np.float32)
        return global_state

    def step(self, action):
        super().step(action)


def env(**kwargs):
    """Instantiate a PettingoZoo environment."""
    env = SumoEnvironmentPZWithGlobalState(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


@dataclass
class SumoEnvConfig:
    """SUMO环境配置类

    包含SUMO交通仿真环境的所有配置参数
    """

    net_file: str
    route_file: str
    out_csv_name: Optional[str] = None
    use_gui: bool = False
    virtual_display: list[int] = field(default_factory=lambda: [3200, 1800])
    begin_time: int = 0
    num_seconds: int = 20000
    max_depart_delay: int = -1
    waiting_time_memory: int = 1000
    time_to_teleport: int = -1
    delta_time: int = 5
    yellow_time: int = 2
    min_green: int = 5
    max_green: int = 50
    enforce_max_green: bool = False
    single_agent: bool = False
    reward_fn: str = "diff-waiting-time"
    reward_weights: Optional[list[float]] = None

    observation_class: Union[str, type] = "DefaultObservationFunction"

    add_system_info: bool = True
    add_per_agent_info: bool = True
    sumo_seed: Union[str, int] = "random"
    fixed_ts: bool = False
    sumo_warnings: bool = True
    additional_sumo_cmd: Optional[str] = None
    render_mode: Optional[str] = None

    events: Optional[list[Event]] = None


ActionType = np.ndarray[Any, np.dtype[np.int32]]
ObsType = np.ndarray[Any, np.dtype[Union[np.float32, np.int32]]]
StateType = np.ndarray[Any, np.dtype[Union[np.float32, np.int32]]]


TAgentId = str
TEnv = aec_to_parallel_wrapper[str, ObsType, ActionType]
TArgs = dict[str, Any]
TDeepDict = dict[TAgentId, dict[str, Any]]
ObsWrappedType = dict[TAgentId, ObsType]


class PettingZooSumoEnv(
    HarlEnvWithEvents[
        str,
        aec_to_parallel_wrapper,
        SumoEnvConfig,
        ObsType,
        ActionType,
        StateType,
        SumoEnvironment,
    ]
):
    events: list[Event]
    n_agents: int
    share_observation_space: list[gym.spaces.Box]
    observation_space: list[gym.spaces.Box]
    action_space: list[Union[gym.spaces.Box, gym.spaces.Discrete]]
    agents: list[TAgentId]
    _seed: int

    def __init__(self, args: SumoEnvConfig):
        self.args: SumoEnvConfig = copy.deepcopy(args)

        args.observation_class = sumo_rl.DefaultObservationFunction

        # sumo是离散动作
        self.discrete: bool = True

        self.max_cycles: int = (args.num_seconds) // args.delta_time - 1

        self.cur_step: int = 0

        dict_args = asdict(args)
        dict_args["virtual_display"] = tuple(dict_args["virtual_display"])
        del dict_args["observation_class"]
        del dict_args["events"]
        self.env = parallel_env(**dict_args, observation_class=args.observation_class)
        self.env.reset()

        self.n_agents = self.env.num_agents
        self.agents = self.env.agents

        # 如果是dict, unwrap
        self.observation_space = self.unwrap(self.env.observation_spaces)  # type: ignore
        self.action_space = self.unwrap(self.env.action_spaces)  # type: ignore
        self.share_observation_space = self.repeat(self.env.state_space)
        self._seed = 0
        self.cur_step = 0

        # events
        if args.events is not None:
            self._init_event_mapping()
            self._init_event(args.events)

        super().__init__(args)

    def _init_event_mapping(self) -> None:
        from .events.lane_closed import LaneCloseEventManager

        self.event_mapping = {
            "lane_closed": LaneCloseEventManager,
        }

    @property
    def global_state(self) -> StateType:
        return self.env.state()

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        actions_wrapped = self.wrap(actions.flatten().tolist())

        _override_signal = {
            # "A2": [-1, -1, 25, 15],
            "B2": [-1, -1, 45, -1],
        }

        if self.cur_step == 20:
            self.traffic_info = {
                "A2": [0, 0, 0, 0],
                "B2": [0, 0, 0, 0],
                "now_A2": 0,
                "now_B2": 0,
            }
        # if self.cur_step > 20 and self.cur_step < 5000:
        #     for agent in self.agents:
        #         if agent != "B2":
        #             continue
        #         now_green_phase = self.traffic_info[f"now_{agent}"]
        #         proposed_next_action = actions_wrapped[agent]
        #         self.traffic_info[agent][now_green_phase] += self.args.delta_time
        #         if (
        #             self.traffic_info[agent][now_green_phase]
        #             <= _override_signal[agent][now_green_phase]
        #         ):
        #             # 如果还没达到最低要求，就不许换action
        #             actions_wrapped[agent] = now_green_phase
        #             print(
        #                 f"ts={self.cur_step}, agent={agent}, current_green_phase={now_green_phase}, policy wants {proposed_next_action}, this_has_been: {self.traffic_info[agent][now_green_phase]}, [not allowed] to change since min is {_override_signal[agent][now_green_phase]}"
        #             )
        #         else:
        #             # actions_wrapped[agent] = current_green_phase + 1
        #             print(
        #                 f"ts={self.cur_step}, agent={agent}, current_green_phase={now_green_phase}, policy wants {proposed_next_action}, this_has_been: {self.traffic_info[agent][now_green_phase]}, [allowed] to change! change to {proposed_next_action}"
        #             )
        #             self.traffic_info[agent][now_green_phase] = 0
        #             self.traffic_info[f"now_{agent}"] = proposed_next_action

        # 可以在这里延长绿灯时间；设个参数，atleast>33s; 小于33的时候不许关绿色信号。
        obs, rew, term, trunc, info = self.env.step(actions_wrapped)  # type: ignore
        # 这里的析构是aec_to_parallel_wrapper负责的

        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True

        for agent in self.agents:
            info[agent]["curr_step"] = self.cur_step

        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        global_state = self.repeat(self.global_state)
        total_reward: float = sum([rew[agent] for agent in self.agents])
        rewards: list[list[float]] = [[total_reward]] * self.n_agents
        # self.trigger_event()
        return (
            self.unwrap(obs),
            global_state,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """重置环境并返回初始观测和状态"""
        self._seed += 1
        self.cur_step = 0
        obs, infos = self.env.reset(seed=self._seed)  # type: ignore
        obs = self.unwrap(obs)
        s_obs = self.repeat(self.global_state)
        return obs, s_obs, self.get_avail_actions()

    def render(self) -> None:
        self.env.render()

    def close(self) -> None:
        self.env.close()

    def seed(self, seed: int) -> None:
        self._seed = seed
