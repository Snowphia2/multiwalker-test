import copy
import logging
import supersuit as ss
from typing import Any, cast, Union
from typing_extensions import override

from .walker.multiwalker.multiwalker_custom import (
    VIEWPORT_W,
    FPS,
    SCALE,
)
import numpy as np
import gymnasium as gym

# from pettingzoo.sisl import multiwalker_v9
from .walker.multiwalker.multiwalker import env_with_raw
from pettingzoo.utils.conversions import aec_to_parallel_wrapper
from ..harl_env_with_events import (
    HarlEnvWithEvents,
    Event,
    AllAgentActionAvailableType,
)

from .walker.multiwalker.multiwalker_base import MultiWalkerEnv as _env
from .walker.multiwalker.multiwalker_stable import MultiWalkerEnv as _env_stable
from .walker.multiwalker.mw_obs_fric import MultiWalkerEnv as _env_fric
from .walker.multiwalker.mw_obs_fric_stable import MultiWalkerEnv as _env_fric_stable
from .walker.multiwalker.mw_obs_motor_stable import MultiWalkerEnv as _env_motor
from .walker.multiwalker.mw_obs_package_mass_stable import (
    MultiWalkerEnv as _env_package_mass,
)
from .walker.multiwalker.mw_talk import MultiWalkerEnv as _env_talk
from .walker.multiwalker.multiwalker_custom import MultiWalkerEnv as _env_custom
from .walker.multiwalker.mw_move import MultiWalkerEnv as _env_move

logging.basicConfig()
logging.getLogger().setLevel(logging.WARNING)


ActionType = np.ndarray[(4,), np.dtype[np.float32]]  # shape=(4,), dtype=np.float32
ObsType = np.ndarray[Any, np.dtype[np.float32]]  # shape=(31,), dtype=np.float32
StateType = np.ndarray[Any, np.dtype[np.float32]]  # shape=(31,), dtype=np.float32
# self.n_walkers * 24 + 3,
all_multiwalkers_united = Union[
    _env,
    _env_stable,
    _env_fric,
    _env_fric_stable,
    _env_motor,
    _env_package_mass,
    _env_talk,
    _env_custom,
    _env_move,
]

TAgentId = str
TEnv = aec_to_parallel_wrapper[str, ObsType, ActionType]
TEnvRaw = all_multiwalkers_united
TArgs = dict[str, Any]
TDeepDict = dict[TAgentId, dict[str, Any]]
ObsWrappedType = dict[TAgentId, ObsType]


class PettingZooMWEnv(
    HarlEnvWithEvents[
        TAgentId, TEnv, TArgs, ObsType, ActionType, StateType, all_multiwalkers_united
    ]
):
    multiwalker_env: all_multiwalkers_united

    events: list[Event]
    max_cycles: int
    discrete: bool
    n_agents: int
    share_observation_space: list[gym.spaces.Box]
    observation_space: list[gym.spaces.Box]
    action_space: list[Union[gym.spaces.Box, gym.spaces.Discrete]]
    cur_step: int
    agents: list[TAgentId]
    env: TEnv
    args: TArgs

    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.discrete = False
        if "max_cycles" in self.args:
            self.max_cycles = self.args["max_cycles"]
            self.args["max_cycles"] += 1
        else:
            self.max_cycles = 500
            self.args["max_cycles"] = 501
        self.cur_step = 0
        # self.module = multiwalker_v9
        self.base_env, self.raw_env = env_with_raw(**self.args)
        self.multiwalker_env = self.raw_env.env
        self.env = cast(
            aec_to_parallel_wrapper[str, ObsType, ActionType],
            ss.pad_action_space_v0(
                ss.pad_observations_v0(aec_to_parallel_wrapper(self.base_env))
            ),
        )
        self._seed: int = 0
        _ = self.env.reset(seed=self._seed)

        self.n_agents = self.env.num_agents
        self.agents = self.env.agents
        self.share_observation_space = self.repeat(self.env.state_space)  # type: ignore
        self.observation_space = self.unwrap(
            {agent: self.env.observation_space(agent) for agent in self.agents}
        )
        self.action_space = self.unwrap(
            {agent: self.env.action_space(agent) for agent in self.agents}
        )
        super().__init__(args)

    @override
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
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        obs, rew, term, trunc, info = self.env.step(self.wrap(list(actions)))
        obs: ObsWrappedType

        self.cur_step += 1

        for agent in self.agents:
            assert self.multiwalker_env.package is not None
            info[agent]["package_angle"] = (
                self.multiwalker_env.package.angle / 3.14 * 180
            )
            info[agent]["curr_step"] = self.cur_step
            info[agent]["package_x"] = self.multiwalker_env.package.position.x
            if isinstance(self.multiwalker_env, _env_move):
                assert self.multiwalker_env.target_v is not None
                assert self.multiwalker_env.walkers[0].hull is not None
                info[agent]["v_deviation"] = abs(
                    self.multiwalker_env.target_v
                    - 0.3
                    * self.multiwalker_env.walkers[0].hull.linearVelocity.x
                    * (VIEWPORT_W / SCALE)
                    / FPS
                )
            # print("--------------------------------")
            # print(f"v_deviation: {info[agent]['v_deviation']}")
            # print(f"v_x: {self.multiwalker_env.walkers[0].hull.linearVelocity.x}")
            # print(
            #     f"v_x_scaled: {0.3 * self.multiwalker_env.walkers[0].hull.linearVelocity.x * (VIEWPORT_W / SCALE) / FPS}"
            # )
            # print(f"target_v: {self.multiwalker_env.target_v}")
            # print("--------------------------------")
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones: dict[TAgentId, bool] = {
            agent: term[agent] or trunc[agent] for agent in self.agents
        }
        total_reward: float = sum([rew[agent] for agent in self.agents])
        rewards: list[list[float]] = [[total_reward]] * self.n_agents

        info = cast(TDeepDict, info)
        s_obs: StateType = cast(StateType, self.env.state())
        assert s_obs is not None, "s_obs is None"

        return (
            self.unwrap(obs),
            self.repeat(s_obs),
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def get_thru_lidar_obs(self) -> Union[list[list[float]], None]:
        if isinstance(self.multiwalker_env, (_env_move, _env_custom)):
            return self.multiwalker_env.get_thru_lidar_obs()
        else:
            return None

    @override
    def reset(
        self,
    ) -> tuple[
        list[ObsType], list[StateType], Union[AllAgentActionAvailableType, None]
    ]:
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        obs, infos = self.env.reset(seed=self._seed)
        obs = self.unwrap(obs)
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.get_avail_actions()

    @override
    def render(self) -> Union[np.ndarray[Any, np.dtype[np.uint8]], None]:
        render_result = self.raw_env.render()
        return render_result

    @override
    def close(self):
        self.env.close()

    @override
    def seed(self, seed: int) -> None:
        self._seed = seed
        _ = self.env.reset(seed=self._seed)

    def _init_event_mapping(self) -> None:
        pass

    def _init_event(self, events: list[Event]) -> None:
        pass
