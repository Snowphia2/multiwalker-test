import copy
import logging

from dataclasses import dataclass
from typing import Optional, Union, final, cast
from gymnasium import spaces
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

import sumo_rl
import numpy as np

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class SumoEnvironmentPZWithGlobalState(sumo_rl.SumoEnvironmentPZ):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # def observation_space(self, agent):
    #     ts = self.env.traffic_signals[self.env.ts_ids[0]]
    #     return spaces.Box(
    #         low=np.zeros(ts.num_green_phases + 1 + 2 * len(ts.lanes), dtype=np.float32),
    #         high=np.ones(ts.num_green_phases + 1 + 2 * len(ts.lanes), dtype=np.float32),
    #     )

    @property
    def action_spaces(self):
        return {a: self.env.action_spaces(a) for a in self.agents}

    @property
    def observation_spaces(self):
        return {a: self.env.observation_spaces(a) for a in self.agents}

    @property
    def state_space(self):
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
    virtual_display: tuple[int, int] = (3200, 1800)
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
    add_system_info: bool = True
    add_per_agent_info: bool = True
    sumo_seed: Union[str, int] = "random"
    fixed_ts: bool = False
    sumo_warnings: bool = True
    additional_sumo_cmd: Optional[str] = None
    render_mode: Optional[str] = None


@final
class PettingZooMPEEnv:
    def __init__(self, args: SumoEnvConfig):
        self.args: SumoEnvConfig = copy.deepcopy(args)

        # sumo是离散动作
        self.discrete: bool = True

        self.max_cycles: int = (
            args.begin_time + args.num_seconds
        ) // args.delta_time + 1

        self.cur_step: int = 0
        self.env: SumoEnvironmentPZWithGlobalState = cast(
            SumoEnvironmentPZWithGlobalState,
            parallel_env(  # this parallel_env here uses the above env function
                net_file="sumo_rl/nets/4x4-Lucas/4x4.net.xml",
                route_file="sumo_rl/nets/4x4-Lucas/4x4c1c2c1c2.rou.xml",
                out_csv_name="outputs/4x4grid/ppo",
                use_gui=False,
                num_seconds=80000,
            ),
        )
        self.env.reset()

        self.n_agents = self.env.num_agents
        self.agents = self.env.agents

        # 如果是dict, unwrap
        self.observation_space = self.unwrap(self.env.observation_spaces)
        self.action_space = self.unwrap(self.env.action_spaces)
        self.share_observation_space = self.repeat(self.env.state_space)
        self._seed = 0
        self.cur_step = 0

    @property
    def global_state(self):
        return self.env.state()

    def step(self, actions):
        """
        return local_obs, global_state, rewards, dones, infos, available_actions
        """
        actions_wrapped = self.wrap(actions.flatten())
        obs, rew, term, trunc, info = self.env.step(actions_wrapped)  # type: ignore
        # 这里的析构是aec_to_parallel_wrapper负责的

        self.cur_step += 1
        if self.cur_step == self.max_cycles:
            trunc = {agent: True for agent in self.agents}
            for agent in self.agents:
                info[agent]["bad_transition"] = True
        dones = {agent: term[agent] or trunc[agent] for agent in self.agents}
        global_state = self.repeat(self.env.state())
        total_reward = sum([rew[agent] for agent in self.agents])
        rewards = [[total_reward]] * self.n_agents
        return (
            self.unwrap(obs),
            global_state,
            rewards,
            self.unwrap(dones),
            self.unwrap(info),
            self.get_avail_actions(),
        )

    def reset(self):
        """Returns initial observations and states"""
        self._seed += 1
        self.cur_step = 0
        obs = self.unwrap(self.env.reset(seed=self._seed))
        s_obs = self.repeat(self.env.state())
        return obs, s_obs, self.get_avail_actions()

    def get_avail_actions(self):
        if self.discrete:
            avail_actions = []
            for agent_id in range(self.n_agents):
                avail_agent = self.get_avail_agent_actions(agent_id)
                avail_actions.append(avail_agent)
            return avail_actions
        else:
            return None

    def get_avail_agent_actions(self, agent_id):
        """Returns the available actions for agent_id"""
        return [1] * self.action_space[agent_id].n

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self, seed):
        self._seed = seed

    def wrap(self, l):
        d = {}
        for i, agent in enumerate(self.agents):
            d[agent] = l[i]
        return d

    def unwrap(self, d):
        l = []
        for agent in self.agents:
            l.append(d[agent])
        return l

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
