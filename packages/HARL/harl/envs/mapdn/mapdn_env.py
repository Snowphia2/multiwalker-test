import copy
import logging

from dataclasses import dataclass
from typing import Optional, Union
from gymnasium import spaces
from pettingzoo.utils import wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

import sumo_rl
import numpy as np
from mapdn.environments.var_voltage_control.voltage_control_env import VoltageControl


logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


class SumoEnvironmentPZWithGlobalState(sumo_rl.SumoEnvironmentPZ):
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

    observation_class: Union[str, type] = "DefaultObservationFunction"

    add_system_info: bool = True
    add_per_agent_info: bool = True
    sumo_seed: Union[str, int] = "random"
    fixed_ts: bool = False
    sumo_warnings: bool = True
    additional_sumo_cmd: Optional[str] = None
    render_mode: Optional[str] = None


class MAPDNEnv:
    def __init__(self, args):
        self.args = copy.deepcopy(args)
        self.env = VoltageControl(self.args, [])  # FIXME!

        self.env.reset()

        # sumo是离散动作
        self.discrete: bool = False

        # self.max_cycles: int = (args.num_seconds) // args.delta_time - 1

        self.cur_step: int = 0

        self.n_agents = self.env.get_num_of_agents()
        self.agents = [i for i in range(self.n_agents)]

        # 如果是dict, unwrap
        single_observation_space = spaces.Box(  # FIXME!
            low=np.zeros(self.env.obs_size, dtype=np.float32),
            high=np.ones(self.env.obs_size, dtype=np.float32),
        )
        single_action_space = spaces.Box(
            low=np.array(
                [-self.args.action_scale + self.args.action_bias], dtype=np.float32
            ),
            high=np.array(
                [self.args.action_scale + self.args.action_bias], dtype=np.float32
            ),
        )
        self.observation_space = self.unwrap(single_observation_space)  # FIXME!
        self.action_space = self.unwrap(single_action_space)
        self.share_observation_space = self.repeat(self.env.state_space)  # FIXME!
        self._seed = 0
        self.cur_step = 0

    @property
    def global_state(self):
        return self.env.get_state()

    def step(self, actions):  # FIXME!
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

    def reset(self):  # FIXME
        """重置环境并返回初始观测和状态"""
        self._seed += 1
        self.cur_step = 0
        obs, infos = self.env.reset(seed=self._seed)  # type: ignore
        obs = self.unwrap(obs)
        s_obs = self.repeat(self.env.state())
        avail_actions = self.get_avail_actions()

        obs_n = np.array(obs)
        s_obs_n = np.array(s_obs)
        avail_actions_n = np.array(avail_actions)
        return obs_n, s_obs_n, avail_actions_n

    def get_avail_actions(self):
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

    def wrap(self, target):
        return {agent: target[i] for i, agent in enumerate(self.agents)}

    def unwrap(self, target):
        return [target[agent] for agent in self.agents]

    def repeat(self, a):
        return [a for _ in range(self.n_agents)]
