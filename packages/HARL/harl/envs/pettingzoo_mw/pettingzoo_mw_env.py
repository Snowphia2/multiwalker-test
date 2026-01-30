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
import sys

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
        self.disabled_walker_id = self.args.get('disabled_walker_id', -1)
        self.disturb = self.args.get('disturb')
        
        # 扰动实验配置 - 提取并从args中移除，避免传递给底层环境
        self.disturbance_mode = self.args.get('disturbance_mode', None)
        self.disturb_target_agent = self.args.get('disturb_target_agent', -1)
        self.disturb_magnitude = self.args.get('disturb_magnitude', 0.0)
        self.disturb_start_step = self.args.get('disturb_start_step', 100)
        self.failure_threshold = 6.5  # 失效角度阈值（度）
        self.recovery_threshold = 5.0  # 恢复角度阈值（度）
        self.failure_consecutive_steps = 10  # 连续多少步超过阈值算失效

        # 追踪变量
        self.disturbance_active = False
        self.disturbance_injected_step = None
        self.failure_detected_step = None
        self.disturbance_cancelled_step = None
        self.recovery_detected_step = None
        self.angle_history = []
        self.max_angle_in_episode = 0.0
        
        # 创建过滤后的args，移除扰动参数避免传给底层环境
        filtered_args = {k: v for k, v in self.args.items() 
                        if k not in ['disturbance_mode', 'disturb_target_agent', 
                                    'disturb_magnitude', 'disturb_start_step']}
        
        # self.module = multiwalker_v9
        self.base_env, self.raw_env = env_with_raw(**filtered_args)
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

        self.sigh = True

        super().__init__(args)

    def Delete_specified_agent(self, agent_id):
        if self.disabled_walker_id >= 0 and self.cur_step >= 100:
            disabled_id = disabled_agent_id = f'walker_{self.disabled_walker_id}'
            disabled_id = int(disabled_agent_id.split('_')[-1])

            actions[disabled_id] = np.array([-15.0, -15.0, -15.0, -15.0])
            obs, rew, term, trunc, info = self.env.step(self.wrap(list(actions)))
            obs: ObsWrappedType

            for agent_id in self.agents:
                current_id_num = int(agent_id.split('_')[-1])
                # 获取被禁用 agent 的左邻居的 ID
                left_neighbor_id = f"walker_{disabled_id - 1}"
                right_neighbor_id = f"walker_{disabled_id + 1}"
                if current_id_num == disabled_id + 1:
                    # 确保这个左邻居真的存在于环境中
                    if left_neighbor_id in obs:
                        # 核心步骤：将被禁用 agent 的左邻居观测信息，赋予给当前 agent 的左邻居观测
                        obs[agent_id][24:26] = obs[left_neighbor_id][24:26]
                    else:
                        obs[agent_id][24:26] = np.zeros_like(obs[agent_id][24:26])
                elif current_id_num == disabled_id - 1:
                        if right_neighbor_id in obs:
                            obs[agent_id][26:28] = obs[right_neighbor_id][26:28]
                    # 如果被禁用 agent 是最左边的 (walker_0)
                        else:
                            obs[agent_id][26:28] = np.zeros_like(obs[agent_id][26:28])
            
            # 3. 强制清空被禁用 Agent 自己的所有观测
                if agent_id == disabled_agent_id:
                    if agent_id in obs:
                        obs[agent_id][:] = np.zeros_like(obs[agent_id])

    def _check_failure_condition(self, current_angle: float) -> bool:
        """检查是否满足失效条件：连续10步角度超过6.5度"""
        self.angle_history.append(abs(current_angle))
        if len(self.angle_history) > self.failure_consecutive_steps:
            self.angle_history.pop(0)
        
        if len(self.angle_history) == self.failure_consecutive_steps:
            return all(angle > self.failure_threshold for angle in self.angle_history)
        return False

    def _check_recovery_condition(self, current_angle: float) -> bool:
        """检查是否已恢复：角度小于5度"""
        return abs(current_angle) < self.recovery_threshold

    def _calculate_metrics(self) -> dict:
        """计算三个关键指标"""
        metrics = {
            'mttf': None,  # 平均失效时间
            'recovery_time': None,  # 恢复速度
            'max_angle': self.max_angle_in_episode  # 最大角度
        }
        
        if self.disturbance_injected_step is not None and self.failure_detected_step is not None:
            metrics['mttf'] = self.failure_detected_step - self.disturbance_injected_step
        
        if self.disturbance_cancelled_step is not None and self.recovery_detected_step is not None:
            metrics['recovery_time'] = self.recovery_detected_step - self.disturbance_cancelled_step
        
        return metrics

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
        # 新的自适应扰动逻辑
        if self.disturbance_mode == 'adaptive':
            # 1. 在指定步数注入扰动
            if (self.disturb_target_agent >= 0 and 
                self.cur_step == self.disturb_start_step):
                self.disturbance_active = True
                self.disturbance_injected_step = self.cur_step
            
            # 2. 如果扰动激活，修改动作（所有4个维度）
            if self.disturbance_active and self.disturb_target_agent >= 0:
                actions[self.disturb_target_agent] = actions[self.disturb_target_agent] + self.disturb_magnitude
        
        # 修改某一维观测值，num是某一维
        if self.disabled_walker_id != -1 and self.cur_step >= 100 and self.cur_step <= 1500:
            num = 30
            disabled_agent_id = f'walker_{self.disabled_walker_id}'
            obs, rew, term, trunc, info = self.env.step(self.wrap(list(actions)))
            obs: ObsWrappedType
            obs[disabled_agent_id][3] += 0.2
            obs[disabled_agent_id][4] += 0.02
            # print(f"obs: {obs[disabled_agent_id][0:4]}")
            # print(f"disabled_agent_id: {disabled_agent_id}, obs: {obs[disabled_agent_id]}")
        
        # 修改动作，添加随机扰动
        elif self.disabled_walker_id == 100 and self.cur_step >= 100:
            disabled_agent_id = f'walker_{self.disabled_walker_id}'
            print(f"self.disturb={self.disturb}")
            if self.sigh:
                actions[self.disabled_walker_id][0] += self.disturb
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
            obs: ObsWrappedType
            if self.sigh and abs(obs[disabled_agent_id][30])/ 3.14 * 180 >= 6.5:
                self.sigh = False
            print(f"{abs(obs[disabled_agent_id][30])/ 3.14 * 180},{self.sigh}")
            

            # obs[disabled_agent_id] [:-1] += self.disturb
        
        else:
            obs, rew, term, trunc, info = self.env.step(self.wrap(actions))
            obs: ObsWrappedType


        # 自适应扰动：监测失效和恢复
        if self.disturbance_mode == 'adaptive':
            # 获取当前角度
            assert self.multiwalker_env.package is not None
            current_angle_rad = self.multiwalker_env.package.angle
            current_angle_deg = abs(current_angle_rad) / 3.14 * 180
            
            # 更新最大角度
            self.max_angle_in_episode = max(self.max_angle_in_episode, current_angle_deg)
            
            # 监测失效并自动取消扰动
            if self.disturbance_active and self._check_failure_condition(current_angle_deg):
                if self.failure_detected_step is None:
                    self.failure_detected_step = self.cur_step
                    self.disturbance_active = False
                    self.disturbance_cancelled_step = self.cur_step
            
            # 监测恢复
            if (self.disturbance_cancelled_step is not None and 
                self.recovery_detected_step is None and
                self._check_recovery_condition(current_angle_deg)):
                self.recovery_detected_step = self.cur_step

        self.cur_step += 1

        for agent in self.agents:
            assert self.multiwalker_env.package is not None
            info[agent]["package_angle"] = (
                self.multiwalker_env.package.angle / 3.14 * 180
            )
            info[agent]["curr_step"] = self.cur_step
            info[agent]["package_x"] = self.multiwalker_env.package.position.x
            # 添加package接触地面标志
            info[agent]["package_touched_ground"] = getattr(self.multiwalker_env, 'package_touched_ground', False)
            
            # 添加扰动指标到info
            if self.disturbance_mode == 'adaptive':
                metrics = self._calculate_metrics()
                info[agent].update({
                    'disturbance_mttf': metrics['mttf'],
                    'disturbance_recovery_time': metrics['recovery_time'],
                    'disturbance_max_angle': metrics['max_angle'],
                    'disturbance_active': self.disturbance_active,
                    'current_angle_deg': abs(self.multiwalker_env.package.angle) / 3.14 * 180
                })

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

        # # 手动创建移除agent的信息   没印象了，这是在干嘛？
        # if self.cur_step >= 100:
        #     for agent_id in self.agents:
        #         if agent_id not in obs:
        #             obs[agent_id] = np.zeros_like(obs[list(obs.keys())[0]])
        #             rew[agent_id] = 0.0
        #             term[agent_id] = True
        #             info[agent_id] = {}

        # if all(dones.values()):
        #     self.multiwalker_env.render(close=True)
        #     env_instance = self.multiwalker_env
        #     if hasattr(env_instance, 'total_steps') and env_instance.total_steps > 0:
        #         average_angle = env_instance.total_package_angle / env_instance.total_steps
        #         print("-" * 50)
        #         print(f"111回合结束! 杆子平均角度: {average_angle:.4f} 弧度")
        #         print(f"总步数: {env_instance.total_steps}")
        #     else:
        #         print("本次运行步数为0，无法计算平均角度。")

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
        self.sigh = True
        
        # 重置扰动追踪变量
        self.disturbance_active = False
        self.disturbance_injected_step = None
        self.failure_detected_step = None
        self.disturbance_cancelled_step = None
        self.recovery_detected_step = None
        self.angle_history = []
        self.max_angle_in_episode = 0.0
        
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
