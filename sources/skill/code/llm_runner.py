from harl.runners.on_policy_ma_runner import OnPolicyMARunner
import torch
import omegaconf
import os
import rich
import time
import numpy as np
from harl.utils.trans_tools import _t2n
from harl.envs.pettingzoo_mw.pettingzoo_mw_env import PettingZooMWEnv
from harl.envs.pettingzoo_mw.walker.multiwalker.mw_move import MultiWalkerEnv
from .llm.agent import generate_prompt
from openai import AsyncOpenAI
from dotenv import load_dotenv


def _to_dict(cfg1) -> dict:
    return omegaconf.OmegaConf.to_container(cfg1, resolve=True, throw_on_missing=True)  # pyright: ignore


# 加载环境变量
load_dotenv()

# 配置OpenAI客户端
client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
)


async def get_model_response(model, prompt_content):
    """获取单个模型的响应"""
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt_content}],
        )
        print(response)
        return model, response.choices[0].message.content
    except Exception as e:
        return model, f"Error: {str(e)}"


class InstructRunner(OnPolicyMARunner):
    def __init__(self, args, algo_args, env_args):
        super().__init__(args, algo_args, env_args)
        # {"fast": "...model_dir..."}
        self.envs: PettingZooMWEnv = self.envs

    def _policy_reset(self):
        self.restore()

    def _get_raw_env(self) -> MultiWalkerEnv:
        return self.envs.raw_env.env

    @torch.no_grad()
    async def exec(self, render_mode: str = "rgb_array"):
        """Render the model."""
        print("start rendering14")
        render_rgb_array = []
        rewards_arr = []
        episode_obses_arr = []
        lidar_obs_arr = []
        for _i in range(self.algo_args["render"]["render_episodes"]):
            episode_rgb_array = []
            episode_obses = [[] for _ in range(self.num_agents)]
            episode_lidar_obs = [[] for _ in range(self.num_agents)]
            eval_obs, _, eval_available_actions = self.envs.reset()
            eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
            eval_available_actions = (
                np.expand_dims(np.array(eval_available_actions), axis=0)
                if eval_available_actions is not None
                else None
            )
            eval_rnn_states = np.zeros(
                (
                    self.env_num,
                    self.num_agents,
                    self.recurrent_n,
                    self.rnn_hidden_size,
                ),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)
            rewards = 0  # 把此处的reward也返回出去
            steps = 0
            self._policy_reset()
            while True:
                steps += 1
                eval_actions_collector = []
                # obs送入rl，产生action
                for agent_id in range(self.num_agents):
                    eval_actions, temp_rnn_state = self.actor[agent_id].act(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions is not None
                        else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)

                    eval_actions_collector.append(_t2n(eval_actions))

                # step，产生新的obs
                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                (
                    eval_obs,
                    _,
                    eval_rewards,
                    eval_dones,
                    _,
                    eval_available_actions,
                ) = self.envs.step(eval_actions[0])

                # 记录reward
                rewards += eval_rewards[0][0]
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                eval_available_actions = (
                    np.expand_dims(np.array(eval_available_actions), axis=0)
                    if eval_available_actions is not None
                    else None
                )
                # render
                if self.manual_render:
                    if render_mode == "rgb_array":
                        episode_rgb_array.append(self.envs.render())
                    else:
                        self.envs.render()

                # 记录obs
                for agent_id in range(self.num_agents):
                    episode_obses[agent_id].append(eval_obs[:, agent_id])
                    episode_lidar_obs[agent_id].append(
                        self.envs.get_thru_lidar_obs()[agent_id]
                    )
                if self.manual_delay:
                    time.sleep(0.1)
                if eval_dones[0]:
                    print(f"total reward of this episode: {rewards}, {steps}")
                    if steps < 500:
                        print(f"{_i} terminate early: {steps}")
                    break

                # 如果此时还能运行，说明没done，引入llm代码
                if steps % 50 == 0:
                    # 每100步调用一轮llm
                    llm_obses = [
                        episode_obses[agent_id][-1]
                        for agent_id in range(self.num_agents)
                    ]
                    llm_lidar_obses = [
                        episode_lidar_obs[agent_id][-1]
                        for agent_id in range(self.num_agents)
                    ]
                    raw_env = self._get_raw_env()
                    target_vs = [
                        raw_env.get_target_v_agent(agent_id)
                        for agent_id in range(self.num_agents)
                    ]
                    prompt = generate_prompt(
                        llm_obses, llm_lidar_obses, target_vs
                    )  # needs
                    print(prompt)
                    start_time = time.time()
                    model, response = await get_model_response("gpt-4o", prompt)
                    end_time = time.time()
                    rich.print(f"Time taken: {end_time - start_time} seconds")
                    rich.print(response)
                    if response is None:
                        raise Exception("Response is None")
                    import json

                    target_vs = json.loads(response)["target_vs"]
                    for agent_id in range(self.num_agents):
                        raw_env.set_t_v_agent(agent_id, target_vs[agent_id])
                        print(
                            f"Changed actor {agent_id}  target_v to {target_vs[agent_id]}"
                        )

            render_rgb_array.append(episode_rgb_array)
            rewards_arr.append(rewards)
            episode_obses_arr.append(episode_obses)
            lidar_obs_arr.append(episode_lidar_obs)

        if render_mode == "rgb_array":
            return render_rgb_array, rewards_arr, episode_obses_arr, lidar_obs_arr
        else:
            return None, None, None, None
