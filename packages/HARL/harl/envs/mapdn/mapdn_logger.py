from harl.common.base_logger import BaseLogger


class MAPDNLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(MAPDNLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.episode = 1
        self.is_testing = False
        self.test_data = {
            "terminate_at": [],
        }

    def eval_init(self):
        super().eval_init()
        self.test_data = {
            "terminate_at": [],
        }

    def get_task_name(self):
        return "mapdn"

    def eval_per_step(self, eval_data):
        """Log evaluation information per step."""

        (
            eval_obs,
            eval_share_obs,
            eval_rewards,
            eval_dones,
            eval_infos,
            eval_available_actions,
        ) = eval_data
        for i in range(len(eval_infos)):
            if eval_dones[i][0]:
                self.test_data["terminate_at"].append(eval_infos[i][0]["curr_step"])
        for eval_i in range(self.algo_args["eval"]["n_eval_rollout_threads"]):
            self.one_episode_rewards[eval_i].append(eval_rewards[eval_i])
        self.eval_infos = eval_infos

    # def eval_log(self, eval_episode):
    #     """Log evaluation information at the end of an episode."""
    #     self.test_data["terminate_at"].append(eval_episode["terminate_at"])
