from harl.common.base_logger import BaseLogger


class PettingZooSumoLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(PettingZooSumoLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.episode = 1
        self.is_testing = False

    def init(self, episodes):
        """Initialize the logger."""
        super().init(episodes)

    def get_task_name(self):
        return "pettingzoo_sumo"

    def eval_init(self):
        super().eval_init()
