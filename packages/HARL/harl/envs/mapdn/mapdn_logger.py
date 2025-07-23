from harl.common.base_logger import BaseLogger


class MAPDNLogger(BaseLogger):
    def __init__(self, args, algo_args, env_args, num_agents, writter, run_dir):
        super(MAPDNLogger, self).__init__(
            args, algo_args, env_args, num_agents, writter, run_dir
        )
        self.episode = 1
        self.is_testing = False

    def get_task_name(self):
        return "mapdn"
