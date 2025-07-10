from harl.envs.pettingzoo_mw.walker.disturbances.move import SkillMoveBase
from harl.envs.pettingzoo_mw.walker.multiwalker.mw_move import MOVE_DOESNT_CARE


class SkillMoveSpeed(SkillMoveBase):
    """
    对包裹质量做扰动的类。

    disturbance_args: dict = {"mass": 4.57}
    """

    def start(self):
        super().start()
        self.env.set_target_v(self.disturbance_args["speed"])
        print(f"SkillMoveSpeed: {self.disturbance_args['speed']}")

    def end(self):
        self.env.set_target_v(MOVE_DOESNT_CARE)
        super().end()
        print(f"SkillMoveSpeed: {MOVE_DOESNT_CARE}")
