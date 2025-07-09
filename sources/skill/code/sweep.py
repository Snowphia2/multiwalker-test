#!/usr/bin/env python3
"""
使用joblib并发执行wandb sweep agent命令
"""

import os
import argparse
from joblib import Parallel, delayed


def run_wandb_agent(agent_id: str):
    """执行单个wandb agent命令"""
    cmd = f"wandb agent {agent_id}"
    print(f"启动agent: {agent_id}")
    os.system(cmd)
    print(f"完成agent: {agent_id}")


# wandb agent yuzh2001-iscas/HARL_mw-src/pfewgbxu
def main():
    parser = argparse.ArgumentParser(description="并发执行wandb sweep agents")
    parser.add_argument(
        "--agent_id",
        type=str,
        default="yuzh2001-iscas/mw_harl_new/dsj7sr4l",
        help="wandb agent ID",
    )
    parser.add_argument("--n_jobs", type=int, default=12, help="并发数量")

    args = parser.parse_args()

    print(f"启动 {args.n_jobs} 个并发agent执行: {args.agent_id}")

    # 使用joblib并发执行
    Parallel(n_jobs=3)(delayed(run_wandb_agent)(args.agent_id) for _ in range(3))

    print("所有agent执行完成")


if __name__ == "__main__":
    main()
