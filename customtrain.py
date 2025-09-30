#!/usr/bin/env python3
"""
分布式训练脚本 - Python 版本
功能等同于原始 Bash 脚本：
  #!/usr/bin/env bash
  CONFIG=$1
  GPUS=$2
  PORT=${PORT:-28509}

  PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
  python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
      $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import random


def main():
    # 创建参数解析器
    config='./projects/configs/bevformerv2/bevformerv2-r50-t8-24ep.py'
    gpus=1

    # 设置默认端口（如果未提供环境变量）
    r = random.randint(-100, 100)
    port = os.environ.get('PORT', str(28600+r))

    # 获取当前脚本所在目录
    current_dir = Path(__file__).parent.resolve()
    parent_dir = current_dir.parent

    # 设置 PYTHONPATH
    os.environ['PYTHONPATH'] = f"{parent_dir}:{os.environ.get('PYTHONPATH', '')}"

    # 构建训练脚本路径
    train_script = current_dir / "train.py"

    sys.path.append(parent_dir)

    # 构建完整的命令列表
    command = [
        sys.executable,  # 使用当前 Python 解释器
        "-m", "torch.distributed.launch",
        f"--nproc_per_node={gpus}",
        f"--master_port={port}",
        str(train_script),
        config,
        "--launcher", "pytorch",
        "--deterministic"
    ]

    # 打印调试信息（可选）
    print("执行命令:")
    print(" ".join(command))
    print(f"工作目录: {os.getcwd()}")
    print(f"PYTHONPATH: {os.environ['PYTHONPATH']}")

    # 执行命令
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"训练失败，退出码: {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()