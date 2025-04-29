import logging
import sys

import gym

loc = "uam.envs:"

# ----------- 环境注册--------------------------

gym.register(
    id="UAM-Modular-v0",
    entry_point=loc + "UAM_Modular"
)
gym.register(
    id="UAM-Modular-Validation-v0",
    entry_point=loc + "UAM_Modular_Validation"
)

# 初始化记录器logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('[%(levelname)s] - %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)
