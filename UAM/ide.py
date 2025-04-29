import uam.envs
from uam.run import train_continuous as cont
from uam.run import visualize_continuous as vizcont
from uam.agents import validate_agent
from uam.common.configparser import ConfigFile
from uam.configs import __path__ as cont_path

# ---------------- 设置-----------------------------

TASK        = "viz"              # train或viz
CONFIG_FILE = "uam_modular_sim_study.yaml"     # 配置
SEED        = 42                   # 种子
AGENT_NAME  = "LSTMRecTD3"         # 智能体
ACTOR_WEIGHTS = "experiments/LSTMRecTD3_UAM-Modular-v0__2025-04-18_42/LSTMRecTD3_actor_best_weights.pth"        # 权重初始化的文件路径（连续操作）
CRITIC_WEIGHTS = "experiments/LSTMRecTD3_UAM-Modular-v0__2025-04-18_42/LSTMRecTD3_critic_best_weights.pth"       # 权重初始化的文件路径（连续操作）

# ------------------------------------------------------------
# --- 验证智能体名称有效性 ---
validate_agent(AGENT_NAME)  

# --- 配置路径 ---
config_path = f"{cont_path[0]}/{CONFIG_FILE}"
config = ConfigFile(config_path)

# 种子
if SEED is not None:
    config.overwrite(seed=SEED)

# 权重
if CRITIC_WEIGHTS is not None:
    config.overwrite(critic_weights=CRITIC_WEIGHTS)

if ACTOR_WEIGHTS is not None:
    config.overwrite(actor_weights=ACTOR_WEIGHTS)

# 最大步长
config.max_episode_handler()

# 模式
if TASK == "train":
        cont.train(config, AGENT_NAME)
elif TASK == "viz":
        vizcont.test(config, AGENT_NAME)
