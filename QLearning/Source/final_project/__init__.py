from maddpg_master.experiments import GLOBALS
g_env = GLOBALS.global_env

from maddpg_master.maddpg.common import tf_util as U
from multiagent_particle_envs_master.multiagent import scenarios
from multiagent_particle_envs_master.multiagent.environment import MultiAgentEnv
from maddpg_master.maddpg.trainer.maddpg import MADDPGAgentTrainer
from maddpg_master.maddpg.trainer import replay_buffer


import train


arg_list = train.parse_args()
train.train(arg_list)

