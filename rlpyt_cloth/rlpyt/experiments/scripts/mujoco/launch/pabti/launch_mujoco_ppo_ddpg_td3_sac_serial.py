
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

affinity_code = encode_affinity(
    n_cpu_core=8,
    n_gpu=8,
    contexts_per_gpu=1,
    hyperthread_offset=24,
    n_socket=1,
    # cpu_per_run=2,
)
runs_per_setting = 5
variant_levels_1M = list()
variant_levels_3M = list()

n_steps = [1e6]
values = list(zip(n_steps))
dir_names = ["1M"]
keys = [("runner", "n_steps")]
variant_levels_1M.append(VariantLevel(keys, values, dir_names))

n_steps = [3e6]
values = list(zip(n_steps))
dir_names = ["3M"]
keys = [("runner", "n_steps")]
variant_levels_3M.append(VariantLevel(keys, values, dir_names))


env_ids = ["Hopper-v3", "Walker2d-v3"]
values = list(zip(env_ids))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels_1M.append(VariantLevel(keys, values, dir_names))

env_ids = ["Ant-v3", "HalfCheetah-v3"]
values = list(zip(env_ids))
dir_names = ["{}".format(*v) for v in values]
keys = [("env", "id")]
variant_levels_3M.append(VariantLevel(keys, values, dir_names))


variants_1M, log_dirs_1M = make_variants(*variant_levels_1M)
variants_3M, log_dirs_3M = make_variants(*variant_levels_3M)
variants = variants_1M + variants_3M
log_dirs = log_dirs_1M + log_dirs_3M

default_config_key = "ppo_1M_serial"
script = "rlpyt/experiments/scripts/mujoco/pg/train/mujoco_ppo_serial.py"
experiment_title = "ppo_mujoco"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)

# default_config_key = "ppo_32ep_1mb"
# script = "rlpyt/experiments/scripts/mujoco/pg/train/mujoco_ppo_serial.py"
# experiment_title = "ppo_mujoco_32x1"

# run_experiments(
#     script=script,
#     affinity_code=affinity_code,
#     experiment_title=experiment_title,
#     runs_per_setting=runs_per_setting,
#     variants=variants,
#     log_dirs=log_dirs,
#     common_args=(default_config_key,),
# )


default_config_key = "ddpg_from_td3_1M_serial"
script = "rlpyt/experiments/scripts/mujoco/qpg/train/mujoco_ddpg_serial.py"
experiment_title = "ddpg_mujoco"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)

default_config_key = "td3_1M_serial"
script = "rlpyt/experiments/scripts/mujoco/qpg/train/mujoco_td3_serial.py"
experiment_title = "td3_mujoco"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)

default_config_key = "sac_1M_serial"
script = "rlpyt/experiments/scripts/mujoco/qpg/train/mujoco_sac_serial.py"
experiment_title = "sac_mujoco"

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
