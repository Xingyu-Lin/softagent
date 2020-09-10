
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/dm_control/qpg/sac/train/dm_control_sac.py"
affinity_code = encode_affinity(
    n_cpu_core=20,
    n_gpu=4,
    contexts_per_gpu=1,
)

runs_per_setting = 2
default_config_key = "sac_state_cloth_sim"
experiment_title = "sac_dm_control_state_cloth_sim"
variant_levels = list()

modes = ['corners', 'border', 'inner_border', '3x3', '5x5', '9x9']
values = list(zip(modes))
dir_names = ['mode_{}'.format(*v) for v in values]
keys = [('env', 'task_kwargs', 'mode')]
variant_levels.append(VariantLevel(keys, values, dir_names))

max_path_lengths = [15, 30]
values = list(zip(max_path_lengths))
dir_names = ['mpl_{}'.format(*v) for v in values]
keys = [('env', 'max_path_length')]
variant_levels.append(VariantLevel(keys, values, dir_names))

variants, log_dirs = make_variants(*variant_levels)

run_experiments(
    script=script,
    affinity_code=affinity_code,
    experiment_title=experiment_title,
    runs_per_setting=runs_per_setting,
    variants=variants,
    log_dirs=log_dirs,
    common_args=(default_config_key,),
)
