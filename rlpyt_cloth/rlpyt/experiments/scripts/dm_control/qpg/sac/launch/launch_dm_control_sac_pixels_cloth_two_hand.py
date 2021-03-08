
from rlpyt.utils.launching.affinity import encode_affinity
from rlpyt.utils.launching.exp_launcher import run_experiments
from rlpyt.utils.launching.variant import make_variants, VariantLevel

script = "rlpyt/experiments/scripts/dm_control/qpg/sac/train/dm_control_sac.py"
affinity_code = encode_affinity(
    n_cpu_core=20,
    n_gpu=4,
    contexts_per_gpu=1,
)

runs_per_setting = 4
default_config_key = "sac_pixels_cloth_corner"
experiment_title = "sac_dm_control_pixels_cloth_two_hand_fixed_env"
variant_levels = list()

domain = ['cloth_two_hand']
task = ['easy']
values = list(zip(domain, task))
dir_names = ["env_{}_{}".format(*v) for v in values]
keys = [('env', 'domain'), ('env', 'task')]
variant_levels.append(VariantLevel(keys, values, dir_names))

model_cls = ['PiConvModel']#, 'GumbelPiConvModel']
random_location = [True]#, False]
sac_module = ['sac_v2']#, 'sac_v2_generic']
sac_agent_module = ['sac_agent_v2']#, 'sac_agent_v2_generic']
state_keys= [['location', 'pixels']]#, ['pixels']]
values = list(zip(model_cls, random_location, sac_module, sac_agent_module, state_keys))
dir_names = ["model_cls_{}_rnd_loc_{}".format(*v) for v in values]
keys = [('agent', 'ModelCls'), ('env', 'task_kwargs', 'random_location'),
        ('sac_module',), ('sac_agent_module',), ('state_keys',)]
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
