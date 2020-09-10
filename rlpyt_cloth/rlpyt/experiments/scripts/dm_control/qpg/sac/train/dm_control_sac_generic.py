
import sys

from rlpyt.utils.launching.affinity import affinity_from_code
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.envs.dm_control_env import DMControlEnv, init_namedtuples
from rlpyt.algos.qpg.sac_v2 import SAC
from rlpyt.agents.qpg.sac_agent_v2 import SacAgent
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.experiments.configs.dm_control.qpg.sac.dm_control_sac import configs


def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
    affinity = affinity_from_code(slot_affinity_code)
    config = configs[config_key]
    variant = load_variant(log_dir)
    config = update_config(config, variant)
    print('Config', config)

    if 'pixel_wrapper_kwargs' in config['env']:
        info_keys = config.get('info_keys', None)
        state_keys = config.get('state_keys', None)
        init_namedtuples(info_keys=info_keys, state_keys=state_keys)

    sampler = CpuSampler(
        EnvCls=DMControlEnv,
        env_kwargs=config["env"],
        CollectorCls=CpuResetCollector,
        eval_env_kwargs=config["env"],
        **config["sampler"]
    )
    algo = SAC(optim_kwargs=config["optim"], **config["algo"])
    agent = SacAgent(**config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "sac_{}_{}".format(config['env']['domain'],
                              config['env']['task'])

    with logger_context(log_dir, run_ID, name, log_params=config, snapshot_mode='last'):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
