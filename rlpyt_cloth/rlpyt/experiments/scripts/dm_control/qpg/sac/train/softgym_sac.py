import sys
import importlib

from rlpyt.utils.launching.affinity import affinity_from_code, encode_affinity
from rlpyt.samplers.parallel.cpu.sampler import CpuSampler
from rlpyt.samplers.parallel.cpu.collectors import CpuResetCollector
from rlpyt.envs.dm_control_env import DMControlEnv, init_namedtuples
from rlpyt.runners.minibatch_rl import MinibatchRlEval
from rlpyt.utils.logging.context import logger_context
from rlpyt.utils.launching.variant import load_variant, update_config

from rlpyt.experiments.configs.dm_control.qpg.sac.softgym_sac import configs
from softgym.registered_env import SOFTGYM_ENVS, env_arg_dict, ClothFlattenEnv
from softgym.envs.qpg_wrapper import QpgWrapper

import cv2 as cv


# def build_and_train(slot_affinity_code, log_dir, run_ID, config_key):
#     variants = load_variant(log_dir)
#     print(variants)
#     exit()

def update_env_kwargs(vv):
    new_vv = vv.copy()
    for v in vv:
        if v.startswith('env_kwargs_'):
            arg_name = v[len('env_kwargs_'):]
            new_vv['env_kwargs'][arg_name] = vv[v]
            del new_vv[v]
    return new_vv


def cv_render(img, name='GoalEnvExt', scale=1):
    '''Take an image in ndarray format and show it with opencv. '''
    img = img[:, :, :3]
    new_img = img[:, :, (2, 1, 0)] / 256.
    h, w = new_img.shape[:2]
    new_img = cv.resize(new_img, (w * scale, h * scale))
    cv.imshow(name, new_img)
    cv.waitKey(20)


def run_task(vv, log_dir, exp_name):
    vv = update_env_kwargs(vv)
    run_ID = vv['seed']
    config_key = vv['config_key']
    # Original experiments: slot_affinity_code ='0slt_20cpu_1gpu_36hto_1skt'
    slot_affinity_code = encode_affinity(
        n_cpu_core=20,
        n_gpu=2,
        cpu_per_run=2,
        cpu_per_worker=2,
        n_socket=1,
        run_slot=0,
        set_affinity=True,  # it can help to restrict workers to individual CPUs
    )
    # slot_affinity_code = '0slt_20cpu_1gpu_36hto_1skt'

    # slot_affinity_code = vv['affinity_code']
    affinity = affinity_from_code(slot_affinity_code)
    # affinity['all_cpus'] = (0,)
    # affinity['master_cpus'] = (0,)
    # affinity['workers_cpus'] = (0,)
    config = configs[config_key]
    config.update(**vv)
    config["env"] = env_arg_dict[config['env_name']]
    vv['env_kwargs']['headless'] = True
    # env = ClothFlattenEnv(
    #     observation_mode='cam_rgb',
    #     action_mode='picker_qpg',
    #     num_picker=1,
    #     render=True,
    #     headless=False,
    #     horizon=75,
    #     action_repeat=1,
    #     render_mode='cloth',
    #     save_cached_states=False)
    # env = QpgWrapper(env)
    # obs = env.reset()
    # for i in range(50):
    #     action = env.action_space.sample()
    #     location = env.sample_location(obs)
    #     action[:2] = location
    #     obs, _, _, _ = env.step(action)

    sac_module = 'rlpyt.algos.qpg.{}'.format(config['sac_module'])
    sac_agent_module = 'rlpyt.agents.qpg.{}'.format(config['sac_agent_module'])

    sac_module = importlib.import_module(sac_module)
    sac_agent_module = importlib.import_module(sac_agent_module)

    SAC = sac_module.SAC
    SacAgent = sac_agent_module.SacAgent

    if 'pixel_wrapper_kwargs' in config['env']:
        info_keys = config.get('info_keys', None)
        state_keys = config.get('state_keys', None)
        init_namedtuples(info_keys=info_keys, state_keys=state_keys)

    #     eval_env = config['env'].copy()
    #     eval_env['task_kwargs']['train_mode'] = False

    sampler = CpuSampler(
        EnvCls=SOFTGYM_ENVS[vv['env_name']],
        env_kwargs=vv['env_kwargs'],
        CollectorCls=CpuResetCollector,
        eval_env_kwargs=vv['env_kwargs'],
        **config["sampler"]
    )
    # sampler = CpuSampler(
    #     EnvCls=SOFTGYM_ENVS[vv['env_name']],
    #     env_kwargs=vv['env_kwargs'],
    #     CollectorCls=CpuResetCollector,
    #     eval_env_kwargs=vv['env_kwargs'],
    #     **config["sampler"]
    # )
    algo = SAC(optim_kwargs=config["optim"], **config["algo"])
    agent = SacAgent(**config["agent"])
    runner = MinibatchRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        affinity=affinity,
        **config["runner"]
    )
    name = "sac_{}".format(vv['env_name'])

    with logger_context(log_dir, run_ID, name, log_params=config, snapshot_mode='last'):
        runner.train()


if __name__ == "__main__":
    build_and_train(*sys.argv[1:])
