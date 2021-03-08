import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from drq.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '1029-Rope-rerun-drq'

    reward_scales = {
        'PourWater': 20.0,
        'PassWaterTorus': 20.0,
        'TransportTorus': 20.0,
        'PassWater': 20.0,
        'PourWaterAmount': 20.0,
        'ClothFold': 50.0,
        'ClothFoldCrumpled': 50.0,
        'ClothFoldDrop': 50.0,
        'ClothFlatten': 50.0,
        'ClothDrop': 50.0,
        'RopeFlatten': 50.0,
        'RopeFlattenNew': 50.0,
        'RopeAlphaBet': 50.0,
        'RigidClothFold': 50.0,
        'RigidClothDrop': 50.0,
    }

    clip_obs = {
        'PassWater': None,
        'PourWater': None,
        'PassWaterTorus': None,
        'PassWater': None,
        'TransportTorus': None,
        'PourWaterAmount': None,
        'ClothFold': None, #(-3, 3),
        'ClothFoldCrumpled': None,
        'ClothFoldDrop': None,
        'ClothFlatten': None, #(-2, 2),
        'ClothDrop': None,
        'RopeFlatten': None,
        'RopeFlattenNew': None, #(-3, 3),
        'RopeAlphaBet': None,
        'RigidClothFold': None, #(-3, 3),
        'RigidClothDrop': None,
    }

    def get_critic_lr(env_name, obs_mode):
        # if env_name in ['ClothFold', 'RigidClothFold', 'PassWaterTorus'] or (env_name =='RopeFlattenNew' and obs_mode =='point_cloud'):
        #     if obs_mode == 'cam_rgb':
        #         return 1e-4
        #     else:
        #         return 5e-4
        # if obs_mode == 'cam_rgb':
        #     return 3e-4
        # else:
        #     return 1e-3
        return 1e-3

    def get_alpha_lr(env_name, obs_mode):
        # if env_name in ['RigidClothFold', 'ClothFold']:
        #     return 2e-5
        # else:
        #     return 1e-3
        return 1e-3

    def get_lr_decay(env_name, obs_mode):
        # if env_name == 'RopeFlattenNew' or (env_name == 'ClothFlatten' and obs_mode == 'cam_rgb') \
        #   or (env_name == 'RigidClothFold' and obs_mode == 'key_point'):
        #     return 0.01
        # elif obs_mode == 'point_cloud':
        #     return 0.01
        # elif env_name == 'PassWaterTorus':
        #     return 0.01
        # else:
        #     return None
        return None

    vg = VariantGenerator()

    vg.add('env_name', ['RopeFlattenNew'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_observation_mode', ['cam_rgb'])

    # vg.add('algorithm', ['CURL'])
    # vg.add('alpha_fixed', [False])
    # vg.add('critic_lr', lambda env_name, env_kwargs_observation_mode: [get_critic_lr(env_name, env_kwargs_observation_mode)])
    # vg.add('actor_lr', lambda critic_lr: [critic_lr])
    # vg.add('alpha_lr', lambda env_name, env_kwargs_observation_mode: [get_alpha_lr(env_name, env_kwargs_observation_mode)])
    # vg.add('lr_decay', lambda env_name, env_kwargs_observation_mode: [get_lr_decay(env_name, env_kwargs_observation_mode)])
    # vg.add('init_temperature', lambda env_kwargs_observation_mode: [0.1] if env_kwargs_observation_mode == 'cam_rgb' else [0.1])
    vg.add('replay_buffer_capacity', lambda env_kwargs_observation_mode: [100000] if env_kwargs_observation_mode == 'cam_rgb' else [100000])
    vg.add('num_train_steps', lambda env_kwargs_observation_mode: [1000000] if env_kwargs_observation_mode == 'cam_rgb' else [1000000])
    vg.add('scale_reward', lambda env_name: [reward_scales[env_name]])
    vg.add('clip_obs', lambda env_name, env_kwargs_observation_mode: [clip_obs[env_name]] if env_kwargs_observation_mode == 'key_point' else [None])
    vg.add('batch_size', [128])
    vg.add('im_size', [128])
    vg.add('env_kwargs_deterministic', [False])
    vg.add('log_save_tb', [False])
    vg.add('save_video', [True])
    vg.add('save_model', [False])
    vg.add('log_interval', [10000])
    # vg.add('num_seed_steps', [100])

    if not debug:
        vg.add('seed', [100, 200, 300])
        pass
    else:
        vg.add('seed', [100])
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()
    gpu_num = torch.cuda.device_count()

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss', 'autobot']:
            if idx == 0:
                compile_script = 'compile_1.0.sh'  # For the first experiment, compile the current softgym
                wait_compile = None
            else:
                compile_script = None
                wait_compile = 120  # Wait 30 seconds for the compilation to finish
        elif mode == 'ec2':
            compile_script = 'compile_1.0.sh'
            wait_compile = None
        else:
            compile_script = wait_compile = None
        if hostname.startswith('autobot') and gpu_num > 0:
            env_var = {'CUDA_VISIBLE_DEVICES': str(idx % gpu_num)}
        else:
            env_var = None
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
