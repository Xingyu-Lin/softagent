import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from curl.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0718_corl_rope_curl_lr'
    reward_scales = {
        'PourWater': 20.0,
        'PassWaterTorus': 20.0,
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
        'PourWater': None,
        'PassWaterTorus': None,
        'PourWaterAmount': None,
        'ClothFold': (-3, 3),
        'ClothFoldCrumpled': None,
        'ClothFoldDrop': None,
        'ClothFlatten': (-2, 2),
        'ClothDrop': None,
        'RopeFlatten': None,
        'RopeFlattenNew': (-3, 3),
        'RopeAlphaBet': None,
        'RigidClothFold': (-3, 3),
        'RigidClothDrop': None,
    }

    vg = VariantGenerator()

    vg.add('env_name', ['RopeFlattenNew'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_observation_mode', ['cam_rgb', 'key_point'])

    vg.add('algorithm', ['CURL'])
    vg.add('alpha_fixed', [False])
    vg.add('critic_lr', lambda env_kwargs_observation_mode: [3e-4] if env_kwargs_observation_mode == 'cam_rgb' else [1e-3])
    vg.add('actor_lr', lambda critic_lr: [critic_lr])
    vg.add('lr_decay', [0.01])
    vg.add('init_temperature', lambda env_kwargs_observation_mode: [0.1] if env_kwargs_observation_mode == 'cam_rgb' else [0.1])
    vg.add('replay_buffer_capacity', lambda env_kwargs_observation_mode: [100000] if env_kwargs_observation_mode == 'cam_rgb' else [100000])
    vg.add('num_train_steps', lambda env_kwargs_observation_mode: [1000000] if env_kwargs_observation_mode == 'cam_rgb' else [2000000])

    vg.add('scale_reward', lambda env_name: [reward_scales[env_name]])
    vg.add('clip_obs', lambda env_name: [clip_obs[env_name]])
    vg.add('batch_size', [128])
    vg.add('env_kwargs_deterministic', [False])
    vg.add('save_tb', [False])
    vg.add('save_video', [True])
    vg.add('save_model', [True])
    vg.add('seed', [100, 200, 300, 400, 500])

    if not debug:
        pass
    else:
        pass
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
