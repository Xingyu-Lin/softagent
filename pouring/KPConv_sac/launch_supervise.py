import time
import torch
import click
import socket
from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from pouring.KPConv_sac.supervise_rim_to_state import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0929-supervise-rim-2-reduced-state-train-longer'

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

    vg.add('env_name', ['PourWater'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_observation_mode', ['rim_interpolation'])
    vg.add('env_kwargs_action_mode', ['rotation_top'])

    vg.add('algorithm', ['KPConv_supervise_rim_2_rstate'])
    vg.add('batch_size', [200])
    vg.add('scale_reward', lambda env_name: [reward_scales[env_name]])
    vg.add('clip_obs', lambda env_name, env_kwargs_observation_mode: [clip_obs[env_name]] if env_kwargs_observation_mode == 'key_point' else [None])
    vg.add('KPConv_config_final_hidden_dim', [1024])
    vg.add('KPConv_config_first_subsampling_dl', [None])
    vg.add('KPConv_config_in_features_dim', [1])
    vg.add('KPConv_config_use_batch_norm', [True, False])
    vg.add('KPConv_deform', [True, False])
    vg.add('env_kwargs_deterministic', [False])
    vg.add('save_tb', [False])
    vg.add('save_video', [True])
    vg.add('save_model', [False])
    vg.add('output_dim', [9])
    vg.add('valid_interval', [1])
    vg.add('save_interval', [25])
    vg.add('num_episodes', [1000])
    vg.add('train_epochs', [2000])
    vg.add('lr', [1e-3])
    vg.add('load_data_path', ['data/local/pouring/supervise_data'])
    vg.add('save_data_path', ['data/local/pouring/supervise_data'])

    if not debug:
        vg.add('seed', [100])
    else:
        vg.add('seed', [200])
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
