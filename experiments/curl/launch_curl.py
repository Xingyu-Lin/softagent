import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from curl.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0713_cloth_flatten'
    vg = VariantGenerator()

    vg.add('env_name', ['ClothFlatten'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_observation_mode', ['key_point', 'cam_rgb'])

    vg.add('algorithm', ['CURL'])
    vg.add('alpha_fixed', [False])
    vg.add('critic_lr', lambda env_kwargs_observation_mode: [3e-4] if env_kwargs_observation_mode == 'cam_rgb' else [1e-3])
    vg.add('actor_lr', lambda critic_lr: [critic_lr])
    vg.add('init_temperature', lambda env_kwargs_observation_mode: [0.1] if env_kwargs_observation_mode == 'cam_rgb' else [0.1])
    vg.add('replay_buffer_capacity', lambda env_kwargs_observation_mode: [100000] if env_kwargs_observation_mode == 'cam_rgb' else [100000])
    vg.add('num_train_steps', lambda env_kwargs_observation_mode: [1000000] if env_kwargs_observation_mode == 'cam_rgb' else [2000000])

    vg.add('scale_reward', [50.])
    vg.add('batch_size', [128])
    vg.add('env_kwargs_deterministic', [False])
    vg.add('save_tb', [False])
    vg.add('save_video', [True])
    vg.add('save_model', [True])
    vg.add('seed', [100, 200])

    if not debug:
        pass
    else:
        pass
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode == 'seuss':
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
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
