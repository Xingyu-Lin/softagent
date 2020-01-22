import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from experiments.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0121_three_soft_envs'
    env_arg_dict = {
        'PourWater': {'observation_mode': 'cam_rgb',
                      'action_mode': 'direct',
                      'render_mode': 'fluid',
                      'deterministic': False,
                      'render': True,
                      'headless': True,
                      'horizon': 75,
                      'camera_name': 'default_camera',
                      'delta_reward': True},
        'RopeFlatten': {'observation_mode': 'cam_rgb',
                        'action_mode': 'picker',
                        'num_picker': 2,
                        'render': True,
                        'headless': True,
                        'horizon': 75,
                        'action_repeat': 8,
                        'render_mode': 'cloth',
                        'num_variations': 200,
                        'use_cached_states': True,
                        'deterministic': False},
        'ClothFlatten': {'observation_mode': 'cam_rgb',
                         'action_mode': 'picker',
                         'num_picker': 2,
                         'render': True,
                         'headless': True,
                         'horizon': 100,
                         'action_repeat': 8,
                         'render_mode': 'cloth',
                         'num_variations': 200,
                         'use_cached_states': True,
                         'deterministic': False},
        'ClothFold': {'observation_mode': 'cam_rgb',
                      'action_mode': 'picker',
                      'num_picker': 2,
                      'render': True,
                      'headless': True,
                      'horizon': 100,
                      'action_repeat': 8,
                      'render_mode': 'cloth',
                      'num_variations': 200,
                      'use_cached_states': True,
                      'deterministic': False}
    }
    vg = VariantGenerator()
    vg.add('algorithm', ['planet'])
    vg.add('env_name', ['RopeFlatten', 'ClothFlatten', 'ClothFold', 'PourWater'])
    # vg.add('env_name', ['RopeFlatten'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_camera_name', ['default_camera'])
    vg.add('train_episode', [1000])
    vg.add('planning_horizon', [12, 24])
    vg.add('use_value_function', [False])
    vg.add('seed', [100, 200])

    if not debug:
        vg.add('collect_interval', [100])
        # Add possible vgs for non-debug purpose
        pass
    else:
        vg.add('collect_interval', [1])
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 2:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode == 'seuss':
            if idx == 0:
                compile_script = 'compile.sh'  # For the first experiment, compile the current softgym
                wait_compile = None
            else:
                compile_script = None
                wait_compile = 120  # Wait 30 seconds for the compilation to finish
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
