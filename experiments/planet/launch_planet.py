import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from experiments.planet.train import run_task
from softgym.registered_env import env_arg_dict


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '1122_planet'
    vg = VariantGenerator()
    vg.add('algorithm', ['planet'])
    if debug:
        vg.add('env_name', ['ClothFlatten','PourWater', 'ClothDrop','RopeFlatten'])
    else:
        vg.add('env_name', ['ClothDrop', 'PourWater', 'PassWater', 'ClothFlatten', 'RopeFlatten', 'ClothFold'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_camera_name', ['default_camera'])
    vg.add('env_kwargs_delta_reward', [False])

    vg.add('train_epoch', [1200])
    vg.add('planning_horizon', [24])
    vg.add('use_value_function', [False])
    if debug:
        vg.add('seed', [100])
    else:
        vg.add('seed', [100, 200, 300])

    if not debug:
        vg.add('collect_interval', [100])
        vg.add('test_interval', [10])
        vg.add('test_episodes', lambda env_name: [900 // env_arg_dict[env_name]['horizon']])
        vg.add('episodes_per_loop', lambda env_name: [900 // env_arg_dict[env_name]['horizon']])
        # Add possible vgs for non-debug purpose
        pass
    else:
        vg.add('collect_interval', [1])
        vg.add('test_interval', [1])
        vg.add('test_episodes', [1])
        vg.add('episodes_per_loop', [1])
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
