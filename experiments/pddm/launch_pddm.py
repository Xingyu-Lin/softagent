import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from experiments.pddm.train import run_task
from softgym.registered_env import env_arg_dict


per_step_total_num = 21000
planning_horizons = {
    'PassWater': 7,
    'PourWater': 40, 
    'ClothFold': 15,
    'ClothFlatten': 15,
    'ClothDrop': 15,
    'RopeFlatten': 15
}

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0309-pddm-all-fix-per-step-num-longer-horizon'
    vg = VariantGenerator()
    vg.add('algorithm', ['PDDM'])
    if not debug:
        vg.add('env_name', ['PassWater', 'PourWater', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'RopeFlatten'])
    else:
        vg.add('env_name', ['PassWater'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_camera_name', ['default_camera'])
    vg.add('env_kwargs_render', [False])
    vg.add('env_kwargs_observation_mode', ['key_point'])
    vg.add('seed', [100])
    vg.add('max_episode_length', [200])

    if not debug:
        vg.add('sample_size', lambda env_name: [per_step_total_num // planning_horizons[env_name]])
        vg.add('beta', [0.8])
        vg.add('action_correlation', [True, False])
        vg.add('gamma', [1.0])
        vg.add('sigma', [0.9])
        vg.add('test_episodes', [10])
        vg.add('plan_horizon', lambda env_name: [planning_horizons[env_name]])
    else:
        vg.add('sample_size', [1000])
        vg.add('action_correlation', [False])
        vg.add('beta', [0.8])
        vg.add('gamma', [1.0])
        vg.add('sigma', [0.9])
        vg.add('test_episodes', [2])
        vg.add('plan_horizon', [3])
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 1:
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
