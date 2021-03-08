import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from experiments.cem.eval_cem import run_task
from softgym.registered_env import env_arg_dict


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0808_cem_cloth_ppp'
    vg = VariantGenerator()
    cem_plan_horizon = {
        'PassWater': 7,
        'PourWater': 40,
        'PourWaterAmount': 40,
        'ClothFold': 15,
        'ClothFoldPPP': 5,
        'ClothFoldCrumpled': 30,
        'ClothFoldDrop': 30,
        'ClothFlatten': 15,
        'ClothFlattenPPP': 5,
        'ClothDrop': 15,
        'RopeFlatten': 15,
        'RopeFlattenNew': 15,
        'RopeAlphaBet': 15,
        'RigidClothFold': 15,
        'RigidClothDrop': 15,
    }
    vg.add('env_kwargs_camera_name', ['default_camera'])
    vg.add('env_kwargs_render', [False])
    vg.add('env_kwargs_observation_mode', ['key_point'])
    vg.add('env_kwargs_reward_type', lambda env_name: ['index', 'bigraph'] if env_name == 'RopeAlphaBet' else [None])  # only for ropealphabet

    vg.add('seed', [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])
    vg.add('max_episode_length', [200])

    if not debug:
        vg.add('max_iters', [10])
        vg.add('plan_horizon', lambda env_name: [cem_plan_horizon[env_name]])
        vg.add('timestep_per_decision', [21000])
        vg.add('test_episodes', [1])
        vg.add('use_mpc', [True])
        # Add possible vgs for non-debug purpose
        pass
    else:
        vg.add('max_iters', [1])
        vg.add('test_episodes', [1])
        vg.add('timestep_per_decision', [100])
        vg.add('use_mpc', [True])
        vg.add('plan_horizon', [7])
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
