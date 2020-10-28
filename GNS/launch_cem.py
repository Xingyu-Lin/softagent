import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from GNS.run_cem import run_task
from softgym.registered_env import env_arg_dict


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '1023_cem'
    vg = VariantGenerator()
    cem_plan_horizon = {
        'ClothFlatten': 15
    }
    vg.add('algorithm', ['CEM'])
    vg.add('env_name', ['ClothFlatten'])
    vg.add('action_low', [[-0.01, -0.01, -0.01, 0] * 2])
    vg.add('action_high', [[0.01, 0.01, 0.01, 1] * 2])
    vg.add('action_size', [8])
    vg.add('normalize', [False])

    # low-level action tunes this
    vg.add('action_repeat', [1])
    vg.add('mode', ['pick-and-place-uv'])
    vg.add('uv_sample_method', ['uniform', 'bounding_box'])
    if mode == 'seuss':
        vg.add('model_dir', ['data/local/1018_GNS_deeper_larger_noise_smaller_picker/1018_GNS_deeper_larger_noise_smaller_picker_2020_10_18_16_55_11_0007'])
        vg.add('model_name', ['net_epoch_31_iter_85499.pth'])
        vg.add('num_worker', [10])
    else:
        vg.add('model_dir', ['data/yufei_seuss_data/1018_GNS_deeper_larger_noise_smaller_picker/1018_GNS_deeper_larger_noise_smaller_picker/1018_GNS_deeper_larger_noise_smaller_picker_2020_10_18_16_55_11_0007'])
        vg.add('model_name', ['net_epoch_31_iter_85499.pth'])
        vg.add('num_worker', [0])

    if not debug:
        # low-level actions
        vg.add('optimisation_iters', [10])
        vg.add('planning_horizon', lambda env_name: [cem_plan_horizon[env_name]])
        vg.add('timestep_per_decision', [21000])
        vg.add('test_episodes', [5])
        
        # pick-and-place
        vg.add('cem_num_pick', [500])
        vg.add('delta_y', [0.3])
        vg.add('move_distance', [0.15])
        vg.add('pick_and_place_num', [5])
        vg.add('cem_stage_1_step', [30])
        vg.add('cem_stage_2_step', [60])
        vg.add('cem_stage_3_step', [40])

        vg.add('seed', [100, 200, 300, 400, 500])

    else:
        vg.add('optimisation_iters', [5])
        vg.add('test_episodes', [1])
        vg.add('timestep_per_decision', [100])
        vg.add('planning_horizon', [7])

        vg.add('cem_num_pick', [5])
        vg.add('delta_y', [0.3])
        vg.add('pick_and_place_num', [4])
        vg.add('move_distance', [0.15])
        vg.add('cem_stage_1_step', [30])
        vg.add('cem_stage_2_step', [60])
        vg.add('cem_stage_3_step', [40])

        vg.add('seed', [100])

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
