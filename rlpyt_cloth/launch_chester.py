import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from softgym.registered_env import env_arg_dict
from rlpyt_cloth.rlpyt.experiments.scripts.dm_control.qpg.sac.train.softgym_sac import run_task

@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'corl_camera_ready_qpg_cloth'
    vg = VariantGenerator()
    vg.add('algorithm', ['qpg'])
    vg.add('env_name', ['ClothFlatten', 'ClothFold'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_camera_name', ['default_camera'])
    vg.add('env_kwargs_render', [True])
    vg.add('env_kwargs_observation_mode', ['cam_rgb'])
    vg.add('env_kwargs_num_picker', [1])
    vg.add('env_kwargs_action_repeat', [1])
    vg.add('env_kwargs_horizon', [20])
    vg.add('env_kwargs_action_mode', ['picker_qpg'])
    vg.add('env_kwargs_reward_type', lambda env_name: ['index', 'bigraph'] if env_name == 'RopeAlphaBet' else [None])  # only for ropealphabet
    vg.add('config_key', ['sac_pixels_cloth_corner_softgym'])
    vg.add('random_location', [True])
    vg.add('sac_module', ['sac_v2'])
    vg.add('sac_agent_module', ['sac_agent_v2'])
    vg.add('seed', [100, 200, 300])

    if not debug:
        # Add possible vgs for non-debug purpose
        pass
    else:
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
