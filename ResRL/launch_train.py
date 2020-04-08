import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from ResRL.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0408_resRL_ablation_unet'
    vg = VariantGenerator()

    env_arg_dict = {
        "Box1D": {}
    }
    vg.add('env_name', ['Box1D'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_image_observation', [True])
    vg.add('env_kwargs_image_dim', [128])
    vg.add('visual_encoder_name', lambda env_kwargs_image_observation: [None] if not env_kwargs_image_observation else [
        'UNet',
        # 'Residual',
        # 'VisualEncoderFc1d',
        # 'VisualEncoderConv1d',
        # 'VisualEncoder',
    ])
    vg.add('weight_decay',
           lambda visual_encoder_name: [0, 5e-5] if visual_encoder_name is not None and 'Residual' in visual_encoder_name else [0, 5e-5])
    vg.add('max_episode_length', [200])  # Upper bound on the horizon. Not used here
    vg.add('max_timesteps', [2e5])
    # vg.add('max_timesteps', [1e4])
    vg.add('seed', [100, 200, 300])

    if not debug:
        # Add possible vgs for non-debug purpose
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
        # if mode == 'seuss':
        #     if idx == 0:
        #         compile_script = 'compile.sh'  # For the first experiment, compile the current softgym
        #         wait_compile = None
        #     else:
        #         compile_script = None
        #         wait_compile = 120  # Wait 30 seconds for the compilation to finish
        # else:
        compile_script = wait_compile = None  # No need to compile Flex for this

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
