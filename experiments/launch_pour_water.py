import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from experiments.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = 'pour_water'
    env_arg_dict = {
        'PourWater': {'observation_mode': 'cam_img',
                      'action_mode': 'direct',
                      'render_mode': 'fluid',
                      'deterministic': True,
                      'render': True,
                      'headless': True,
                      'horizon': 75,
                      'camera_name': 'cam_2d'},
    }
    vg = VariantGenerator()
    vg.add('env_name', ['PourWater'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('seed', [100])

    if not debug:
        # Add possible vgs for non-debug purpose
        pass
    else:
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))

    sub_process_popens = []
    for vv in vg.variants():
        while len(sub_process_popens) >= 1:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
