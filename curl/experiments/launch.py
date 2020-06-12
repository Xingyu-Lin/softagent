import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
from curl.train import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    exp_prefix = '0612_curl'
    vg = VariantGenerator()

    task_dict = {
        'cartpole': 'swingup',
        'finger': 'spin',
        'reacher': 'easy',
        'cheetah': 'run',
        'walker': 'walk',
        'ball_in_cup': 'catch'
    }
    vg.add('domain_name', ['cartpole', 'finger', 'reacher'])
    vg.add('task_name', lambda domain_name: [task_dict[domain_name]])
    vg.add('bc_update', [True, False])
    vg.add('bc_actor_loss_threshold', lambda bc_update: [1e-1] if bc_update else [None])
    vg.add('bc_critic_loss_threshold', lambda bc_update: [1e-3] if bc_update else [None])
    vg.add('bc_aux_repeat', lambda bc_update: [1, 5] if bc_update else [1])
    vg.add('action_repeat', [8])
    vg.add('save_tb', [True])
    vg.add('save_video', [True])
    # vg.add('slow_update', [True, False])
    vg.add('seed', [100, 200, 300])

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
