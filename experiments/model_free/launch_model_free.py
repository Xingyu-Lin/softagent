import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator

from experiments.model_free.train_model_free import run_task
from experiments.registered_env import env_arg_dict


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)  # mainly for debug
def main(mode, debug, dry):
    exp_prefix = '0125_td3_key_point'
    vg = VariantGenerator()
    vg.add('env_name', ['ClothFlatten', 'PourWater', 'RopeFlatten', 'ClothFold'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_observation_mode', ['key_point'])
    vg.add('algorithm', ['TD3'])
    vg.add('version', ['normal'])
    vg.add('layer_size', [256])
    vg.add('replay_buffer_size', [int(1E5)])
    vg.add('embedding_size', [1024])
    vg.add('image_dim', [128])
    vg.add('trainer_kwargs', [dict(discount=0.99,
                                   soft_target_tau=5e-3,
                                   target_update_period=1,
                                   policy_lr=3E-4,
                                   qf_lr=3E-4,
                                   reward_scale=1,
                                   use_automatic_entropy_tuning=True,
                                   )])
    vg.add('max_episode_length', [200])
    vg.add('seed', [100, 200])

    if not debug:
        vg.add('algorithm_kwargs', [dict(num_epochs=3000,
                                         num_eval_steps_per_epoch=5000,
                                         num_trains_per_train_loop=1000,
                                         num_expl_steps_per_train_loop=1000,
                                         min_num_steps_before_training=1000,
                                         max_path_length=200,
                                         batch_size=256)])
    else:
        vg.add('algorithm_kwargs', [dict(num_epochs=3000,
                                         num_eval_steps_per_epoch=120,
                                         num_trains_per_train_loop=2,
                                         num_expl_steps_per_train_loop=120,
                                         min_num_steps_before_training=120,
                                         max_path_length=200,
                                         batch_size=256)])
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
