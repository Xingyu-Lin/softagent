import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator

from experiments.model_free.train_model_free import run_task
from softgym.registered_env import env_arg_dict


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)  # mainly for debug
def main(mode, debug, dry):
    exp_prefix = '0205_sac_cloth_flatten_updated_cam_rgb'
    vg = VariantGenerator()
    vg.add('env_name', ['ClothFlatten'])
    vg.add('env_kwargs', lambda env_name: [env_arg_dict[env_name]])
    vg.add('env_kwargs_observation_mode', ['cam_rgb'])
    vg.add('algorithm', ['SAC'])
    vg.add('version', ['normal'])
    vg.add('layer_size', [256])
    vg.add('replay_buffer_size', lambda env_kwargs_observation_mode: [int(8E4) if env_kwargs_observation_mode == 'cam_rgb' else int(1E6)])
    vg.add('embedding_size', [256])
    vg.add('image_dim', [128])
    vg.add('trainer_kwargs', lambda algorithm: [dict(discount=0.99,
                                                     soft_target_tau=5e-3,
                                                     target_update_period=1,
                                                     policy_lr=3E-4,
                                                     qf_lr=3E-4,
                                                     reward_scale=1,
                                                     use_automatic_entropy_tuning=True,
                                                     ) if algorithm == 'SAC' else dict(discount=0.99,
                                                                                       policy_learning_rate=1e-3,
                                                                                       qf_learning_rate=1e-3)])
    vg.add('max_episode_length', [200])
    vg.add('seed', [100, 200, 300, 400, 500])
    if not debug:
        vg.add('algorithm_kwargs', [dict(num_epochs=500,
                                         num_eval_steps_per_epoch=900,
                                         num_trains_per_train_loop=1000,
                                         num_expl_steps_per_train_loop=1000,
                                         min_num_steps_before_training=1000,
                                         max_path_length=200,
                                         batch_size=256,
                                         dump_policy_video_interval=20)])
    else:
        vg.add('algorithm_kwargs', [dict(num_epochs=500,
                                         num_eval_steps_per_epoch=120,
                                         num_trains_per_train_loop=2,
                                         num_expl_steps_per_train_loop=120,
                                         min_num_steps_before_training=120,
                                         max_path_length=200,
                                         batch_size=256,
                                         dump_policy_video_interval=1)])
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
