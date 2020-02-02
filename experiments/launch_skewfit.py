import time
import click
from chester.run_exp import run_experiment_lite, VariantGenerator
import numpy as np
import rlkit.torch.vae.vae_schedules as vae_schedules
from rlkit.launchers.skewfit_experiments import skewfit_full_experiment
from rlkit.torch.vae.conv_vae import imsize48_default_architecture, imsize84_default_architecture, imsize128_default_architecture
from softgym.registered_env import env_arg_dict as env_arg_dicts


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False) # mainly for debug
def main(mode, debug, dry):
    
    # env_arg_dicts = {
    #     "PourWater": {
    #         'observation_mode': 'point_cloud', # will be later wrapped by ImageEnv
    #         'action_mode': 'direct',
    #         'render_mode': 'fluid',
    #         'deterministic': False,
    #         'render': True,
    #         'headless': True,
    #         'horizon': 75,
    #         "num_variations": 200,
    #     },
    #     "PassWater": {
    #         "observation_mode": 'point_cloud', 
    #         "horizon": 75, 
    #         "action_mode": 'direct', 
    #         "deterministic": False, 
    #         "render_mode":'fluid', 
    #         "render": True, 
    #         "headless": True,
    #         "num_variations": 200,
    #     }
    # }


    skewfit_args = dict(
        algorithm='Skew-Fit',
        double_algo=False,
        online_vae_exploration=False,
        imsize=128,
        init_camera=None,
        skewfit_variant=dict(
            save_video=True,
            custom_goal_sampler='replay_buffer',
            online_vae_trainer_kwargs=dict(
                beta=20,
                lr=1e-3,
            ),
            save_video_period=100,
            qf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            policy_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            vf_kwargs=dict(
                hidden_sizes=[400, 300],
            ),
            max_path_length=75,
            algo_kwargs=dict(
                batch_size=1024,
                num_epochs=1000,
                num_eval_steps_per_epoch=900,
                num_expl_steps_per_train_loop=500, # how many env transitions to add to replay buffer before each new train loop.
                num_trains_per_train_loop=1000, # epoch -> train loop -> num_trains_per_train_loop
                min_num_steps_before_training=10000,
                vae_training_schedule=vae_schedules.custom_schedule_2,
                oracle_data=False,
                vae_save_period=20,
                parallel_vae_train=False,
                dump_policy_video_interval=20
            ),
            twin_sac_trainer_kwargs=dict(
                discount=0.99,
                reward_scale=1,
                soft_target_tau=1e-3,
                target_update_period=1,  # 1
                use_automatic_entropy_tuning=True,
            ),
            replay_buffer_kwargs=dict(
                start_skew_epoch=10,
                max_size=int(100000),
                fraction_goals_rollout_goals=0.2,
                fraction_goals_env_goals=0.5,
                exploration_rewards_type='None',
                vae_priority_type='vae_prob',
                priority_function_kwargs=dict(
                    sampling_method='importance_sampling',
                    decoder_distribution='gaussian_identity_variance',
                    num_latents_to_sample=10,
                ),
                power=0,
                relabeling_goal_sampling_mode='vae_prior', # this is useless
            ),
            exploration_goal_sampling_mode='reset_of_env', # using this, we do not use goals sampled from vae prior for exploration
            evaluation_goal_sampling_mode='reset_of_env',
            normalize=False,
            render=False,
            exploration_noise=0.0,
            exploration_type='ou',
            training_mode='train',
            testing_mode='test',
            reward_params=dict(
                type='latent_distance',
            ),
            observation_key='latent_observation', # sac uses latent_observation as input
            desired_goal_key='latent_desired_goal', # reward is computed as latent space distance
            vae_wrapped_env_kwargs=dict(
                sample_from_true_prior=True,
            ),
        ),
        train_vae_variant=dict(
            representation_size=4,
            beta=20,
            num_epochs=0,
            dump_skew_debug_plots=False, # not actually used in the code
            decoder_activation='gaussian',
            generate_vae_dataset_kwargs=dict(
                N=40,
                test_p=.9,
                use_cached=False,
                show=False,
                oracle_dataset=True,
                oracle_dataset_using_set_to_goal= True,
                n_random_steps=75,
                non_presampled_goal_img_is_garbage=False,
            ),
            vae_kwargs=dict(
                input_channels=3,
                architecture=imsize128_default_architecture,
                decoder_distribution='gaussian_identity_variance',
            ),
            # TODO: why the redundancy?
            algo_kwargs=dict(
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=64,
                lr=1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                skew_dataset=True,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                    num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=25,
        ),
    )


    exp_prefix = 'RIG-128-0201-all'
    vg = VariantGenerator()
    assert skewfit_args['imsize'] == 128
    print("imsize is: ", skewfit_args['imsize'])

    import copy
    skewfit_argss = []

    if debug:
        env_list = ['RopeManipulate']
    else:
        env_list = ['RopeManipulate', "PourWaterGoal", "PassWaterGoal", "ClothDropGoal", "ClothManipulate", "ClothFoldGoal"]
    for env_id in env_list:
        print("-" * 50, env_id, '-' * 50)
        env_arg_dict = env_arg_dicts[env_id]
        for representation_size in [32]:
            for power in [0]:
                s_args = copy.deepcopy(skewfit_args)
                s_args['env_id'] = env_id
                s_args['env_arg_dict'] = env_arg_dict
                s_args['train_vae_variant']['representation_size'] = representation_size
                s_args['skewfit_variant']['replay_buffer_kwargs']['power'] = power
                s_args['skewfit_variant']['max_path_length'] = env_arg_dict['horizon']
                s_args['train_vae_variant']['algo_kwargs']['skew_config']['power'] = power
                if debug:
                    print("=" * 50, "debug", "=" * 50)
                    s_args['skewfit_variant']['algo_kwargs']['batch_size'] = 10
                    s_args['skewfit_variant']['algo_kwargs']['num_trains_per_train_loop'] = 1
                    s_args['skewfit_variant']['algo_kwargs']['min_num_steps_before_training'] = 10
                    s_args['skewfit_variant']['algo_kwargs']['dump_policy_video_interval'] = 1
                    s_args['skewfit_variant']['replay_buffer_kwargs']['max_size'] = int(1000)
                    s_args['train_vae_variant']['generate_vae_dataset_kwargs']['N'] = 5

                print("representation size {} power {}".format(representation_size, power))
                skewfit_argss.append(s_args)

    vg.add('skewfit_kwargs', skewfit_argss)
    if debug:
        vg.add('seed', [100])
    else:
        vg.add('seed', [100, 200, 300, 400, 500])

    print("num of envs: ", len(skewfit_argss))
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
            stub_method_call=skewfit_full_experiment,
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
