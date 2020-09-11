
import copy

configs = dict()

config = dict(
    agent=dict(
        q_model_kwargs=dict(hidden_sizes=[256, 256]),
        model_kwargs=dict(hidden_sizes=[256, 256]),
    ),
    algo=dict(
        discount=0.99,
        batch_size=1024,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=6e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1,
        replay_ratio=128,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=2e5,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=32,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=50,
    ),
    env=dict(
        domain='cloth_v0',
        task='easy',
        max_path_length=1200,
        task_kwargs=dict(reward='diagonal')
    ),
)

configs["sac_state_clothv0"] = config

config = dict(
    agent=dict(
        ModelCls='PiConvModel',
        QModelCls='QofMuConvModel',
        q_model_kwargs=dict(channels=(64, 64, 64),
                            kernel_sizes=(3, 3, 3), strides=(2, 2, 2),
                            hidden_sizes=[256, 256]),
        model_kwargs=dict(channels=(64, 64, 64),
                          kernel_sizes=(3, 3, 3), strides=(2, 2, 2),
                          hidden_sizes=[256, 256]),
    ),
    algo=dict(
        discount=0.99,
        batch_size=1024,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=6e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1,
        replay_ratio=128,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=2e5,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        is_pixel=True,
        batch_T=1,
        batch_B=16,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=50,
    ),
    env=dict(
        domain='cloth_v0',
        task='easy',
        max_path_length=1200,
        task_kwargs=dict(reward='diagonal'),
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=True,
                                  render_kwargs=dict(width=64, height=64))
    ),
)

configs["sac_pixels_clothv0"] = config

config = copy.deepcopy(configs['sac_state_clothv0'])
config['env']['domain'] = 'cloth_v8'
config['env']['task_kwargs']['mode'] = 'corners'
config['env']['task_kwargs']['distance_weight'] = 0.0

configs["sac_state_clothv8"] = config

config = copy.deepcopy(configs['sac_pixels_clothv0'])
config['env']['domain'] = 'cloth_v0'
config['env']['task_kwargs']['mode'] = 'corners'
config['env']['task_kwargs']['distance_weight'] = 0.0

configs["sac_pixels_clothv8"] = config

config = copy.deepcopy(configs['sac_state_clothv8'])
config['env']['domain'] = 'cloth_v7'
config['env']['task_kwargs']['mode'] = 'corners'
config['env']['task_kwargs']['distance_weight'] = 0.0

configs["sac_state_clothv7"] = config

config = copy.deepcopy(configs['sac_pixels_clothv8'])
config['env']['domain'] = 'cloth_v8'
config['env']['task_kwargs']['mode'] = 'corners'
config['env']['task_kwargs']['distance_weight'] = 0.0

configs["sac_pixels_clothv8"] = config

config = copy.deepcopy(configs['sac_state_clothv8'])
config['env']['domain'] = 'cloth_sim_state'
config['env']['max_path_length'] = 30
config['env']['task_kwargs'] = dict(mode='corners')
config['agent']['q_model_kwargs']['n_tile'] = 20

configs["sac_state_cloth_sim"] = config

config = copy.deepcopy(configs['sac_pixels_clothv8'])
config['env']['domain'] = 'cloth_v8'
config['env']['max_path_length'] = 30
del config['env']['task_kwargs']
config['agent']['q_model_kwargs']['n_tile'] = 20

configs["sac_pixels_cloth_sim"] = config

config = copy.deepcopy(configs['sac_state_clothv0'])
config['runner']['n_steps'] = 1e6
config['env']['domain'] = 'rope_v1'
config['env']['max_path_length'] = 1000
del config['env']['task_kwargs']

configs["sac_state_ropev1"] = config

config = copy.deepcopy(configs['sac_pixels_clothv0'])
config['runner']['n_steps'] = 1e6
config['env']['domain'] = 'rope_v1'
config['env']['max_path_length'] = 1000
del config['env']['task_kwargs']

configs["sac_pixels_ropev1"] = config

config = dict(
    sac_module='sac_v2',
    sac_agent_module='sac_agent_v2',
    name='',
    agent=dict(
        ModelCls='PiMlpModel',
        q_model_kwargs=dict(hidden_sizes=[256, 256]),
        model_kwargs=dict(hidden_sizes=[256, 256]),
    ),
    algo=dict(
        discount=0.99,
        batch_size=1024,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=6e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1,
        replay_ratio=128,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e5,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=32,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=10,
    ),
    env=dict(
        domain='cloth_corner',
        task='easy',
        max_path_length=120,
        task_kwargs=dict(random_location=True)
    ),
)
configs["sac_state_cloth_corner"] = config


config = dict(
    state_keys=None,
    info_keys=None,
    sac_module='sac_v2',
    sac_agent_module='sac_agent_v2',
    name='',
    agent=dict(
        ModelCls='GumbelPiConvModel',
        QModelCls='QofMuConvModel',
        q_model_kwargs=dict(channels=(64, 64, 4),
                            kernel_sizes=(3, 3, 3), strides=(2, 2, 2),
                            hidden_sizes=[256, 256]),
        model_kwargs=dict(channels=(64, 64, 4),
                          kernel_sizes=(3, 3, 3), strides=(2, 2, 2),
                          hidden_sizes=[256, 256]),
        n_qs=2,
    ),
    algo=dict(
        discount=0.99,
        batch_size=1024,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=6e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1,
        replay_ratio=128,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=3e5,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        is_pixel=True,
        batch_T=1,
        batch_B=16,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=50,
    ),
    env=dict(
        domain='cloth_corner',
        task='easy',
        max_path_length=120,
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=False, # to not take away non pixel obs
                                  render_kwargs=dict(width=64, height=64, camera_id=0)),
        task_kwargs=dict(random_location=True, pixels_only=True, train_mode=True) # to not return positions and only pick location
    ),
)
configs["sac_pixels_cloth_corner"] = config

config = dict(
    sac_module='sac_v2',
    sac_agent_module='sac_agent_v2',
    name='',
    agent=dict(
        ModelCls='PiMlpModel',
        q_model_kwargs=dict(hidden_sizes=[256, 256]),
        model_kwargs=dict(hidden_sizes=[256, 256]),
    ),
    algo=dict(
        discount=0.99,
        batch_size=1024,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=6e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1,
        replay_ratio=128,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e5,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        batch_T=1,
        batch_B=32,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=50,
    ),
    env=dict(
        domain='rope_v2',
        task='easy',
        max_path_length=200,
        task_kwargs=dict(random_location=True)
    ),
)
configs["sac_state_rope_v2"] = config

config = dict(
    state_keys=None,
    info_keys=None,
    sac_module='sac_v2',
    sac_agent_module='sac_agent_v2',
    name='',
    agent=dict(
        ModelCls='PiConvModel',
        QModelCls='QofMuConvModel',
        q_model_kwargs=dict(channels=(64, 64, 4),
                            kernel_sizes=(3, 3, 3), strides=(2, 2, 2),
                            hidden_sizes=[256, 256]),
        model_kwargs=dict(channels=(64, 64, 4),
                          kernel_sizes=(3, 3, 3), strides=(2, 2, 2),
                          hidden_sizes=[256, 256]),
        n_qs=2,
    ),
    algo=dict(
        discount=0.99,
        batch_size=1024,
        target_update_tau=0.005,
        target_update_interval=1,
        learning_rate=6e-4,
        reparameterize=True,
        policy_output_regularization=0.0,
        reward_scale=1,
        replay_ratio=128,
    ),
    model=dict(),
    optim=dict(),
    runner=dict(
        n_steps=5e5,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        is_pixel=True,
        batch_T=1,
        batch_B=16,
        max_decorrelation_steps=0,
        eval_n_envs=10,
        eval_max_steps=20000,
        eval_max_trajectories=50,
    ),
    env=dict(
        domain='rope_sac',
        task='easy',
        max_path_length=200,
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=False, # to not take away non pixel obs
                                  render_kwargs=dict(width=64, height=64, camera_id=0)),
        task_kwargs=dict() # to not return positions and only pick location
    ),
)
configs["sac_pixels_rope"] = config
