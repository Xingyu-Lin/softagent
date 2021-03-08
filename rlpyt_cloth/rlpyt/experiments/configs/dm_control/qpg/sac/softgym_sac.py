configs = {}
config = dict(
    env_name='ClothFlatten',
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
        n_steps=1e6,
        log_interval_steps=1e4,
    ),
    sampler=dict(
        is_pixel=True,
        batch_T=1,
        batch_B=16,
        max_decorrelation_steps=0,
        eval_n_envs=16,
        eval_max_steps=5000,
        eval_max_trajectories=50,
    ),
    env=dict(
        domain='cloth_corner',
        task='easy',
        max_path_length=120,
        pixel_wrapper_kwargs=dict(observation_key='pixels', pixels_only=False,  # to not take away non pixel obs
                                  render_kwargs=dict(width=64, height=64, camera_id=0)),
        task_kwargs=dict(random_location=True, pixels_only=True, train_mode=True)  # to not return positions and only pick location
    ),
)
configs["sac_pixels_cloth_corner_softgym"] = config
