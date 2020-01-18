# parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
# parser.add_argument('--env', type=str, default='Pendulum-v0', choices=GYM_ENVS + CONTROL_SUITE_ENVS,
#                     help='Gym/Control Suite environment')
# parser.add_argument('--episodes', type=int, default=1000, metavar='E', help='Total number of episodes')
#
# parser.add_argument('--checkpoint-experience', action='store_true', help='Checkpoint experience replay')
# parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
# parser.add_argument('--render', action='store_true', help='Render environment')

DEFAULT_PARAMS = {
    'image_dim': 128,
    'hidden_size': 200,
    'belief_size': 200,
    'state_size': 30,
    'embedding_size': 1024,
    # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used

    'collect_interval': 100,  # Number of optimizer steps per data collection

    'activation_function': 'relu',
    'learning_rate_repr': 6e-4,
    'learning_rate_ac': 8e-5,
    'adam_epsilon': 1e-4,
    'experience_replay': None,  # Path to the saved experiences,
    'experience_size': 1000000,  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    'seed_episodes': 5,
    'action_noise': 0.3,

    'batch_size': 50,
    'chunk_size': 50,
    'saved_models': None,
    'free_nats': 3,

    'global_kl_beta': 0,  # Global KL weight (0 to disable)
    'grad_clip_norm': 1000,
    'imagine_horizon': 12,  # Planning horizon
    'gamma': 0.99,  # Value function
    'lambda': 0.95,  # Value function

    # ENV parameters
    'symbolic_env': False,  # Whether the environment returns symbolic observation
    'bit_depth': 5,  # Image bit depth (quantisation)
    'action_repeat': 1,
    'max_episode_length': 75,

    # Testing parameters
    'test_interval': 10,
    'test_episodes': 8,
    'checkpoint_interval': 50,
    'checkpoint_experience': False,  # Whether to save the collected experience replay to file
}
