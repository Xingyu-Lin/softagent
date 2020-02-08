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
    'learning_rate_schedule': 0,  # Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)
    'learning_rate': 1e-3,
    'adam_epsilon': 1e-4,
    'experience_replay': None,  # Path to the saved experiences,
    'experience_size': 80000,  # Original implementation has an unlimited buffer size, but 1 million is the max experience collected anyway
    'seed_episodes': 5,
    'action_noise': 0.3,

    'batch_size': 50,
    'chunk_size': 50,
    'saved_models': None,
    'free_nats': 3,

    'global_kl_beta': 0,  # Global KL weight (0 to disable)
    'grad_clip_norm': 1000,

    # Overshooting parameters
    'overshooting_distance': 50,  # Latent overshooting distance/latent overshooting weight for t = 1
    'overshooting_kl_beta': 0,  # Latent overshooting KL weight for t > 1 (0 to disable)
    'overshooting_reward_scale': 0,  # Latent overshooting reward prediction weight for t > 1 (0 to disable)

    # Planner parameters:
    'planning_horizon': 12,  # Planning horizon
    'optimisation_iters': 10,  # Planning optimization iterations
    'candidates': 1000,  # Candidate samples per iteration
    'top_candidates': 100,  # Number of top candidates to fit

    # ENV parameters
    'symbolic_env': False,  # Whether the environment returns symbolic observation
    'bit_depth': 5,  # Image bit depth (quantisation)
    'action_repeat': 1,

    # Testing parameters
    'test_interval': 10,
    'test_episodes': 30,
    'checkpoint_interval': 50,
    'checkpoint_experience': False,  # Whether to save the collected experience replay to file
}
