env_configs = \
    {
        'ClothFlatten': dict(
            # object states:
            # 5 velocity + one-hot of picked / normal particle
            state_dim=17,
            position_dim=0,
            relation_dim = 6,
            # object attr:
            attr_dim=0,
            env_name='ClothFlatten',

            time_step=100,
            time_step_clip=0,

            neighbor_radius=0.008,
            phases_dict=dict(),
            n_rollout=1000,# Was 500 # Was 1000 with 50 timesteps
            nstep_eval_rollout=2,
            dataf='./datasets/ClothFlatten_small/'
        ),
        'BoxBath': dict(
            n_rollout = 3000,

            # object states:
            # [v_i, v_i-1, v_i-2, v_i-3, v_i-4, distance_to_wall (5)] (3*5 + 5)
            state_dim = 20,
            position_dim = 0,

            # object attr:
            # MaterialEncoder(One_Hot_Vector)
            attr_dim = 2,

            # relation attr:
            # [(x_i - x_j), || x_i - x_j ||] (3 + 1)
            relation_dim = 4,

            time_step = 151,
            time_step_clip = 0,
            n_instance = 2,
            n_stages = 4,

            neighbor_radius = 0.08,

            # ball, fluid
            phases_dict=dict(
                instance_idx = [0, 64, 1024],
                radius = 0.08,
                instance = ['cube', 'fluid'],
                material = ['rigid', 'fluid'],
            )
        )
    }
        
