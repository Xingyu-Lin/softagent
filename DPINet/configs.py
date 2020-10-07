env_configs = \
    {
        'ClothFlatten': dict(
            # object states:
            # [x, y, z, xdot, ydot, zdot]
            state_dim=6,
            position_dim=3,
            # object attr:
            # [cloth, root, sphere]
            attr_dim=3,

            # relation attr:
            # [none]
            # relation_dim=3, # Determined by the launch file

            n_instance=1,
            time_step=100,
            time_step_clip=0,
            n_stages=4,
            n_roots=30,

            neighbor_radius=0.008,
            phases_dict=dict(
                root_num=30,
                root_sib_radius=[[5.0]],  # NOTE: not actually used
                root_des_radius=[[0.2]],  # NOTE: not actually used
                root_pstep=[[2]],
                instance=["fluid"],  # NOTE: not actually used
                material=["fluid"]

            ),
            n_rollout=100, # Was 1000 with 50 timesteps
            nstep_eval_rollout=2,
            dataf='./datasets/ClothFlatten/'
        )
    }
