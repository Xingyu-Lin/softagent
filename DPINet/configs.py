env_configs = \
    {
        'ClothFlatten': dict(
            # object states:
            # [x, y, z, xdot, ydot, zdot]
            state_dim=6,
            position_dim=3,
            # object attr:
            # [cloth, root, sphere 0 - 1]
            attr_dim=4,

            # relation attr:
            # [none]
            relation_dim=1,

            n_instance=1,
            time_step=51,
            time_step_clip=0,
            n_stages=4,
            n_roots=30,

            neighbor_radius=0.013,
            phases_dict=dict(
                root_num=30,
                root_sib_radius=[[5.0]],  # NOTE: not actually used
                root_des_radius=[[0.2]],  # NOTE: not actually used
                root_pstep=[[2]],
                instance=["fluid"],  # NOTE: not actually used
                material=["fluid"]

            ),
            n_rollout=1000,
            dataf='./datasets/ClothFlatten/'
        )
    }
