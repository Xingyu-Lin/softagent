GNN_default_config = dict(
    actor_kwargs=dict(
        feature_dim=256,
        log_std_min=-2,
        log_std_max=10,
    ),

    critic_kwargs=dict(
        q_kwargs=dict(
            feature_dim=256,
        )
    ),

    encoder_kwargs=dict(
        input_node_dim=4,
        input_edge_dim=1, 
        feature_dim=256,
        gnn_layer_kwargs=dict(
            node_dim=256,
            edge_dim=256,
            effect_processor_hidden_layers=[256],
            update_processor_hidden_layers=[256]
        ),
        pooling_layer_kwargs=dict(
            in_channels=256,
            ratio=0.8,
        ),
        num_gnn_layers=3
    )
)