# Learning to Simulate Complex Physics with Graph Networks (https://arxiv.org/pdf/2002.09405.pdf)

The code only implements what's needed for training the Graph Network Simulator from data. It expects the data to be pre-generated and stored in the same way as DPI-Net stores it:

```
├── data
│  ├── your_env (as set by args.dataf and args.env)
│  │   ├── train
│  │   │   ├── 0
│  │   │   │   ├── 0.h5
│  │   │   │   └── ....
│  │   │   ├── 1
│  │   │   │   ├── 0.h5
│  │   │   │   └── ....
│  │   │   └── ....
│  │   ├── valid
│  │   │   ├── 0
│  │   │   │   ├── 0.h5
│  │   │   │   └── ....
│  │   │   ├── 1
│  │   │   │   ├── 0.h5
│  │   │   │   └── ....
│  │   │   └── ....
│  │   └── stat.h5
│  └──
└──
```

### Dependencies

NumPy, SciPy, PyTorch, PyFleX, torch_geometric, torch_scatter

### Graph Network Model Implementation (models_graph_res.py)

The Graph Network Module is defined by three modules, an Encoder, a Processor, and a Decoder. The Encoder is used for encoding the nodes and edges in the input graph, the Processor is used
for message passing on the encoded graph, and the Decoder uses the latent encodings back to predict the acceleration.

The backbone of the Processor is implemented as GNBlock, as decribed in "Relational inductive biases, deep learning, and graph networks" (https://arxiv.org/pdf/1806.01261.pdf). A number of these
blocks are stacked together with residual connections to perform multi-step message passing. Currently it is hardcoded to perform 5 steps of passing, but can be easily extended in the Processor class.

Note that despite the paper not using global updates, this feature has been enabled by default. It can be turned off to save computational resources by passing use_global=False to the Processor.

### Data loading (data_graph.py)

This file needs to be editted to parse input from new/different environments (if args.env == XXX). One must make sure to supply the correct keys to self.data_names (the code assumes that the first two
loaded keys are particle positions and velocities) to load the stored h5py file and calculate the correct distance to walls function.

If particle positions and velocities are not the first two loaded items in self.data_names, the rest of the code must also be adjusted accordingly.

### Training (train_graph_res.py)

This file needs to be edited for environment specific arguments (if args.env == XXX). Available arguments are similar to what DPI-Net expects.

Run

    python train_graph_res.py --env <ENV_NAME_HERE>

to train the model.

### Evaluation (eval_graph.py)

This file needs to be edited for environment specific arguments (if args.env == XXX). Available arguments are similar to what DPI-Net expects. Additionally, prepare_input() should also be updated
to reflect the version used in data_graph.py if it was edited.

The evaluation script first performs the rollouts, saves the results, then renders them. The rendering blocks will need to be commented out if rendering fails (eg. on clusters).

Run

    python eval_graph.py --env <ENV_NAME_HERE>

to evaluate the model. The saved model with the lowest validation loss will be used.