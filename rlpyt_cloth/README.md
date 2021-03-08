# Learning to Manipulate Deformable Objects without Demonstrations

This directory contains code that adapts from the [paper](https://arxiv.org/abs/1910.13439)

## rlpyt Usage
See the [original library](https://github.com/astooke/rlpyt) for more information on the design of the library

## Installation

Install mujoco-py with mujoco200

Install a custom version of [dm_control](https://github.com/wilson1yan/dm_control)

Install a custom version of [dm_env](https://github.com/wilson1yan/dm_env)

Install the original rlpyt environment

## Running

All launch scripts are in rlpyt/experiments/scripts/dm_control/qpg/sac/launch

### Cloth

For Cloth (State), see launch_dm_control_sac_state_cloth_point.py

For Cloth (Pixel), see launch_dm_control_sac_pixels_cloth_point.py

For Cloth-Simplified (State), see launch_dm_control_sac_state_cloth_corner.py

### Rope

For Rope (State), see launch_dm_control_state_rope.py

For Rope (Pixel), see launch_dm_control_pixels_rope.py
