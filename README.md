# SoftAgent
This repository contains the benchmarked algorithms for environments in [SoftGym](https://github.com/Xingyu-Lin/softgym) ([paper](https://arxiv.org/abs/2011.07215)). The benchmarked algorithms include
* Cross Entropy Method(CEM) [[source](./cem)]  
* CURL/SAC [[source](./curl)] [[paper](https://proceedings.icml.cc/static/paper_files/icml/2020/5951-Paper.pdf)] 
    * We use the [original implementation](https://github.com/MishaLaskin/curl)
* DrQ [[source](./drq)] [[paper](https://arxiv.org/abs/2004.13649)]
    * We use the [original implementation](https://github.com/denisyarats/drq)
* PlaNet [[source](./planet)] [[paper](https://arxiv.org/abs/1811.04551)]
    * We use this [customized pytorch version](https://github.com/Kaixhin/PlaNet)
* MVP [[source](./rlpyt_cloth)] [[paper](https://arxiv.org/abs/1910.13439)]
    * We build on top of the [original implementation](https://github.com/wilson1yan/rlpyt) 
## Installation 

1. Install SoftGym by following the instructions in [SoftGym](https://github.com/Xingyu-Lin/softgym) repository. Then, copy the softgym code to the SoftAgent root directory so we have the following file structure:
    ```
    softagent
    ├── cem
    ├── ...
    ├── softgym
    ```
2. Update conda env with additional packages required by SoftAgent: `conda env update  --file environment.yml  --prune`
3. Activate the conda environment by running `. ./prepare_1.0.sh`.

4. For running MVP, please refer to the [original implementation](https://github.com/wilson1yan/rlpyt) for dependencies.

## Running benchmarked experiments 

1. Generating initial states for different SoftGym environments: `python experiments/generate_cached_states.py`

2. Running CEM experiments: `python experiments/run_cem.py`. Refer to `run_cem.py` for different arguments.

3. Running CURL/SAC experiments: `python experiments/run_curl.py`. Refer to `run_curl.py` for different arguments.

4. Running PlaNet experiments: `python experiments/run_planet.py`. Refer to `run_planet.py` for different arguments.

5. Running DrQ experiments: `python experiments/run_drq.py`. Refer to `run_drq.py` for different arguments.

5. Train an MVP policy: `python experiments/run_mvp.py`. Refer to `run_mvp.py` for different arguments. Once the model is trained, use `rlpyt_cloth/max_q_eval_policy` to evaluate the policy that selects the pick location with the maximum Q value.

**Note**: Default number of environment variations are set to 1. Set them to 1000 to reproduce the original experiments.

<!-- ### PyFleX APIs
Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs. -->

## Cite
If you find this codebase useful in your research, please consider citing:
```
@inproceedings{corl2020softgym,
 title={SoftGym: Benchmarking Deep Reinforcement Learning for Deformable Object Manipulation},
 author={Lin, Xingyu and Wang, Yufei and Olkin, Jake and Held, David},
 booktitle={Conference on Robot Learning},
 year={2020}
}
```

## References
- CURL implementation is from the official release: https://github.com/MishaLaskin/curl
- PlaNet implementation is modified from this repository: https://github.com/Kaixhin/PlaNet
- DrQ implementation is from the official repository: https://github.com/denisyarats/drq
- MVP implementation is from the official repository: https://github.com/wilson1yan/rlpyt
- Softgym repository: https://github.com/Xingyu-Lin/softgym
<!-- - Experiments organizer is from this repository: https://github.com/Xingyu-Lin/chester -->
