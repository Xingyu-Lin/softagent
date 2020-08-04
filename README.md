# SoftAgent
This repository contains all the algorithms that we run for the CoRL sobumission of the SoftGym paper. 

### Instructions
1. Install SoftGym by following the instructions in SoftGym repository. Then, copy the softgym as a submodule in this directory by running
```
cp -r ../softgym ./
```

2. Activate the conda environment by running `. ./prepare_1.0.sh`.

3. Generating initial states for different SoftGym environments: `python experiments/generate_cached_states.py` 

4. Running CEM experiments: `python experiments/cem/launch_cem.py`

5. Running CURL/SAC experiments: `python experiments/curl/launch_curl.py`

6. Running PlaNet experiments: `python experiments/planet/launch_planet.py` 

### PyFleX APIs
Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs.

## References
- CURL implementation is from the official release: https://github.com/MishaLaskin/curl
- PlaNet implementation is modified from this repository: https://github.com/Kaixhin/PlaNet
- Experiments organizer is from this repository: https://github.com/Xingyu-Lin/chester
- Experiments visualization is from rllab: https://github.com/rll/rllab