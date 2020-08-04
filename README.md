# Softagent
This repository contains all the reinforcement learning algorithms that we run for the CoRL sobumission of the SoftGym paper. 

### Instructions
1. Install SoftGym by following the instructions in SoftGym repository. Then, copy the softgym as a submodule in this directory by running
```
cp -r ../softgym ./
```

2. Activate the conda environment by running `. ./prepare_1.0`.

3. Running CEM algorithms: CEM experiments can be run by `python experiments/cem`

### PyFleX APIs
Please see the example test scripts and the bottom of `bindings/pyflex.cpp` for available APIs.

## References

- NVIDIA FleX - 1.2.0 [README](doc/README_FleX.md)
- PyFleX: https://github.com/YunzhuLi/PyFleX