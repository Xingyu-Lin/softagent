PATH=~/software/miniconda3/bin:$PATH
cd softgym
. prepare.sh
cd ..
export PYTORCH_JIT=0
export PYFLEXROOT=${PWD}/softgym/PyFlexRobotics
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export MUJOCO_gl=egl
export EGL_GPU=$CUDA_VISIBLE_DEVICES
