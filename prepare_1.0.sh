cd softgym
. prepare_1.0.sh
cd ..
export PYTORCH_JIT=0
export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}/rlpyt_cloth:${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:${PWD}/rlkit/rlkit:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export MUJOCO_gl=egl
export MUJOCO_GL=osmesa # This is for running the rlpyt code from the cloth manipulation paper
export EGL_GPU=$CUDA_VISIBLE_DEVICES # This can be used for specifying the GPU used to run the environment.