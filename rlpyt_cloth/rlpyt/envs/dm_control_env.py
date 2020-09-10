from dm_control import suite
from dm_control.suite.wrappers import pixels
from dm_env.specs import Array, BoundedArray

import numpy as np
import os
import atari_py
import cv2
import copy
from collections import namedtuple, OrderedDict
from rlpyt.utils.collections import namedarraytuple

from rlpyt.envs.base import Env, EnvStep, EnvSpaces
from rlpyt.spaces.box import Box
from rlpyt.spaces.composite import Composite
from rlpyt.utils.quick_args import save__init__args
from rlpyt.samplers.collections import TrajInfo

State = None

def convert_dm_control_to_rlpyt_space(dm_control_space):
    """Recursively convert dm_control_space into gym space.

    Note: Need to check the following cases of the input type, in the following
    order:
       (1) BoundedArray
       (2) Array
       (3) OrderedDict.

    - Generally, dm_control observation_specs are OrderedDict with other spaces
      (e.g. Array) nested in it.
    - Generally, dm_control action_specs are of type `BoundedArray`.

    To handle dm_control observation_specs as inputs, we check the following
    input types in order to enable recursive calling on each nested item.
    """
    if isinstance(dm_control_space, BoundedArray):
        rlpyt_box = Box(
            low=dm_control_space.minimum,
            high=dm_control_space.maximum,
            shape=None,
            dtype=dm_control_space.dtype)
        assert rlpyt_box.shape == dm_control_space.shape, (
            (rlpyt_box.shape, dm_control_space.shape))
        return rlpyt_box
    elif isinstance(dm_control_space, Array):
        if isinstance(dm_control_space, BoundedArray):
            raise ValueError("The order of the if-statements matters.")
        return Box(
            low=-float("inf"),
            high=float("inf"),
            shape=dm_control_space.shape,
            dtype=dm_control_space.dtype)
    elif isinstance(dm_control_space, OrderedDict):
        global State
        if State is None:
            State = namedtuple('State', list(dm_control_space.keys()))
        return Composite([convert_dm_control_to_rlpyt_space(value)
                          for value in dm_control_space.values()], State)
    else:
        raise ValueError(dm_control_space)

EnvInfo = None
Observation = None

def init_namedtuples(info_keys=None, state_keys=None):
    global EnvInfo, Observation, State

    if info_keys is None:
        info_keys = ['traj_done']

    if state_keys is None:
        state_keys = ['pixels']

    EnvInfo = namedtuple('EnvInfo', info_keys)
    Observation = namedarraytuple('Observation', state_keys)
    State = namedtuple('State', state_keys)

class DMControlEnv(Env):

    def __init__(self,
                 domain,
                 task,
                 frame_skip=1,
                 normalize=False,
                 pixel_wrapper_kwargs=None,
                 task_kwargs={},
                 environment_kwargs={},
                 max_path_length=1200,
                 ):
        save__init__args(locals(), underscore=True)

        env = suite.load(domain_name=domain,
                         task_name=task,
                         task_kwargs=task_kwargs,
                         environment_kwargs=environment_kwargs)
        if normalize:
            np.testing.assert_equal(env.action_spec().minimum, -1)
            np.testing.assert_equal(env.action_spec().maximum, 1)
        if pixel_wrapper_kwargs is not None:
            env = pixels.Wrapper(env, **pixel_wrapper_kwargs)
        self._env = env

        self._observation_keys = tuple(env.observation_spec().keys())
        observation_space = convert_dm_control_to_rlpyt_space(
            env.observation_spec())
        self._observation_space = observation_space

        action_space = convert_dm_control_to_rlpyt_space(env.action_spec())
        if len(action_space.shape) > 1:
            raise NotImplementedError(
                "Shape of the action space ({}) is not flat, make sure to"
                " check the implemenation.".format(action_space))
        self._action_space = action_space

        self._step_count = 0

    def reset(self):
        self._step_count = 0
        time_step = self._env.reset()
        observation = self._filter_observation(time_step.observation)

        global Observation
        if Observation is None:
            Observation = namedarraytuple("Observation", list(observation.keys()))
        observation = Observation(**{k: v for k, v in observation.items()
                                     if k in self._observation_keys})
        return observation

    def step(self, action):
        time_step = self._env.step(action)
        reward = time_step.reward
        terminal = time_step.last()
        info = time_step.info
        info.update({
            key: value
            for key, value in time_step.observation.items()
            if key not in self._observation_keys
        })
        observation = self._filter_observation(time_step.observation)

        self._step_count += 1
        info['traj_done'] = self._step_count >= self._max_path_length

        global EnvInfo
        if EnvInfo is None:
            EnvInfo = namedtuple("EnvInfo", list(info.keys()))
        info = EnvInfo(**{k: v for k, v in info.items() if k in EnvInfo._fields})

        global Observation
        if Observation is None:
            Observation = namedarraytuple("Observation", list(observation.keys()))
        observation = Observation(**{k: v.copy() for k, v in observation.items()
                                     if k in self._observation_keys})

        return EnvStep(observation, reward, terminal, info)

    def render(self, *args, mode='rgb_array', width=256, height=256,
               cameria_id=0, **kwargs):
        if mode == 'human':
            raise NotImplementedError(
                "TODO(Alacarter): Figure out how to not continuously launch"
                " viewers if one is already open."
                " See: https://github.com/deepmind/dm_control/issues/39.")
        elif mode == 'rgb_array':
            return self._env.physics.render(width=width, height=height,
                                            camera_id=cameria_id, **kwargs)
        raise NotImplementedError(mode)

    def get_obs(self):
        obs = self._env.task.get_observation(self._env.physics)
        obs['pixels'] = self._env.physics.render(**self._env._render_kwargs)
        obs = self._filter_observation(obs)
        obs = Observation(**{k: v for k, v in obs.items()
                             if k in self._observation_keys})
        return obs

    def get_state(self, ignore_step=True):
        if ignore_step:
            return self._env.physics.get_state()
        return self._env.physics.get_state(), self._step_count

    def set_state(self, state, ignore_step=True):
        if ignore_step:
            self._env.physics.set_state(state)
            self._env.step(np.zeros(self.action_space.shape))
        else:
            self._env.physics.set_state(state[0])
            self._env.step(np.zeros(self.action_space.shape))
            self._step_count = state[1]

    def get_geoms(self):
        return self._env.task.get_geoms(self._env.physics)

    @property
    def spaces(self):
        return EnvSpaces(
            observation=self._observation_space,
            action=self._action_space,
        )

    ###########################################################################
    # Helpers

    def _filter_observation(self, observation):
        observation = type(observation)([
            (name, value)
            for name, value in observation.items()
            if name in self._observation_keys
        ])
        return observation

    ###########################################################################
    # Properties
