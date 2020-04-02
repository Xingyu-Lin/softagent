import gym
from gym.spaces import Box
import numpy as np
import cv2 as cv


# ^ y
# |
# |
# |
# O------> x
class Box1d(gym.Env):
    def __init__(self, horizon=10, canvas_size=500, image_dim=64, box_size_range=(0.01, 0.3), image_observation=False,
                 reward_type='normalized_dist', **kwargs):
        self.horizon = horizon
        self.image_dim = image_dim
        self.canvas_size = canvas_size
        self.box_size_range = box_size_range
        self.box_pos_range = (0, 1)
        self.box_pos = self.box_goal_pos = self.box_size = self.time_step = None
        self.action_space = Box(low=-0.15, high=0.15, shape=(1,), dtype=np.float32)
        self.image_observation = image_observation
        self.reward_type = reward_type  # Reward type should be in {'dist', 'normalized_dist'}
        if image_observation:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(image_dim, image_dim, 2), dtype=np.float32)
        else:
            self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)

    def reset(self):
        self.time_step = 0
        self.box_size = np.random.uniform(*self.box_size_range)
        self.box_pos = np.random.uniform(*self.box_pos_range)
        self.box_goal_pos = np.random.uniform(*self.box_pos_range)
        while abs(self.box_pos - self.box_goal_pos) < 0.01:
            self.box_goal_pos = np.random.uniform(*self.box_pos_range)
        self.box_init_dist = abs(self.box_pos - self.box_goal_pos)
        # self.box_size = 0.3
        # self.box_pos = 0.3
        # self.box_goal_pos = 0.5
        return self._get_current_obs()

    def step(self, action):
        self.box_pos += action
        if self.reward_type == 'dist':
            reward = -abs(self.box_pos - self.box_goal_pos)
        elif self.reward_type == 'normalized_dist':
            reward = -abs(self.box_pos - self.box_goal_pos) / self.box_init_dist
        else:
            raise NotImplementedError
        obs = self._get_current_obs()
        info = self._get_current_info()
        self.time_step += 1
        return obs, reward, self.time_step == self.horizon, info

    def render(self, mode='human'):  # Render to screen
        obs = self._get_current_obs()
        # curr_img, goal_img = obs[:, :, 0].T, obs[:, :, 1].T
        # curr_img, goal_img = cv.resize(obs[:, :, 0].T, (self.canvas_size, self.canvas_size), interpolation=cv.INTER_NEAREST), \
        #                      cv.resize(obs[:, :, 1].T, (self.canvas_size, self.canvas_size), interpolation=cv.INTER_NEAREST)
        # padding = np.ones([curr_img.shape[0], 5]) * 0.5
        # img = np.hstack([curr_img, padding, goal_img])
        if mode == 'human':
            curr_img = self._draw_box(self.box_pos, self.box_size)[:, :, 0].T / 255.
            goal_img = self._draw_box(self.box_goal_pos, self.box_size)[:, :, 0].T / 255.
            padding = np.ones([5, curr_img.shape[1]]) * 0.5
            img = np.vstack([curr_img, padding, goal_img])
            cv.imshow('Box1d', np.vstack([img]))
            cv.waitKey(10)
        else:
            if self.image_observation:
                return np.vstack([obs[:, :, 0].T, obs[:, :, 1].T])
            else:
                curr_img = self._draw_box(self.box_pos, self.box_size)[:, :, 0].T
                goal_img = self._draw_box(self.box_goal_pos, self.box_size)[:, :, 0].T
                curr_img = cv.resize(curr_img, (self.image_dim, self.image_dim), interpolation=cv.INTER_LINEAR)
                goal_img = cv.resize(goal_img, (self.image_dim, self.image_dim), interpolation=cv.INTER_LINEAR)
                return np.vstack([curr_img, goal_img])

    def _to_canvas(self, x):
        return int(np.round(x * self.canvas_size))

    def _draw_box(self, box_pos, box_size):
        box_pos = self._to_canvas(box_pos)
        box_size = self._to_canvas(box_size)

        img = np.zeros(shape=(self.canvas_size, self.canvas_size, 1), dtype=np.uint8)
        minx = max(0, box_pos - box_size // 2)
        maxx = min(self.canvas_size - 1, box_pos + box_size // 2)
        miny = max(0, self.canvas_size // 2 - box_size // 2)
        maxy = min(self.canvas_size - 1, self.canvas_size // 2 + box_size // 2)
        if minx >= maxx or miny >= maxy:
            return img
        img[minx:maxx, miny:maxy, 0] = 255
        return img

    def _get_current_obs(self):
        if self.image_observation:
            curr_img = self._draw_box(self.box_pos, self.box_size)
            goal_img = self._draw_box(self.box_goal_pos, self.box_size)
            curr_img = cv.resize(curr_img, (self.image_dim, self.image_dim), interpolation=cv.INTER_LINEAR)
            goal_img = cv.resize(goal_img, (self.image_dim, self.image_dim), interpolation=cv.INTER_LINEAR)
            obs = np.dstack([curr_img, goal_img])
        else:
            obs = np.array([self.box_pos, self.box_goal_pos], dtype=np.float32)
        return obs

    def _get_current_info(self):
        return {'box_pos': self.box_pos, 'box_goal_pos': self.box_goal_pos}
