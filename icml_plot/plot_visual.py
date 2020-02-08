import cv2
import numpy as np
import torch, torchvision

envs = ['pour_water', 'pass_water', 'cloth_flatten', 'cloth_fold', 'cloth_drop', 'rope_flatten']

pics = []
for env in envs:
    pic = cv2.imread('data/icml/{}.jpg'.format(env))
    pic = pic.astype(np.float64).transpose(2, 0, 1)
    print(pic.shape)
    pics.append(torch.from_numpy(pic.copy()))

grid_imgs = torchvision.utils.make_grid(pics, padding=0, pad_value=120, nrow=1).data.cpu().numpy().transpose(1, 2, 0)
# grid_imgs=grid_imgs[:, :, ::-1]
save_name = 'data/icml/visual.jpg'
print(save_name)
cv2.imwrite(save_name, grid_imgs)


pics = []
for env in envs:
    pic = cv2.imread('data/icml/{}_goal.jpg'.format(env))
    pic = pic.astype(np.float64).transpose(2, 0, 1)
    print(pic.shape)
    pics.append(torch.from_numpy(pic.copy()))

grid_imgs = torchvision.utils.make_grid(pics, padding=20, pad_value=120, nrow=1).data.cpu().numpy().transpose(1, 2, 0)
# grid_imgs=grid_imgs[:, :, ::-1]
save_name = 'data/icml/visual_goal.jpg'
print(save_name)
cv2.imwrite(save_name, grid_imgs)