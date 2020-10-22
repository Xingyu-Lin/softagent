import h5py
import numpy as np
import os.path as osp
import os

def _load_data_file(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()

    return data

def _store_data(data_names, data, path):
    hf = h5py.File(path, 'w')
    for i in range(len(data_names)):
        hf.create_dataset(data_names[i], data=data[i])
    hf.close()

def transform_data(data):
    all_positions, all_velocities, _, scene_params, picked_points, single_action = data
    _, cloth_x_dim, cloth_y_dim, _ = scene_params
    particle_num = int(cloth_x_dim * cloth_y_dim)
    assert len(all_positions) == particle_num + 1
    assert len(all_velocities) == particle_num + 1
    positions = all_positions[:particle_num]
    velocities = all_velocities[:particle_num]
    picked_point_positions = 0
    picker_position = np.ones((2, 3)) * -1
    picker_position[0] = all_positions[-1]
    shape_positions = picker_position.copy()
    action = np.zeros((2, 4))
    action[0] = single_action
    # print(picked_points)
    new_picked_points = [int(picked_points), int(-1)]
    store_data = [
        positions, velocities, new_picked_points, picked_point_positions, picker_position, action, scene_params,
        shape_positions
    ]

    return store_data

load_data_names = [
    "positions", "velocities", "clusters", "scene_params", "picked", "action"
]

store_data_names = [
    'positions', 'velocities', 'picked_points', 'picked_point_positions', 
    'picker_position', 'action', 'scene_params', 'shape_positions'
]

train_path_load = 'datasets/ClothFlatten_xingyu/ClothFlatten/train'
valid_path_load = 'datasets/ClothFlatten_xingyu/ClothFlatten/valid'

train_path_store = 'datasets/ClothFlatten_xingyu2/train'
valid_path_store = 'datasets/ClothFlatten_xingyu2/valid'

for traj_idx in range(450):
    print("Load train traj {}".format(traj_idx))
    os.makedirs(osp.join("datasets/ClothFlatten_xingyu2/train", str(traj_idx)))
    for t in range(100):
        data = _load_data_file(load_data_names, osp.join(train_path_load, str(traj_idx), str(t) + '.h5'))
        store_data = transform_data(data)
        _store_data(store_data_names, store_data, osp.join(
            train_path_store, str(traj_idx), str(t) + '.h5'
        ))

for traj_idx in range(50):
    print("Load valid traj {}".format(traj_idx))
    os.makedirs(osp.join("datasets/ClothFlatten_xingyu2/valid", str(traj_idx)))
    for t in range(100):
        data = _load_data_file(load_data_names, osp.join(valid_path_load, str(traj_idx), str(t) + '.h5'))
        store_data = transform_data(data)
        _store_data(store_data_names, store_data, osp.join(
            valid_path_store, str(traj_idx), str(t) + '.h5'
        ))

