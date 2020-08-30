import numpy as np


def pixel_coord_np(width, height):
    """
    Pixel in homogenous coordinate
    Returns:
        Pixel coordinate:       [3, width * height]
    """
    x = np.linspace(0, width - 1, width).astype(np.int)
    y = np.linspace(0, height - 1, height).astype(np.int)
    [x, y] = np.meshgrid(x, y)
    return np.vstack((x.flatten(), y.flatten(), np.ones_like(x.flatten())))


def intrinsic_from_fov(height, width, fov=90):
    """
    Basic Pinhole Camera Model
    intrinsic params from fov and sensor width and height in pixels
    Returns:
        K:      [4, 4]
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])

def get_rotation_matrix(angle, axis):
	axis = axis / np.linalg.norm(axis)
	s = np.sin(angle)
	c = np.cos(angle)

	m = np.zeros((4, 4))

	m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
	# m[0][1] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
	m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
	# m[0][2] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
	m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
	m[0][3] = 0.0

	# m[1][0] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
	m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
	m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
	# m[1][2] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
	m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
	m[1][3] = 0.0

	# m[2][0] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
	m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
	# m[2][1] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
	m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
	m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
	m[2][3] = 0.0

	m[3][0] = 0.0
	m[3][1] = 0.0
	m[3][2] = 0.0
	m[3][3] = 1.0

	return m