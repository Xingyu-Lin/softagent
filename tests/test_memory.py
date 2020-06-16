import numpy as np
import time
import numba as nb
from numba import jit
import torch
import cProfile, pstats, io
# from rlkit.torch.core import np_to_pytorch_batch
# from rlkit.torch import pytorch_util as ptu

N = 80000
D = 49512
batch_size = 256

buffer = np.ones([N, D], dtype=np.uint8)
print('finish creating buffer')


def postprocess_obs(obs):
    return (obs.astype(np.float32) / 256. - 0.5) + np.random.random(obs.shape) / 256.


def postprocess_obs_numba(obs):
    return (obs.astype(np.float32) / 256. - 0.5) + np.random.random(obs.shape) / 256.


def postprocess_obs_cuda(obs):
    obs.div_(256.).sub_(0.5)  # Quantise to given bit depth and centre
    obs.add_(torch.rand_like(obs).div_(256))  # Dequantise (to approx. match likelihood of PDF of continuous images vs. PMF of discrete images)


@nb.njit
def sample_batch_numba(buffer, batch_size, batch):
    indices = np.random.randint(0, N, batch_size)
    for i in range(batch_size):
        batch[i] = buffer[indices[i]]
    return batch


def sample_batch(buffer, batch_size):
    indices = np.random.randint(0, buffer.shape[0], batch_size)
    batch = buffer[indices]
    return batch


def sample_batch_numba_post(buffer, batch_size):
    indices = np.random.randint(0, buffer.shape[0], batch_size)
    batch = buffer[indices]
    return postprocess_obs_numba(batch)


def sample_batch_post(buffer, batch_size):
    indices = np.random.randint(0, buffer.shape[0], batch_size)
    batch = buffer[indices]
    return postprocess_obs(batch)


def sample_batch_cuda(buffer, batch_size):
    indices = np.random.randint(0, buffer.shape[0], batch_size)
    batch = buffer[indices]

    prev_time = time.time()
    batch = torch.from_numpy(batch).to('cuda:0').float()
    transfer_time = time.time() - prev_time

    return postprocess_obs_cuda(batch), transfer_time


def compare_numba():
    batch = np.empty([batch_size, D], dtype=np.int32)
    sample_batch_numba(buffer, batch_size, batch)
    start_time = time.time()
    for i in range(100):
        sample_batch_numba(buffer, batch_size, batch)
    print('numba time:', time.time() - start_time)

    start_time = time.time()
    for i in range(100):
        sample_batch(buffer, batch_size)
    print('no numba time:', time.time() - start_time)


def sample_batch_chunked(buffers, batch_size):
    idx = np.random.randint(10)
    return sample_batch(buffers[idx], batch_size)


def compare_small_buffer():
    buffers = []
    for i in range(10):
        buffers.append(np.ones([8000, D], dtype=np.int32))

    start_time = time.time()
    for i in range(20):
        sample_batch_chunked(buffers, batch_size)
    print('chunked sample time:', time.time() - start_time)

    start_time = time.time()
    for i in range(20):
        sample_batch(buffer, batch_size)
    print('normal time:', time.time() - start_time)


def compare_postprocessing():
    # batch = np.empty([batch_size, D], dtype=np.float32)

    # start_time = time.time()
    # for i in range(10):
    #     sample_batch_post(buffer, batch_size)
    # print('no numba time:', time.time() - start_time)
    #
    # sample_batch_numba_post(buffer, batch_size)
    # start_time = time.time()
    # for i in range(10):
    #     sample_batch_numba_post(buffer, batch_size)
    # print('numba time:', time.time() - start_time)

    # start_time = time.time()

    all_ = 0
    for i in range(100):
        batch, transfer_time = sample_batch_cuda(buffer, batch_size)
        print(transfer_time)
        all_ += transfer_time
    # print('sample time, transfer time:', time.time() - start_time - all_, all_ / 10)


if __name__ == '__main__':
    # compare_small_buffer()
    # compare_numba()
    batch, transfer_time = sample_batch_cuda(buffer, batch_size)

    pr = cProfile.Profile()

    pr.enable()
    compare_postprocessing()

    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('time')
    ps.print_stats(20)
    print(s.getvalue())