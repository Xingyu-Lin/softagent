from rlkit.samplers.rollout_functions import rollout
from rlkit.torch.pytorch_util import set_gpu_mode
import argparse
import torch
import uuid
from rlkit.core import logger

filename = str(uuid.uuid4())


def simulate_policy(args, flex_env):
    data = torch.load(args.file)
    policy = data['evaluation/policy']
    env = data['evaluation/env']
    if flex_env:
        import pyflex
        headless, render, camera_width, camera_height = False, True, 720, 720
        pyflex.init(headless, render, camera_width, camera_height)

    if args.gpu:
        set_gpu_mode(True)
        policy.cuda()
    while True:
        path = rollout(
            env,
            policy,
            max_path_length=args.H,
            render=False,
        )
        if hasattr(env, "log_diagnostics"):
            env.log_diagnostics([path])
        logger.dump_tabular()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='path to the snapshot file')
    parser.add_argument('--H', type=int, default=300,
                        help='Max length of rollout')
    parser.add_argument('--gpu', action='store_false')
    args = parser.parse_args()

    simulate_policy(args, True)
