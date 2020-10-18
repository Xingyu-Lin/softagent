import time
import torch
import click
import socket
import multiprocessing as mp
from chester.run_exp import run_experiment_lite, VariantGenerator
from GNS.train_new import run_task


@click.command()
@click.argument('mode', type=str, default='local')
@click.option('--debug/--no-debug', default=True)
@click.option('--dry/--no-dry', default=False)
def main(mode, debug, dry):
    def get_relation_dim(use_mesh_edge):
        if use_mesh_edge:
            return 6
        else:
            return 4


    exp_prefix = '1011_GNS_deeper_larger_noise'
    vg = VariantGenerator()
    vg.add('env_name', ['ClothFlatten'])
    vg.add('n_epoch', [10000])
    vg.add('train_rollout', [True])
    vg.add('gen_data', [True])
    vg.add('training', [True])

    vg.add('edge_type', ['eight_neighbor'])
    vg.add('down_sample_scale', [3])
    vg.add('use_mesh_edge', [True])
    vg.add('relation_dim', lambda use_mesh_edge: [get_relation_dim(use_mesh_edge)])
    vg.add('seed', [100])
    vg.add('dt', [0.01]) # this is the actual dt in flex
    vg.add('video_iter_interval', [100000])
    vg.add('video_epoch_interval', [1])
    if debug:
        vg.add('n_rollout', [20])
        # vg.add('log_per_iter', [50])
        vg.add('neighbor_radius', [0.026])
        vg.add('noise_scale', [0.007])
        vg.add('predict_vel', [False])
    else:
        vg.add('neighbor_radius', [0.026, 0.036, 0.16])
        vg.add('noise_scale', [0, 0.003, 0.007])
        vg.add('predict_vel', [True, False])
        vg.add('n_rollout', [500])
        vg.add('log_per_iter', [10000])
        vg.add('nstep_eval_rollout', [10])

    if not debug:
        pass
    else:
        pass
        exp_prefix += '_debug'

    print('Number of configurations: ', len(vg.variants()))
    print("exp_prefix: ", exp_prefix)

    hostname = socket.gethostname()
    gpu_num = torch.cuda.device_count()

    sub_process_popens = []
    for idx, vv in enumerate(vg.variants()):
        while len(sub_process_popens) >= 10:
            sub_process_popens = [x for x in sub_process_popens if x.poll() is None]
            time.sleep(10)
        if mode in ['seuss', 'autobot']:
            if idx == 0:
                compile_script = 'compile_1.0.sh'  # For the first experiment, compile the current softgym
                wait_compile = None
            else:
                compile_script = None
                wait_compile = 120  # Wait 30 seconds for the compilation to finish
        elif mode == 'ec2':
            compile_script = 'compile_1.0.sh'
            wait_compile = None
        else:
            compile_script = wait_compile = None
        if hostname.startswith('autobot') and gpu_num > 0:
            env_var = {'CUDA_VISIBLE_DEVICES': str(idx % gpu_num)}
        else:
            env_var = None
        cur_popen = run_experiment_lite(
            stub_method_call=run_task,
            variant=vv,
            mode=mode,
            dry=dry,
            use_gpu=True,
            exp_prefix=exp_prefix,
            wait_subprocess=debug,
            compile_script=compile_script,
            wait_compile=wait_compile,
            env=env_var
        )
        if cur_popen is not None:
            sub_process_popens.append(cur_popen)
        if debug:
            break


if __name__ == '__main__':
    main()
