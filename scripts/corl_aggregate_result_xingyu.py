import sys
import os
import argparse
import subprocess

sys.path.append('.')
from chester import config


def aws_sync(bucket_name, s3_log_dir, target_dir, args):
    cmd = 'aws s3 cp --recursive s3://%s/%s %s' % (bucket_name, s3_log_dir, target_dir)
    # exlus = ['"*.pkl"', '"*.gif"', '"*.png"', '"*.pth"']
    # inclus = []
    # if args.gif:
    #     exlus.remove('"*.gif"')
    # if args.png:
    #     exlus.remove('"*.png"')
    # if args.param:
    #     inclus.append('"params.pkl"')
    #     exlus.remove('"*.pkl"')

    # if not args.include_all:
    #     for exc in exlus:
    #         cmd += ' --exclude ' + exc
    #
    #     for inc in inclus:
    #         cmd += ' --include ' + inc
    # exit()
    subprocess.call(cmd, shell=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    args = parser.parse_args()

    data_paths = [
        # CEM
        ('autobot', '0717_corl_cem_cloth_rope'),  # CEM on ropeflatten,  cloth drop rigid x 20,
        ('autobot', '0719_corl_cem_cloth_fold'),  # CEM Cloth fold x 10
        ('seuss', '0715_corl_cem_cloth_flatten_cloth_drop'),  # CEM, cloth flatten x 10, cloth drop x 10
        ('autobot', '0720_rigid_cloth_fold_cem'),  # CEM Rigid cloth fold x 10

        # PlaNet
        ('seuss', '0716_planet_cloth'),  # PlaNet, cloth fold, cloth flatten x 6
        ('seuss', '0719_planet_cloth_fold'),  # Cloth fold x 3
        ('seuss', '0719_planet_clothdrop'),  # cloth drop x 3
        ('autobot', '0717_planet_rigid_cloth'),  # PlaNet rigid cloth drop x 4
        ('seuss', '0722_planet_rigid_cloth_fold'),  # Rigid cloth fold (simplified version) x 3

        # CURL
        ('seuss', '0717_cloth_flatten'),  # Curl, cloth flatten, both state and RGB, 5 seeds. x 10
        ('seuss', '0719_corl_cloth_flatten'),  # Cloth flatten, RGB, 3 seeds
        ('autobot', '0719_corl_cloth_fold_lr'),  # Cloth fold x 6 (alpha)
        ('autobot', '0722_rigid_cloth_fold'),  # Rigid cloth fold x 6 (alpha)
        ('ec2', '0717-corl-cloth-drop'),  # Curl, cloth drop and rigid cloth drop x 12
        ('autobot', '0718_corl_rope_curl_lr'),  # Curl, rope, 5 seeds, x 10
        ('seuss', '0719_corl_curl_rope'),  # rope with RGB x 5 seeds

        # PointCloud
        ('autobot', '0722_point_cloud'),
    ]
    # Arxived env:
    # Saved cloth fold
    # ('seuss', '0717_cloth_fold'),  # Curl, cloth fold, rigid cloth fold, both state and RGB x 12
    # ('seuss', '0716_planet_cloth_fold_short') # Planet, saved cloth fold with horizon of 50
    # ('autobot', '0717_corl_cem_cloth_rope_short') # Saved CEM of fold cloth and rigid fold cloth

    local_dir = './data/corl_data/'
    s3_bucket = 'chester-softgym'
    for (mode, data_p) in data_paths:
        local_dir = local_dir.rstrip('/')
        # if args.local_dir.rfind('/') != -1:
        #     local_dir = os.path.join('./data', 'seuss', args.local_dir[:args.local_dir.rfind('/')])
        # else:
        #     local_dir = os.path.join('./data', 'seuss', args.local_dir)
        if mode in ['autobot', 'seuss']:
            remote_data_dir = os.path.join('/home/xlin3/Projects/softagent', 'data', 'local', data_p)
            command = """rsync -avzh --delete --progress {mode}:{remote_data_dir} {local_dir}""".format(
                mode=mode,
                remote_data_dir=remote_data_dir,
                local_dir=local_dir
            )

            if args.bare:
                command += """ --exclude '*.pkl' --exclude '*.pth' --exclude '*.out' --exclude '*.pt' --exclude '*.gif' --exclude '*.png' --include '*.csv' --include '*.json' --delete"""
            print(command)
            os.system(command)
        elif mode == 'ec2':
            s3_log_dir = "rllab/experiments/" + data_p
            local_dir = os.path.join(local_dir, data_p)
            if not os.path.exists(local_dir):
                os.makedirs(local_dir)
            aws_sync(s3_bucket, s3_log_dir, local_dir, args)
