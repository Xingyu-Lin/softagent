import sys
import os
import argparse

sys.path.append('.')
from chester import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('host', type=str)
    parser.add_argument('folder', type=str)
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    args = parser.parse_args()

    args.folder = args.folder.rstrip('/')
    if args.folder.rfind('/') !=-1:
        local_dir = os.path.join('./data', args.host, args.folder[:args.folder.rfind('/')])
    else:
        local_dir = os.path.join('./data', args.host, args.folder)
    remote_data_dir = os.path.join(config.REMOTE_DIR[args.host], 'data', 'local', args.folder)
    command = """rsync -avzh --delete --progress {host}:{remote_data_dir} {local_dir}""".format(host=args.host,
                                                                                                remote_data_dir=remote_data_dir,
                                                                                                local_dir=local_dir)
    if args.bare:
        command += """ --exclude '*.pkl' --exclude '*.pth' --exclude '*.pt' --include '*.csv' --include '*.json' --delete"""
    if args.dry:
        print(command)
    else:
        os.system(command)
