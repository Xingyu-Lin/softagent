import sys
import os
import argparse

sys.path.append('.')
from chester import config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry', action='store_true', default=False)
    parser.add_argument('--bare', action='store_true', default=False)
    args = parser.parse_args()

    data_paths = [
        '0713-CoRL-Curl-PourWater',
        '0713-CoRL-CEM-PourWater',
        '0715-CoRL-CEM-PassWater-and-Torus',
        '0715-Curl-PassWater-and-Torus',
        '0717_planet_water',
        '0719_planet_rope',
        '0723-planet-PassWater',
        '0723-Curl-Pourwater-pointcloud',
        '0724-planet-TransportTorus',
        '0724-CoRL-CEM-TransportTorus-2',
        '0724-Curl-transport-torus-2'
    ]
    local_dir = './data/corl_data/'
    for data_p in data_paths:
        args.local_dir = local_dir.rstrip('/')
        local_dir = local_dir.rstrip('/')

        remote_data_dir = os.path.join('/home/yufeiw2/Projects/softagent', 'data', 'local', data_p)
        command = """rsync -avzh --delete --progress seuss:{remote_data_dir} {local_dir}""".format(
            remote_data_dir=remote_data_dir,
            local_dir=local_dir
        )

        if args.bare:
            command += """ --exclude '*.pkl' --exclude '*.pth' --exclude '*.out' --exclude '*.pt' --exclude '*.gif' --exclude '*.png' --include '*.csv' --include '*.json' --delete"""

        print(command)
        os.system(command)
