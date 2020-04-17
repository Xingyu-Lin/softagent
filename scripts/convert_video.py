import ffmpy
import argparse, sys
import os

parser = argparse.ArgumentParser(sys.argv[0])
parser.add_argument('--dir', type = str, default = './data/video/')
args = parser.parse_args()

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(args.dir) if isfile(join(args.dir, f))]

only_mp4 = []
for f in onlyfiles:
    video = f[:-4]
    # if video + '.gif' in onlyfiles:
    #     continue
    if '.mp4' in f:
        only_mp4.append(f)

for f in only_mp4:
    if 'Drop' in f:
        r = 2
    else:
        r = 10

    factor = 30 / r
    ff = ffmpy.FFmpeg(
        inputs = {args.dir + f : None},
        outputs = {args.dir + f[:-4] + '.gif' : '-vf "setpts={}*PTS" -r {}'.format(factor, r)})
    
    ff.run()