import os
import subprocess
import argparse


def aws_sync(bucket_name, s3_log_dir, target_dir, exclude='*.pkl'):
    cmd = 'aws s3 cp --recursive s3://%s/%s %s --exclude %s' % (bucket_name, s3_log_dir, target_dir, exclude)
    print(cmd)
    # exit()
    subprocess.call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('log_dir', type=str, help='S3 Log dir')
    parser.add_argument('-b', '--bucket', type=str, default='chester-softgym', help='S3 Bucket')
    parser.add_argument('-e', '--exclude', type=str, default='*.pkl', help='Exclude')

    args = parser.parse_args()
    s3_log_dir = "rllab/experiments/" + args.log_dir
    local_dir = os.path.join('./data', 'yufei_s3_data', args.log_dir)
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    aws_sync(args.bucket, s3_log_dir, local_dir, exclude=args.exclude)


if __name__ == "__main__":
    main()
