import os.path as osp
import os

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

AWS_NETWORK_INTERFACES = []

AWS_BUCKET_REGION_NAME = 'us-east-2'

MUJOCO_KEY_PATH = osp.expanduser("~/.mujoco")

USE_GPU = True

USE_TF = True

AWS_REGION_NAME = "us-east-2"

if USE_GPU:
    DOCKER_IMAGE = "dementrock/rllab3-shared-gpu"
else:
    DOCKER_IMAGE = "dementrock/rllab3-shared"

DOCKER_LOG_DIR = "/tmp/expt"

CODE_DIR = "/root/code/"

AWS_S3_PATH = "s3://chester-softgym/rllab/experiments"

EBS_OPTIMIZED = True

AWS_EXTRA_CONFIGS = dict()

AWS_CODE_SYNC_S3_PATH = "s3://chester-softgym/rllab/code"

ALL_REGION_AWS_IMAGE_IDS = {
    # "ap-northeast-1": "ami-002f0167",
    # "ap-northeast-2": "ami-590bd937",
    # "ap-south-1": "ami-77314318",
    # "ap-southeast-1": "ami-1610a975",
    # "ap-southeast-2": "ami-9dd4ddfe",
    # "eu-central-1": "ami-63af720c",
    # "eu-west-1": "ami-41484f27",
    # "sa-east-1": "ami-b7234edb",
    "us-east-1": "ami-83f26195",
    "us-east-2": "ami-0ec385d5f98faacc3",   #"ami-0e63a1a8842443350",
    "us-west-1": "ami-576f4b37",
    "us-west-2": "ami-b8b62bd8"
}

AWS_IMAGE_ID = ALL_REGION_AWS_IMAGE_IDS[AWS_REGION_NAME]

if USE_GPU:
    AWS_INSTANCE_TYPE = "p2.xlarge"
else:
    AWS_INSTANCE_TYPE = "c4.4xlarge"

ALL_REGION_AWS_KEY_NAMES = {
    "us-east-1": "rllab-us-east-1",
    "us-east-2": "rllab-us-east-2",
    "us-west-1": "rllab-us-west-1",
    "us-west-2": "rllab-us-west-2"
}

AWS_KEY_NAME = ALL_REGION_AWS_KEY_NAMES[AWS_REGION_NAME]

AWS_SPOT = True

AWS_SPOT_PRICE = '2.0'

AWS_ACCESS_KEY = os.environ.get("AWS_ACCESS_KEY", None)

AWS_ACCESS_SECRET = os.environ.get("AWS_ACCESS_SECRET", None)

AWS_IAM_INSTANCE_PROFILE_NAME = "rllab"

AWS_SECURITY_GROUPS = ["rllab-sg"]

ALL_REGION_AWS_SECURITY_GROUP_IDS = {
    "us-east-1": [
        "sg-e17bfc95"
    ],
    "us-east-2": [
        "sg-1ddb3876"
    ],
    "us-west-1": [
        "sg-cd5f9db4"
    ],
    "us-west-2": [
        "sg-b585a8c9"
    ]
}

AWS_SECURITY_GROUP_IDS = ALL_REGION_AWS_SECURITY_GROUP_IDS[AWS_REGION_NAME]

FAST_CODE_SYNC_IGNORES = [
    ".git",
    "data/autobot",
    "data/corl_s3_data",
    "data/videos",
    "data/open_loop_videos",
    "data/icml",
    "data/local",
    "data/seuss",
    "data/yufei_s3_data",
    "data/icml"
    "data/local",
    "data/archive",
    "data/debug",
    "data/s3",
    "data/video",
    ".idea",
    "tests",
    "examples",
    "docs",
    ".idea",
    ".DS_Store",
    ".ipynb_checkpoints",
    "blackbox",
    "blackbox.zip",
    "*.pyc",
    "*.ipynb",
    "scratch-notebooks",
    "conopt_root",
    "private/key_pairs",
    "DPI-Net",
    "imgs/",
    "imgs",
    "videos"
]

FAST_CODE_SYNC = True

LABEL = ""
