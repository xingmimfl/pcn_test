import os

def check_dir(dir_t):
    if not os.path.exists(dir_t):
        os.makedirs(dir_t)

DEBUG = True

DATA_PATH = "./24"

TRAIN_OUT_ITER = 1000
NUM_WORKERS = 8
#-----
LEARNING_RATE = 0.001
MOMENTUM = 0.9
WEIGHT_DECAY = 0.004
GAMMA = 0.8
STEP_SIZE = 150000
MAX_ITERS = 1500000
SUFFIX = 'pnet_190220'
BATCH_SIZE = 256
DEVICE_IDS = [4]
#------

TEST_DIR = os.path.join(DATA_PATH, 'face_pics', '002')
SNAPSHOT_PATH = 'model_' + SUFFIX
