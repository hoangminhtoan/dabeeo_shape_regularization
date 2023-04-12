# TRAINING
DATASET_RGB = "./data/rgb/*.tif"
DATASET_GTI = "./data/gti/*.tif"
DATASET_SEG = "./data/seg/*.tif"

DEBUG_DIR = "./debug/"


# INFERENCE
INF_RGB = "./data/test/rgb/*.tif"
INF_SEG = "./data/test/seg/*.tif"
INF_OUT = "./data/test/reg_output/"

MODEL_ENCODER = "./checkpoints/saved_models_gan/E140000_e1"
MODEL_GENERATOR = "./saved_models_gan/E140000_net"
