# TRAINING
DATASET_RGB = "/media/toanhoang/Workspace/Projects/PolyShape/INRIA/AerialImageDataset/rgb/*.tif"
DATASET_GTI = "/media/toanhoang/Workspace/Projects/PolyShape/INRIA/AerialImageDataset/gti/*.tif"
DATASET_SEG = "/media/toanhoang/Workspace/Projects/PolyShape/INRIA/AerialImageDataset/seg/*.tif"

DEBUG_DIR = "../debug/"


# INFERENCE
INF_RGB = "../data/test/single_polygon/rgb/*.tif"
INF_SEG = "../data/test/single_polygon/seg/*.tif"
INF_OUT = "../data/test/single_polygon/reg_output/"

MODEL_ENCODER = "../checkpoints/230418_ckpts/E140000_e1"
MODEL_GENERATOR = "../checkpoints/230418_ckpts/E140000_net"
