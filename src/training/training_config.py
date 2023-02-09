from typing import Tuple

# model architecture params
NUM_CLASSES:int = 8
NUM_REGRESSION_NEURONS:int = 2
IMAGE_RESOLUTION:Tuple[int, int] = (224, 224)

# training metaparams
NUM_EPOCHS:int = 100
BATCH_SIZE:int = 128
OPTIMIZER:str = "Adam"
AUGMENT_PROB:float = 0.05
EARLY_STOPPING_PATIENCE:int = 15

# scheduller
LR_SCHEDULLER:str = "Cyclic"
ANNEALING_PERIOD:int = 6
LR_MAX_CYCLIC:float = 0.005
LR_MIN_CYCLIC:float = 0.001
LR_MIN_WARMUP:float = 0.0001
WARMUP_STEPS:int = 200
WARMUP_MODE:str = "linear"

# general params
BEST_MODEL_SAVE_PATH:str = "best_model/"
NUM_WORKERS:int = 8





EMO_CATEGORIES:dict = {
    "N":0,
    "H":1,
    "Sa":2,
    "Su":3,
    "F":4,
    "D":5,
    "A":6,
    "C":7,
}
