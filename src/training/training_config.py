NUM_CLASSES:int = 8
NUM_EPOCHS:int = 100
BATCH_SIZE:int = 128
AUGMENT_PROB:float = 0.05
OPTIMIZER:str = "Adam"
LR_MAX:float = 0.005
LR_MIN:float = 0.0001
LR_SCHEDULLER:str = "Cyclic"
ANNEALING_PERIOD:int = 6
BEST_MODEL_SAVE_PATH:str = "best_model/"






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
