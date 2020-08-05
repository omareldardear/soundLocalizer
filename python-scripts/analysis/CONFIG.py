#######################
# TRAINING PARAMETERS #
#######################
BATCH_SIZE = 8
EPOCHS = 20
INIT_LR = 1e-3
TEST_SUBJECTS = [43, 44, 61, 62]
FEATURE = 'melspec'
INPUT_SHAPE = (128, 32 )
NB_CHANNELS = 1
#############################
# PRE-PROCESSING PARAMETERS #
#############################
NUM_BANDS = 32
RESAMPLING_F = 16000
THRESHOLD_VOICE = 60
N_DELAY = 51
DISTANCE_MIC = 0.14
max_tau = DISTANCE_MIC / 343.2

###################
# PATH PARAMETERS #
###################
PATH_DATASET = "/home/jonas/CLionProjects/soundLocalizer/python-scripts/analysis/data/chunck_dataset-1000.csv"
PATH_DATA = '/home/jonas/CLionProjects/soundLocalizer/dataset/dataset-1000'
