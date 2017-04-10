import datetime

today = datetime.datetime.today()

DATA_PATH = '/home/kadomae.13029/work/data2/'

RAW_PATH = DATA_PATH + 'raw/'
TRAIN_PATH = RAW_PATH + 'train.txt'
DEV_PATH = RAW_PATH + 'dev.txt'
TEST_PATH = RAW_PATH + 'test.txt'

GOLD_PATH = DATA_PATH + 'gold'

PICKLE_PATH = DATA_PATH + 'pickle/'
RELEVANT_VECTORS = DATA_PATH + 'vector/'
WORD_VECTOR = DATA_PATH + 'vecf50.txt'
VEC_SIZE = 50

LOG_DIR = DATA_PATH + 'log/' + str(today.year) + \
          str(today.month) + \
          str(today.day) + \
          str(today.hour) + \
          str(today.minute) + \
          str(today.second)