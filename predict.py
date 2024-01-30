from utils.half_circle import HalfCircle
from config.configuration import *
from os.path import join

predict = HalfCircle(join(HALF_CIRCLE_MODEL_CONFIG.TEST_IMAGE_DIRECTORY, 'testhc_10.jpg'), HALF_CIRCLE_MODEL_CONFIG.MAX_VALUE,)
print(predict.predicted_value)