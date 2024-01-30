from os import getcwd
import sys
sys.path.append(getcwd())
from dataclasses import dataclass
from config.libaries import *
from os.path import join
from typing import Any

@dataclass(frozen=True)
class GENERAL_CONFIG:
    SOURCE_FOLDER: str = "src/"
    UTILITY_FOLDERL: str = "utils/"
    IMAGES_FOLDER: str = "images/"
    CONFIDENCE: float = 0.1


@dataclass(frozen=True)
class FULL_CIRCLE_MODEL_CONFIG:
    MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/full-circle/full-circle-max-mix-detection.pt'))
    NEEDLE_MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/full-circle/full-circle-needle-detection.pt'))
    WHOLE_GUAGE_MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/full-circle/full-circle-whole-guage-detection.pt'))
    TEST_IMAGE_DIRECTORY: str = GENERAL_CONFIG.SOURCE_FOLDER + GENERAL_CONFIG.IMAGES_FOLDER + "full_circle"
    MIN_VALUE: float = 0
    MAX_VALUE: float = 40

@dataclass(frozen=True)
class ONE_QUADANT_MODEL_CONFIG:
    MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/one-quadant/one-quadant-max-min-detection.pt'))
    NEEDLE_MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/one-quadant/one-quadant-needle-detection.pt'))
    TEST_IMAGE_DIRECTORY: str = GENERAL_CONFIG.SOURCE_FOLDER + GENERAL_CONFIG.IMAGES_FOLDER + "one_quadant/"
    MIN_VALUE: float = 0
    MAX_VALUE: float = 40

@dataclass(frozen=True)
class HALF_CIRCLE_MODEL_CONFIG:
    MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/half-circle/half-circle-max-min-detection.pt'))
    NEEDLE_MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/half-circle/half-circle-needle-detection.pt'))
    TEST_IMAGE_DIRECTORY: str = GENERAL_CONFIG.SOURCE_FOLDER + GENERAL_CONFIG.IMAGES_FOLDER + "half_circle/"
    MIN_VALUE: float = 0
    MAX_VALUE: float = 100

@dataclass(frozen=True)
class WNR_MODEL_CONFIG:
    MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/white-and-red/wnr-max-min-detection.pt'))
    NEEDLE_MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/white-and-red/wnr-needle-detection.pt'))
    TEST_IMAGE_DIRECTORY: str = GENERAL_CONFIG.SOURCE_FOLDER + GENERAL_CONFIG.IMAGES_FOLDER + "white_and_red/"
    MIN_VALUE: float = 0
    MAX_VALUE: float = 150
    
@dataclass(frozen=True)
class ACW_MODEL_CONFIG:
    MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/anti-clockwise/acw-generalize.pt'))
    TEST_IMAGE_DIRECTORY: str = GENERAL_CONFIG.SOURCE_FOLDER + GENERAL_CONFIG.IMAGES_FOLDER + "anti_clockwise/"
    MIN_VALUE: float = 0
    MAX_VALUE: float = 10

@dataclass(frozen=True)
class TND_MODEL_CONFIG:
    MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/3-needle/3needle-max-min-detection.pt'))
    NEEDLE_MODEL: Any = YOLO(join(GENERAL_CONFIG.SOURCE_FOLDER,'pt_file/3-needle/3needle-needle-detection.pt'))
    TEST_IMAGE_DIRECTORY: str = GENERAL_CONFIG.SOURCE_FOLDER + GENERAL_CONFIG.IMAGES_FOLDER + "three_needle/"
    MIN_VALUE: float = -20
    MAX_VALUE: float = 200


