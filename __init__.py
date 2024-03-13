from .convert.fp12 import FP12_MAX, FP12_MIN, to_fp12, fp12_to_fp16
from .convert.fp10 import FP10_MAX, FP10_MIN, to_fp10, fp10_to_fp16

from .nn.fp12 import Linear as LinearFP12
from .nn.fp12 import Conv2d as Conv2dFP12
from .nn.fp10 import Linear as LinearFP10
from .nn.fp10 import Conv2d as Conv2dFP10
