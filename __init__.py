from .convert import FP12_MAX, FP12_MIN, FP10_MAX, FP10_MIN
from .convert import to_fp12, fp12_to_fp16
from .convert import to_fp10, fp10_to_fp16
from .nn.mod_fp12 import Linear as LinearFP12
from .nn.mod_fp12 import Conv2d as Conv2dFP12
from .nn.mod_fp10 import Linear as LinearFP10
from .nn.mod_fp10 import Conv2d as Conv2dFP10
