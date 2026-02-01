# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fnmatch
from typing import TYPE_CHECKING

from .base import HfQuantizer
from .quantizers_utils import get_module_from_name

from ..utils import is_torch_available, logging
from ..utils.quantization_config import QuantizationConfigMixin


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

if is_torch_available():
    import torch

    from ..core_model_loading import WeightConverter

logger = logging.get_logger(__name__)


class TinyQuantHfQuantizer(HfQuantizer):

    requires_calibration = False

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        try:
            from tinyquant.quantized_linear import QuantizedLinear  # noqa: F401
            from tinyquant.quantizer import quantize  # noqa: F401
        except ImportError:
            raise ImportError(
                "Using `tinyquant` quantization requires the tinyquant library. "
                "Please install it with: `pip install tinyquant`"
            )

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        **kwargs,
    ):
        if not self.pre_quantized:
            return

        from ..integrations.tinyquant import replace_with_tinyquant_linear

        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model,
            self.quantization_config.modules_to_not_convert,
            model._keep_in_fp32_modules,
        )

        replace_with_tinyquant_linear(
            model,
            quantization_config=self.quantization_config,
            modules_to_not_convert=self.modules_to_not_convert,
        )

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        from tinyquant.quantized_linear import QuantizedLinear

        module, tensor_name = get_module_from_name(model, param_name)

        if isinstance(module, QuantizedLinear):
            return True

        if isinstance(module, torch.nn.Linear) and tensor_name == "weight":
            module_path = param_name.rsplit(".", 1)[0]
            layers_pattern = self.quantization_config.layers or "*"
            return fnmatch.fnmatch(module_path, layers_pattern)

        return False

    @property
    def is_trainable(self) -> bool:
        return True

    def is_serializable(self, safe_serialization=None):
        return True

    def get_quantize_ops(self):
        from ..integrations.tinyquant import TinyQuantQuantize

        return TinyQuantQuantize(self)

    def get_weight_conversions(self):
        from ..integrations.tinyquant import TinyQuantDeserialize

        if self.pre_quantized:
            return [
                WeightConverter(
                    source_patterns=["tq_tensors.*", "weights_dict.*"],
                    target_patterns="",
                    operations=[TinyQuantDeserialize(self)],
                )
            ]
        return []
