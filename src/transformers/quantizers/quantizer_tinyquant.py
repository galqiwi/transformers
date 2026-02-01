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

from ..utils import is_torch_available, logging


if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

if is_torch_available():
    import torch
    from torch import nn

    from ..core_model_loading import WeightConverter

logger = logging.get_logger(__name__)


class TinyQuantHfQuantizer(HfQuantizer):
    requires_calibration = False

    def __init__(self, quantization_config, **kwargs):
        super().__init__(quantization_config, **kwargs)

    def validate_environment(self, *args, **kwargs):
        try:
            import tinyquant  # noqa: F401
        except ImportError:
            raise ImportError(
                "Using TinyQuant quantization requires the tinyquant library. "
                "Please install it with: pip install tinyquant"
            )

    def param_needs_quantization(self, model: "PreTrainedModel", param_name: str, **kwargs) -> bool:
        if not param_name.endswith(".weight"):
            return False
        module_path = param_name.rsplit(".weight", 1)[0]
        if not fnmatch.fnmatch(module_path, self.quantization_config.layers):
            return False
        try:
            module = model.get_submodule(module_path)
            return isinstance(module, nn.Linear)
        except AttributeError:
            return False

    def _process_model_before_weight_loading(
        self,
        model: "PreTrainedModel",
        checkpoint_files=None,
        **kwargs,
    ):
        if not self.pre_quantized:
            return

        from tinyquant.quantized_linear import QuantizedLinear

        checkpoint_tensor_info: dict[str, torch.dtype] = {}
        if checkpoint_files:
            import safetensors
            for checkpoint_file in checkpoint_files:
                try:
                    with safetensors.safe_open(checkpoint_file, framework="pt") as f:
                        for key in f.keys():
                            tensor = f.get_tensor(key)
                            checkpoint_tensor_info[key] = tensor.dtype
                except Exception:
                    state_dict = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
                    for key, tensor in state_dict.items():
                        checkpoint_tensor_info[key] = tensor.dtype

        linear_paths = []

        for module_path, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not fnmatch.fnmatch(module_path, self.quantization_config.layers):
                continue
            linear_paths.append(module_path)

        for linear_path in linear_paths:
            parent_path, linear_name = linear_path.rsplit(".", 1)
            parent = model.get_submodule(parent_path)
            empty_module = QuantizedLinear.empty()

            prefix = f"{linear_path}.tq_tensors."
            for key, dtype in checkpoint_tensor_info.items():
                if key.startswith(prefix):
                    tensor_key = key[len(prefix):]
                    empty_module.tq_tensors[tensor_key] = nn.Parameter(
                        torch.empty([], device="meta", dtype=dtype), requires_grad=False
                    )

            setattr(parent, linear_name, empty_module)

    def _process_model_after_weight_loading(self, model: "PreTrainedModel", **kwargs):
        return model

    def is_serializable(self):
        return True

    @property
    def is_trainable(self) -> bool:
        return True

    @property
    def is_compileable(self) -> bool:
        return True

    def _dequantize(self, model, dtype=None):
        raise NotImplementedError(
            "TinyQuant does not support dequantization. "
            "Please load the original non-quantized model instead."
        )

    def get_quantize_ops(self):
        from ..integrations.tinyquant import TinyQuantQuantize

        return TinyQuantQuantize(self)

    def get_weight_conversions(self):
        from ..integrations.tinyquant import TinyQuantDeserialize

        if self.pre_quantized:
            return [
                WeightConverter(
                    source_patterns=[r"(.*\.tq_tensors\..*)"],
                    target_patterns=[r"\1"],
                    operations=[TinyQuantDeserialize(self)],
                )
            ]
        return []
