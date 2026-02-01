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
from typing import TYPE_CHECKING, Any, Optional
import fnmatch
from torch import nn

if TYPE_CHECKING:
    from ..modeling_utils import PreTrainedModel

from tinyquant.quantized_linear import QuantizedLinear
from tinyquant.quantizer import quantize

from .base import HfQuantizer

from ..utils import is_torch_available
from ..utils.quantization_config import QuantizationConfigMixin


if is_torch_available():
    import torch


class TinyQuantHfQuantizer(HfQuantizer):

    requires_calibration = False
    requires_parameters_quantization = True

    def __init__(self, quantization_config: QuantizationConfigMixin, **kwargs):
        super().__init__(quantization_config, **kwargs)
        self.quantization_config = quantization_config

    def validate_environment(self, device_map, **kwargs):
        pass

    def _process_model_before_weight_loading(
            self,
            model: "PreTrainedModel",
            **kwargs,
    ):
        if not self.pre_quantized:
            return

        linear_paths = []

        for module_path, module in model.named_modules():
            if not self._should_quantize_layer(model, module_path):
                continue
            linear_paths.append(module_path)

        for linear_path in linear_paths:
            parent_path, linear_name = linear_path.rsplit('.', 1)

            parent = model.get_submodule(parent_path)
            setattr(parent, linear_name, QuantizedLinear.empty())

    def create_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        target_device: "torch.device",
        state_dict: dict[str, Any],
        unexpected_keys: Optional[list[str]] = None,
    ):
        if self.pre_quantized:
            self._load_param_into_quantized_linear(model, param_value, param_name)
        else:
            self._quantize_weight_into_quantized_linear(model, param_value, param_name)

    def _quantize_weight_into_quantized_linear(self, model: "PreTrainedModel", param_value: "torch.Tensor", param_name: str):
        module_path, tensor_name = param_name.rsplit(".", 1)

        parent_path, module_name = module_path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)

        assert tensor_name in ["weight", "bias"]

        parent_path, module_name = module_path.rsplit(".", 1)

        current_module = getattr(parent, module_name)

        if tensor_name == "bias":
            if isinstance(current_module, QuantizedLinear):
                current_module.tq_tensors["bias"] = nn.Parameter(param_value, requires_grad=False)
            else:
                current_module.bias = nn.Parameter(param_value, requires_grad=False)
        
        else:
            existing_bias = None
            if isinstance(current_module, torch.nn.Linear) and current_module.bias is not None:
                existing_bias = current_module.bias.data

            quantized_layer = quantize(
                self.quantization_config.tinyquant_method, 
                weight=param_value, 
                bias=existing_bias,
                **self.quantization_config.kwargs
            )
            setattr(parent, module_name, quantized_layer)

    def _load_param_into_quantized_linear(self, model: "PreTrainedModel", param_value: "torch.Tensor", param_name: str):
        parent_path, _, key = param_name.rsplit(".", 2)
        parent = model.get_submodule(parent_path)
        assert isinstance(parent, QuantizedLinear)
        parent.weights_dict[key] = torch.nn.Parameter(param_value, requires_grad=False)

    def update_expected_keys(self, model, expected_keys: list[str], loaded_keys: list[str]) -> list[str]:

        if not self.pre_quantized:
            return expected_keys

        for module_path, module in model.named_modules():
            if not self._should_quantize_layer(model, module_path):
                continue
            assert isinstance(module, QuantizedLinear)
            module_keys = [key.rsplit('.')[-1] for key in loaded_keys if key.startswith(module_path)]
            for key in module_keys:
                module.weights_dict[key] = torch.nn.Parameter(torch.empty([]), requires_grad=False)

        return loaded_keys

    @property
    def is_trainable(self) -> bool:
        return True

    def is_serializable(self, safe_serialization=None):
        return True

    def check_quantized_param(
        self,
        model: "PreTrainedModel",
        param_value: "torch.Tensor",
        param_name: str,
        state_dict: dict[str, Any],
        **kwargs,
    ) -> bool:
        for module_path, module in model.named_modules():
            if not self._should_quantize_layer(model, module_path):
                continue
            if param_name.startswith(module_path):
                return True
        return False

    def _should_quantize_layer(self, model: "PreTrainedModel", module_path: str):

        module = model.get_submodule(module_path)

        if isinstance(module, QuantizedLinear):
            return True

        if isinstance(module, torch.nn.Linear):
            return fnmatch.fnmatch(module_path, self.quantization_config.layers)

        return False

    def _process_model_after_weight_loading(self, model, **kwargs):
        pass
