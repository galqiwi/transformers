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
from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name
from ..utils import is_torch_available


if is_torch_available():
    import torch
    import torch.nn as nn


class TinyQuantQuantize(ConversionOps):
    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list["torch.Tensor"]],
        model: "torch.nn.Module | None" = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        from tinyquant.quantized_linear import QuantizedLinear
        from tinyquant.quantizer import quantize

        weight = None
        bias = None
        weight_key = None
        bias_key = None

        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            if key.endswith("weight"):
                weight = tensor
                weight_key = key
            elif key.endswith("bias"):
                bias = tensor
                bias_key = key

        if weight is None:
            raise ValueError(f"No weight tensor found in input_dict for layer {full_layer_name}")

        module_path = full_layer_name.rsplit(".", 1)[0] if "." in full_layer_name else full_layer_name

        quantized_layer = quantize(
            self.hf_quantizer.quantization_config.tinyquant_method,
            weight=weight,
            bias=bias,
            **self.hf_quantizer.quantization_config.kwargs,
        )

        parent_path, module_name = module_path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        setattr(parent, module_name, quantized_layer)

        quantized_layer._is_hf_initialized = True

        if missing_keys is not None:
            if weight_key:
                missing_keys.discard(weight_key)
            if bias_key:
                missing_keys.discard(bias_key)

        return {}


class TinyQuantDeserialize(ConversionOps):
    apply_dtype_casting = False

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list["torch.Tensor"]],
        model: "torch.nn.Module | None" = None,
        full_layer_name: str | None = None,
        missing_keys=None,
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        from tinyquant.quantized_linear import QuantizedLinear

        parts = full_layer_name.split(".tq_tensors.")
        if len(parts) != 2:
            raise ValueError(f"Unexpected layer name format: {full_layer_name}")

        module_path = parts[0]
        tensor_key = parts[1]

        module = model.get_submodule(module_path)
        if not isinstance(module, QuantizedLinear):
            raise ValueError(f"Expected QuantizedLinear at {module_path}, got {type(module)}")

        tensor = list(input_dict.values())[0]
        if isinstance(tensor, list):
            tensor = tensor[0]

        module.tq_tensors[tensor_key] = nn.Parameter(tensor, requires_grad=False)

        for prop in ['meta', 'quantization_method', 'in_features', 'out_features', 'shape']:
            if prop in module.__dict__:
                del module.__dict__[prop]

        module._is_hf_initialized = True

        if missing_keys is not None:
            missing_keys.discard(full_layer_name)

        return {}
