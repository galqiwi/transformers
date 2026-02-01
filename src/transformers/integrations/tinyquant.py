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

from ..core_model_loading import ConversionOps
from ..quantizers.quantizers_utils import get_module_from_name
from ..utils import is_torch_available, logging


if is_torch_available():
    import torch
    import torch.nn as nn

logger = logging.get_logger(__name__)


class TinyQuantQuantize(ConversionOps):

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list["torch.Tensor"]],
        full_layer_name: str | None = None,
        model: "torch.nn.Module | None" = None,
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        from tinyquant.quantized_linear import QuantizedLinear
        from tinyquant.quantizer import quantize

        value = list(input_dict.values())[0]
        value = value[0] if isinstance(value, list) else value

        module, tensor_name = get_module_from_name(model, full_layer_name)

        if tensor_name == "bias":
            if isinstance(module, QuantizedLinear):
                module.tq_tensors["bias"] = nn.Parameter(value, requires_grad=False)
                return {}
            return {full_layer_name: value}

        if tensor_name == "weight":
            existing_bias = None
            if hasattr(module, "bias") and module.bias is not None:
                existing_bias = module.bias.data
            elif isinstance(module, QuantizedLinear) and "bias" in module.tq_tensors:
                existing_bias = module.tq_tensors["bias"]

            quantized_layer = quantize(
                self.hf_quantizer.quantization_config.tinyquant_method,
                weight=value,
                bias=existing_bias,
                **self.hf_quantizer.quantization_config.kwargs,
            )

            parent_path, module_name = full_layer_name.rsplit(".", 1) if "." in full_layer_name else ("", full_layer_name)
            parent_path = parent_path.rsplit(".", 1)[0] if parent_path else ""
            if parent_path:
                parent = model.get_submodule(parent_path)
            else:
                parent = model
            setattr(parent, module_name.split(".")[0] if "." in module_name else module_name, quantized_layer)

            module._is_hf_initialized = True
            return {}

        return {full_layer_name: value}


class TinyQuantDeserialize(ConversionOps):

    def __init__(self, hf_quantizer):
        self.hf_quantizer = hf_quantizer

    def convert(
        self,
        input_dict: dict[str, list["torch.Tensor"]],
        full_layer_name: str | None = None,
        model: "torch.nn.Module | None" = None,
        **kwargs,
    ) -> dict[str, "torch.Tensor"]:
        from tinyquant.quantized_linear import QuantizedLinear

        parts = full_layer_name.split(".")

        for i in range(len(parts), 0, -1):
            try:
                module_path = ".".join(parts[:i])
                module = model.get_submodule(module_path)
                if isinstance(module, QuantizedLinear):
                    for key, value in input_dict.items():
                        tensor = value[0] if isinstance(value, list) else value
                        tensor_key = key.split(".")[-1]
                        module.weights_dict[tensor_key] = nn.Parameter(tensor, requires_grad=False)
                    module._is_hf_initialized = True
                    return {}
            except (AttributeError, KeyError):
                continue

        result = {}
        for key, value in input_dict.items():
            tensor = value[0] if isinstance(value, list) else value
            result[key] = tensor
        return result


def replace_with_tinyquant_linear(
    model: "torch.nn.Module",
    quantization_config,
    modules_to_not_convert: list[str] | None = None,
):
    from tinyquant.quantized_linear import QuantizedLinear

    modules_to_not_convert = modules_to_not_convert or []
    layers_pattern = quantization_config.layers or "*"

    modules_to_replace = []

    for module_name, module in model.named_modules():
        skip = False
        for excluded in modules_to_not_convert:
            if excluded in module_name:
                skip = True
                break
        if skip:
            continue

        if isinstance(module, nn.Linear):
            if not fnmatch.fnmatch(module_name, layers_pattern):
                continue
            modules_to_replace.append(module_name)

    for module_name in modules_to_replace:
        parent_name, child_name = (
            module_name.rsplit(".", 1) if "." in module_name else ("", module_name)
        )
        parent = model.get_submodule(parent_name) if parent_name else model

        with torch.device("meta"):
            setattr(parent, child_name, QuantizedLinear.empty())

    return model
