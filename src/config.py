from pydantic import BaseModel, field_validator
from typing import List, Tuple, Dict, Optional
from loguru import logger
import sys

logger.add(sys.stderr, format="{time:MMMM D, YYYY > HH:mm:ss} | {level} | {message}")


class CNNArgument(BaseModel):
    num_classes: int
    conv_filter_size: List[int]
    conv_kernel_size: List[int]
    conv_padding: int
    pool_kernel_size: int
    pool_stride: int
    dropout_rate: float
    fc_units: List[int]

    @field_validator("conv_filter_size", "conv_kernel_size", "fc_units")
    def validate_two_layers(cls, v, values, field):
        if len(v) != 2:
            raise ValueError(
                f"The field '{field.name}' must have exactly two values representing two convolutional layers."
            )
        return v

    @field_validator("conv_kernel_size")
    def filter_size_and_kernel_size_match(cls, v, values):
        if "conv_filter_size" in values and len(values["conv_filter_size"]) != len(v):
            raise ValueError(
                "The number of convolutional filter sizes must match the number of kernel sizes."
            )
        return v


# TODO: Complete the arguments for ViT model
class ViTArgument(BaseModel): ...
