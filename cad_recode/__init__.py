# cad_recode/__init__.py
"""
cad_recode
==========
Core package for the CAD-Recode training + evaluation pipeline.

This module provides public access to key classes and functions:

    from cad_recode import CADRecodeModel, CadRecodeDataset

It simplifies reuse across training, evaluation, inference, and CLI tools.
"""

from .dataset import CadRecodeDataset
from .model import CADRecodeModel
from .utils import (
    sample_points_on_shape,
    farthest_point_sample,
    chamfer_distance,
    save_point_cloud,
    edit_distance,
)

__all__ = [
    "CadRecodeDataset",
    "CADRecodeModel",
    "sample_points_on_shape",
    "farthest_point_sample",
    "chamfer_distance",
    "save_point_cloud",
    "edit_distance",
]