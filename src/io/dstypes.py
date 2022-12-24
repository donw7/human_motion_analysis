import numpy as np
import sys
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path

from src.io.utils import Config

@dataclass
class Metadata:
	dataset_name: str = None
	dataset_version: dict = None

@dataclass
class Video:
	images: np.ndarray = None
	kpts_true: np.ndarray = None
	kpts_pred: np.ndarray = None
	edges_true: np.ndarray = None
	edges_pred: np.ndarray = None
	config: Config = None
	path: Path = None
	stem_name: str = None
	exercise_name: str = None
	metadata: Metadata = None


@dataclass
class ModelMetadata(Metadata):
	model_name: str = None
	model_version: dict = None