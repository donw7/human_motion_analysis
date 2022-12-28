import os, sys, yaml
import pandas as pd
from pathlib import Path
from functools import wraps
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List
from itertools import product

def handle_FileNotFoundError(func):
	@wraps(func)
	def wrapper(*args, **kwargs):
		try:
			return func(*args, **kwargs)
		except FileNotFoundError:
			print(f"FileNotFoundError: {args[1]} not found")
		except Exception as e:
			print(f"An unspecified error occurred while loading file {args[1]}: {e}")
	return wrapper

@dataclass
class Config:
	kpts_fname: str = r"keypoints.yaml"
	edgecolors_fname: str = r"edge_colors.yaml"
	params_fname: str = r"analysis_parameters.yaml"
	kpts: dict = None
	edges: dict = None
	params: dict = None
	edge_names: list = field(default_factory=list)

	def __post_init__(self):
		self.kpts, self.edges, self.params = [
			self.load_config(fname) for fname in [
				self.kpts_fname, self.edgecolors_fname, self.params_fname
			]
		]
		self.edge_names = self.get_edge_names()

	@handle_FileNotFoundError
	def load_config(self, fname: str):
		file_path = Path("configs").joinpath(fname)
		with file_path.open() as file:
			config_file = yaml.full_load(file)
		return config_file

	def get_edge_names(self):
		"""construct edge names by mapping edge tuples onto keypoint names"""
		if self.kpts and self.edges:
			num_edges = len(self.edges)
			return [
				f"{list(self.kpts)[edge[0]]}-{list(self.kpts)[edge[1]]}"
				for edge in list(self.edges)
			]

	def get_name_combinations(self):
		name_combinations = []
		for e1,e2 in product(self.edge_names, self.params["XY_NAMES"]):
			name_combinations.append(f"{e1}-{e2}")
		return name_combinations



def convert_inst_to_df(inst: List[SimpleNamespace]):
	"""convert list of class instances to df with each row as each instance and columns being attributes of class instance"""
	return pd.DataFrame([cls.__dict__ for cls in inst])