import os
import yaml
from dataclasses import dataclass, field
from itertools import product

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
        self.kpts, self.edges, self.params = [self.load_config(fname) for fname in [self.kpts_fname, self.edgecolors_fname, self.params_fname]]
        self.edge_names = self.get_edge_names(self.kpts, self.edges)

    def load_config(self, fname: str):
        with open(os.path.join("configs", fname)) as file:
            config_file = yaml.full_load(file)
        return config_file

    def get_edge_names(self, kpts: dict, edges: dict):
        '''construct edge names by mapping edge tuples onto keypoint names'''
        num_edges = len(edges)
        return [
            f"{list(kpts)[edge[0]]}-{list(kpts)[edge[1]]}"
            for edge in list(edges)
        ]

    def get_name_combinations(self):
        name_combinations = []
        for e1,e2 in product(self.edge_names, self.params["XY_NAMES"]):
            name_combinations.append(f"{e1}-{e2}")
        return name_combinations