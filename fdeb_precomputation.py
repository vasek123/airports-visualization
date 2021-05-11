import copy
import numpy as np
from fdeb import FDEB

DEFAULT_SCHEMA = [
    { "cycle": 0, "step_size": 8, "iterations_num": 50 },
    { "cycle": 1, "step_size": 4, "iterations_num": 33 },
    { "cycle": 2, "step_size": 2, "iterations_num": 22 },
    { "cycle": 3, "step_size": 1, "iterations_num": 15 },
    { "cycle": 4, "step_size": 0.5, "iterations_num": 9 },
    { "cycle": 5, "step_size": 0.25, "iterations_num": 7 }
]

class FDEBPrecompute():
    def __init__(self, nodes, edges, K_to_compute, compatibility_measures_file_path=None, schema=None):
        self.nodes = nodes
        self.edges = edges
        self.K_to_compute = K_to_compute
        self.compatibility_measures_file_path = compatibility_measures_file_path

        self.schema = schema if schema is not None else DEFAULT_SCHEMA
        self.total_iterations_count = sum([ cycle["iterations_num"] for cycle in self.schema ])

        number_of_samples = len(K_to_compute)
        self.positions = np.zeros(shape=(number_of_samples, len(edges), 31, 2))

    def compute_for_K(self, K):
        fdeb = FDEB(self.nodes, copy.deepcopy(self.edges), K=K,
                    compatibility_measures_file_path=self.compatibility_measures_input_file_path)

        for cycle in self.schema:
            # Change the step size
            fdeb.step_size = cycle["step_size"]

            for i in cycle["iterations_num"]:
                fdeb.iteration_step()

            # Add new subidivision points at the end of the cycle
            fdeb.add_subdivision_points()


        resulting_positions = np.zeros(shape=(len(self.edges), 31, 2))
        for edge_idx, edge in enumerate(fdeb.edges):
            resulting_positions[edge_idx, :] = edge.get_subdivisions()

        return resulting_positions

    def precompute_positions(self):
        for idx_K, K in enumerate(self.K_to_compute):
            self.positions[idx_K, :] = self.compute_for_K(K)

    def save_positions(file_path):
        pass


