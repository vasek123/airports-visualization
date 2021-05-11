import argparse
import copy
import time
import numpy as np
import networkx as nx
from graph import Node, Edge
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
                    compatibility_measures_file_path=self.compatibility_measures_file_path)

        counter = 0

        for cycle in self.schema:
            cycle_start = time.time()

            # Add new subidivision points at the beginning of the cycle
            fdeb.add_subdivision_points()

            # Change the step size
            fdeb.step_size = cycle["step_size"]

            for i in range(cycle["iterations_num"]):
                fdeb.iteration_step()
                counter += 1
                print("Completed iterations: {}".format(counter))


            cycle_end = time.time()

            print("Cycle duration: {}s".format(cycle_end - cycle_start))


        resulting_positions = np.zeros(shape=(len(self.edges), 31, 2))
        for edge_idx, edge in enumerate(fdeb.edges):
            resulting_positions[edge_idx, :] = edge.get_subdivisions()

        return resulting_positions

    def precompute_positions(self):
        for idx_K, K in enumerate(self.K_to_compute):
            self.positions[idx_K, :] = self.compute_for_K(K)

    def save_positions(self, file_path):
        np.save(file_path, self.positions)
        np.save(file_path + "_k", self.K_to_compute)


def loadGraph(input_file_path):
    graph = nx.read_graphml(input_file_path)

    nodes = [None] * len(graph.nodes())
    for node_id, _node in graph.nodes(data=True):

        node = Node(id=int(node_id), size=None, code=None, name=None,
                    x=float(_node["x"]), y=float(_node["y"]))

        nodes[int(node_id)] = node

    edges = []
    added = []
    for source, target, attr in graph.edges(data=True):
        # Ignore the edge if it's reverse has already been added
        if (int(target), int(source)) in added or (int(source), int(target)) in added:
            continue

        edge = Edge(id=int(attr["id"]), source=nodes[int(source)], target=nodes[int(target)])
        edges.append(edge)
        added.append((min(int(source), int(target)), max(int(source), int(target))))

        # Add the edge id to the node
        nodes[edge.source.id].connected_edges.add(edge.id)
        nodes[edge.target.id].connected_edges.add(edge.id)

    return nodes, edges


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", "-g", type=str, required=False, default="data/airlines-projected.graphml")
    parser.add_argument("--compatibility", "-c", type=str, required=False, default="data/compatibility-measures.npy")

    args = parser.parse_args()

    nodes, edges = loadGraph(args.graph)

    # K_to_compute = np.linspace(start=0, stop=0.2, endpoint=True)
    K_to_compute=[0]
    schema = [
        { "cycle": 0, "step_size": 8, "iterations_num": 1 },
        { "cycle": 0, "step_size": 8, "iterations_num": 1 },
        { "cycle": 0, "step_size": 8, "iterations_num": 1 },
        { "cycle": 0, "step_size": 8, "iterations_num": 1 },
        { "cycle": 0, "step_size": 8, "iterations_num": 1 }
    ]
    precompute = FDEBPrecompute(nodes, edges, K_to_compute, args.compatibility, schema=schema)

    precompute.precompute_positions()
    precompute.save_positions("./precomputed/positions")



