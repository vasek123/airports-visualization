import numpy as np
from graph import Node, Edge
from typing import List

# Possibly compute the compatibilities when needed, not before hand

class FDEB():
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges = edges
        self.compatibility = self.compute_compatibilities()

    def calculate_forces(self):
        subdivision_points_count = len(self.edges[0].subdivision_points)
        forces = np.zeros((len(self.edges), subdivision_points_count, 2))
        
        for idx_a, edge_a in enumerate(self.edges):
            for idx_b, edge_b in enumerate(self.edges):
                if idx_a <= idx_b:
                    continue

                for i in range(subdivision_points_count):
                    direction = np.array(edge_a.subdivision_points[i].get_direction_to(edge_b.subdivision_points[i]))
                    # There will be a problem with forces that are too large for item too close together
                    distance = edge_a.subdivision_points[i].distance_from(edge_b.subdivision_points[i])
                    if distance >= 1:
                        f = 1.0 / distance ** 2
                        f *= self.compatibility[idx_a, idx_b]

                        direction *= f
                        forces[idx_a, i, :] += direction
                        forces[idx_b, i, :] -= direction

            for i in range(subdivision_points_count):
                # Compute the interaction between the neighbouring subdivision points
                sub_point = edge_a.subdivision_points[i]
                for neighbour in [sub_point.previous_neighbour, sub_point.next_neighbour]:
                    neighbour_dir = np.array(sub_point.get_direction_to(neighbour))
                    neighbour_dist = sub_point.distance_from(neighbour)
                    if neighbour_dist > 1:
                        forces[idx_a, i, :] += 0.2 * neighbour_dir * neighbour_dist

        return forces

    def compute_compatibilities(self):
        compatibility = np.ones((len(self.edges), len(self.edges)))

        for idx_a, edge_a in enumerate(self.edges):
            for idx_b, edge_b in enumerate(self.edges):
                if idx_a <= idx_b:
                    continue

                compatibility[idx_a, idx_b] = self.edge_compatibility_measure(edge_a, edge_b)
                compatibility[idx_b, idx_a] = compatibility[idx_a, idx_b]

        counter = 0
        for i in range(len(self.edges)):
            for j in range(len(self.edges)):
                if compatibility[i, j] > 0.05:
                    counter += 1

        print((len(self.edges) ** 2) / 2, "->", counter / 2)

        return compatibility

    def edge_compatibility_measure(self, edge_a: Edge, edge_b: Edge):
        edge_a_dir = np.array(edge_a.get_direction_vector())
        edge_b_dir = np.array(edge_b.get_direction_vector())
        len_a = np.linalg.norm(edge_a_dir)
        len_b = np.linalg.norm(edge_b_dir)
        mid_a = np.array([edge_a.source.x + edge_a_dir[0] / 2, edge_a.source.y + edge_a_dir[1] / 2])
        mid_b = np.array([edge_b.source.x + edge_b_dir[0] / 2, edge_b.source.y + edge_b_dir[1] / 2])
        l_avg = (len_a + len_b) / 2

        x = np.dot(edge_a_dir, edge_b_dir) / (len_a * len_b)
        alpha = np.arccos(x)
        angle_compatibility = abs(np.cos(alpha))

        scale_compatibility = 2 / (l_avg / min(len_a, len_b) + max(len_a, len_b) / l_avg)

        position_compatibility = l_avg / (l_avg + np.linalg.norm(mid_a - mid_b))

        return angle_compatibility * scale_compatibility * position_compatibility

    def apply_forces(self, forces):
        subdivision_points_count = len(self.edges[0].subdivision_points)
        for edge_idx, edge in enumerate(self.edges):
            for sub_idx in range(subdivision_points_count):
                edge.subdivision_points[sub_idx].x += 1 * forces[edge_idx, sub_idx, 0]
                edge.subdivision_points[sub_idx].y += 1 * forces[edge_idx, sub_idx, 1]

    def iteration_step(self):
        forces = self.calculate_forces()
        self.apply_forces(forces)
