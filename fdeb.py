import time
import numpy as np
from numba import jit, prange
from numba.experimental import jitclass
from graph import Node, Edge
from typing import List
import time

# Possibly compute the compatibilities when needed, not before hand

class FDEB():
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges = edges
        self.compatibility_threshold = 0.05
        self.compatibility = self.compute_compatibilities()

        self.step_size = 4
        self.K = 0.1 
        self.k = self.K

    def calculate_forces(self):
        subdivision_points_count = len(self.edges[0].subdivision_points)
        forces = np.zeros((len(self.edges), subdivision_points_count, 2))
        
        for idx_a, edge_a in enumerate(self.edges):
            for idx_b, edge_b in enumerate(self.edges):
                if idx_a <= idx_b or self.compatibility[idx_a, idx_b] < self.compatibility_threshold:
                    continue

                for i in range(subdivision_points_count):
                    direction = np.array(edge_a.subdivision_points[i].get_direction_to(edge_b.subdivision_points[i]))
                    # There will be a problem with forces that are too large for item too close together
                    distance = edge_a.subdivision_points[i].distance_from(edge_b.subdivision_points[i])
                    if distance >= 1:
                        f = self.compatibility[idx_a, idx_b] / distance

                        direction *= f
                        forces[idx_a, i, :] += direction
                        forces[idx_b, i, :] -= direction

            for i in range(subdivision_points_count):
                # Compute the interaction between the neighbouring subdivision points
                sub_point = edge_a.subdivision_points[i]
                for neighbour in [sub_point.previous_neighbour, sub_point.next_neighbour]:
                    neighbour_dir = np.array(sub_point.get_direction_to(neighbour))
                    neighbour_dist = sub_point.distance_from(neighbour)
                    if neighbour_dist > 0:
                        # forces[idx_a, i, :] += self.k * neighbour_dir * neighbour_dist
                        pass

        return forces

    def compute_compatibilities(self):
        compatibility = np.ones((len(self.edges), len(self.edges)))

        start = time.time()
        for idx_a, edge_a in enumerate(self.edges):
            for idx_b, edge_b in enumerate(self.edges):
                if idx_a <= idx_b:
                    continue

                compatibility[idx_a, idx_b] = self.edge_compatibility_measure(
                    (edge_a.source.x, edge_a.source.y), edge_a.get_direction_vector(),
                    (edge_b.source.x, edge_b.source.y), edge_b.get_direction_vector()
                )

                compatibility[idx_b, idx_a] = compatibility[idx_a, idx_b]

        end = time.time()

        # print("Duration:", end - start)

        counter = 0
        print(self.compatibility_threshold)
        for i in range(len(self.edges)):
            for j in range(len(self.edges)):
                if compatibility[i, j] >= self.compatibility_threshold:
                    counter += 1

        print((len(self.edges) ** 2) / 2, "->", counter / 2)

        return compatibility

    @staticmethod
    @jit(nopython=True)
    def edge_compatibility_measure(edge_a_source, edge_a_dir, edge_b_source, edge_b_dir):
        len_a = np.linalg.norm(edge_a_dir)
        len_b = np.linalg.norm(edge_b_dir)
        mid_a = np.array([edge_a_source[0] + edge_a_dir[0] / 2, edge_a_source[1] + edge_a_dir[1] / 2])
        mid_b = np.array([edge_b_source[0] + edge_b_dir[0] / 2, edge_b_source[1] + edge_b_dir[1] / 2])
        l_avg = (len_a + len_b) / 2

        x = np.dot(edge_a_dir, edge_b_dir) / (len_a * len_b)
        alpha = np.arccos(x)
        angle_compatibility = abs(np.cos(alpha))

        scale_compatibility = 2 / (l_avg / min(len_a, len_b) + max(len_a, len_b) / l_avg)

        position_compatibility = l_avg / (l_avg + np.linalg.norm(mid_a - mid_b))

        visibility_compatibility = self.visibility_compatibility(edge_a, edge_b)

        return angle_compatibility * scale_compatibility * position_compatibility * visibility_compatibility

    def visibility_compatibility(self, edge_a: Edge, edge_b):
        return min(
            self.visibility_compatibility_helper(edge_a, edge_b),
            self.visibility_compatibility_helper(edge_b, edge_a)
        )

    def visibility_compatibility_helper(self, edge_a: Edge, edge_b):
        i_0, i_1 = self.get_intersection_points(edge_a, edge_b)
        b_mid = np.array([edge_b.source.x, edge_b.source.y]) + edge_b.get_direction_vector() / 2
        # a_mid = np.array([edge_a.source.x, edge_a.source.y]) + edge_a.get_direction_vector() / 2
        # print("b_mid -> i_mid:", np.linalg.norm(b_mid - (i_0 + i_1) / 2))
        # print("a_mid -> i_mid:", np.linalg.norm(a_mid - (i_0 + i_1) / 2))
        # print(1 - 2 * np.linalg.norm(b_mid - (i_0 + i_1) / 2) / np.linalg.norm(i_1 - i_0))
        return max(0, 1 - 2 * np.linalg.norm(b_mid - (i_0 + i_1) / 2) / np.linalg.norm(i_1 - i_0))

    @staticmethod
    def perpendicular(x):
        return np.array([-x[1], x[0]])

    def get_intersection(self, a1, a2, b1, b2):

        dir_a = a2 - a1
        dir_b = b2 - b1
        dir_off = a1 - b1
        dir_a_perp = self.perpendicular(dir_a)

        x = np.dot(dir_a_perp, dir_off)
        y = np.dot(dir_a_perp, dir_b)

        return (x / y) * dir_b + b1

    def get_intersection_points(self, edge_a: Edge, edge_b: Edge):
        a_source = np.array([edge_a.source.x, edge_a.source.y])
        a_target = np.array([edge_a.target.x, edge_a.target.y])
        b_source = np.array([edge_b.source.x, edge_b.source.y])
        b_target = np.array([edge_b.target.x, edge_b.target.y])

        edge_a_dir_perp = self.perpendicular(edge_a.get_direction_vector())
        first_intersection = self.get_intersection(a_source, a_source + edge_a_dir_perp, b_source, b_target)
        second_intersection = self.get_intersection(a_target, a_target + edge_a_dir_perp, b_source, b_target)

        return first_intersection, second_intersection



    def apply_forces(self, forces):
        subdivision_points_count = len(self.edges[0].subdivision_points)
        for edge_idx, edge in enumerate(self.edges):
            for sub_idx in range(subdivision_points_count):
                edge.subdivision_points[sub_idx].x += self.step_size * forces[edge_idx, sub_idx, 0]
                edge.subdivision_points[sub_idx].y += self.step_size * forces[edge_idx, sub_idx, 1]

    def iteration_step(self, step):
        """
        if step % 20 == 0:
            for edge in self.edges:
                edge.add_subdivisions()
        """

        start = time.time()
        forces = self.calculate_forces()
        end = time.time()

        print("Computing forces took {} s".format(end - start))

        start = time.time()
        self.apply_forces(forces)
        end = time.time()

        print("Applying forces took {} s".format(end - start))
