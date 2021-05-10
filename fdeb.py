import time
import numpy as np
from graph import Node, Edge
from typing import List

class FDEB():
    def __init__(self, nodes: List[Node], edges: List[Edge], compatibility_measures_input_file_path=None):
        self.nodes = nodes
        self.edges = edges
        self.compatibility_threshold = 0.05

        if compatibility_measures_input_file_path is not None:
            self.compatibility = np.load(compatibility_measures_input_file_path)
        else:
            self.compatibility = self.compute_compatibilities()
            # np.save("./data/compatibility_measures.npy", self.compatibility)

        self.compatibility = np.where(np.isnan(self.compatibility), 0, self.compatibility)

        self.step_size = 8
        self.K = 0.3
        self.k = self.K / 2

    def calculate_forces(self):
        subdivision_points_count = len(self.edges[0].subdivision_points)
        forces = np.zeros((len(self.edges), subdivision_points_count, 2))
        
        electro_duration = 0
        spring_duration = 0
        subdivision_duration = 0
        for idx_a, edge_a in enumerate(self.edges):

            subdivision_start = time.time()
            edge_a_subdivisions = edge_a.get_subdivisions()
            subdivision_end = time.time()

            subdivision_duration += subdivision_end - subdivision_start

            electro_start = time.time()

            for idx_b, edge_b in enumerate(self.edges):
                if idx_a <= idx_b or self.compatibility[edge_a.id, edge_b.id] < self.compatibility_threshold:
                    continue

                # Convert this inner part into numpy
                subdivision_start = time.time()
                edge_b_subdivisions = edge_b.get_subdivisions()
                subdivision_end = time.time()

                subdivision_duration += subdivision_end - subdivision_start

                # print("B:", edge_b_subdivisions)
                directions = edge_b_subdivisions - edge_a_subdivisions
                distances = np.linalg.norm(directions, axis=1)

                # Make sure that division by 0 cannot occur
                # The direction is already [0, 0] if the distance is 0
                distances = np.where(distances == 0, 1, distances)
                unit_directions = directions / distances[:, None] # Broadcasting
                distance_mask = (distances >= 1).astype(int)


                compatibility = self.compatibility[edge_a.id, edge_b.id]
                forces_applied = unit_directions * distance_mask[:, None] * compatibility / np.power(distances[:, None], 2)

                forces[idx_a, :, :] += forces_applied
                forces[idx_b, :, :] -= forces_applied

                """
                for i in range(subdivision_points_count):
                    direction = np.array(edge_a.subdivision_points[i].get_direction_to(edge_b.subdivision_points[i]))
                    # print("direction:", direction, unit_directions[i])
                    # There will be a problem with forces that are too large for item too close together
                    distance = edge_a.subdivision_points[i].distance_from(edge_b.subdivision_points[i])
                    # print(distance == distances[i], distance, distances[i], np.all(direction == unit_directions[i]), direction, unit_directions[i])
                    if distance >= 1:
                        f = self.compatibility[idx_a, idx_b] / distance
                        direction *= f
                        forces[idx_a, i, :] += direction
                        forces[idx_b, i, :] -= direction
                """
                # print(np.sum(np.sum(np.abs(forces - new_forces))))

            electro_end = time.time()

            electro_duration += electro_end - electro_start

            spring_start = time.time()
            for i in range(subdivision_points_count):
                # Compute the interaction between the neighbouring subdivision points
                sub_point = edge_a.subdivision_points[i]
                for neighbour in [sub_point.previous_neighbour, sub_point.next_neighbour]:
                    neighbour_dir = np.array(sub_point.get_direction_to(neighbour))
                    neighbour_dist = sub_point.distance_from(neighbour)
                    if neighbour_dist > 0:
                        forces[idx_a, i, :] += self.k * neighbour_dir * neighbour_dist
            
            spring_end = time.time()

            spring_duration += spring_end - spring_start

        # print("Computing electrostatic forces took {}s".format(electro_duration))
        # print("Computing spring forces took {}s".format(spring_duration))
        # print("Generating subdivision points matrices took {}s".format(subdivision_duration))

        return forces

    def compute_compatibilities(self):
        compatibility = np.ones((2101, 2101))

        for idx_a, edge_a in enumerate(self.edges):
            for idx_b, edge_b in enumerate(self.edges):
                if idx_a <= idx_b:
                    continue

                compatibility[edge_a.id, edge_b.id] = self.edge_compatibility_measure(edge_a, edge_b)
                compatibility[edge_b.id, edge_a.id] = compatibility[edge_a.id, edge_b.id]

        counter = 0
        print(self.compatibility_threshold)
        for i in range(len(self.edges)):
            for j in range(len(self.edges)):
                if compatibility[i, j] >= self.compatibility_threshold:
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

    def iteration_step(self, step, verbose=False):

        start = time.time()
        forces = self.calculate_forces()
        end = time.time()

        if verbose:
            print("Computing forces took {} s".format(end - start))

        start = time.time()
        self.apply_forces(forces)
        end = time.time()

        if verbose:
            print("Applying forces took {} s".format(end - start))
