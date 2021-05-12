import numpy as np

class FDEBInterpolation():
    def __init__(self, K_file_path, positions_file_path, edges):
        # Load the precomputed positions
        self.K_available = np.sort(np.load(K_file_path))
        self.positions = np.load(positions_file_path)

        self.edges = edges

    def update_positions(self, K):
        # Find between which precomputed Ks the required K is

        # Return the index of the first element that is smaller or equal to the requested K

        left_point = np.argmin(self.K_available <= K) - 1
        right_point = left_point + 1 #   left_point.......PTR....right

        # print(self.K_available[left_point], K, self.K_available[right_point])

        if right_point >= len(self.K_available):
            right_point = None

        distance_between_points = self.K_available[right_point] - self.K_available[left_point]
        alpha = (K - self.K_available[left_point]) / distance_between_points

        new_positions = (1 - alpha) * self.positions[left_point, :] + alpha * self.positions[right_point, :]

        subdivision_points_count = len(self.edges[0].subdivision_points)
        for edge_idx, edge in enumerate(self.edges):
            for sub_idx in range(subdivision_points_count):
                edge.subdivision_points[sub_idx].position = new_positions[edge_idx, sub_idx, :]

