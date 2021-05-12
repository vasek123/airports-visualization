import numpy as np

class FDEBInterpolation():
    def __init__(self, precomputed_positions_path, edges, K_max=None):
        # Load the precomputed positions
        self.K_available = np.load("{}_k.npy".format(precomputed_positions_path))
        self.positions = np.load("{}.npy".format(precomputed_positions_path))

        self.edges = edges

        self.K_max = K_max
        if self.K_max is None:
            self.K_max = 2 * np.max(self.K_available)

        self.K_available = np.append(self.K_available, self.K_max)
        self.K_available = np.sort(self.K_available)

        self.original_positions = np.zeros(shape=(len(edges), len(edges[0].subdivision_points), 2))
        for edge_idx, edge in enumerate(edges):
            self.original_positions[edge_idx, :] = edge.get_subdivisions()

    def update_positions(self, K):
        # Find between which precomputed Ks the required K is

        # Return the index of the first element that is smaller or equal to the requested K
        left_point = np.argmin(self.K_available < K) - 1
        right_point = left_point + 1

        left_positions = self.positions[left_point, :]
        if right_point >= (len(self.K_available) - 1):
            right_positions = self.original_positions
        else:
            right_positions = self.positions[right_point, :]

        distance_between_points = self.K_available[right_point] - self.K_available[left_point]
        alpha = (K - self.K_available[left_point]) / distance_between_points

        # new_positions = (1 - alpha) * self.positions[left_point, :] + alpha * self.positions[right_point, :]
        new_positions = (1 - alpha) * left_positions + alpha * right_positions

        subdivision_points_count = len(self.edges[0].subdivision_points)
        for edge_idx, edge in enumerate(self.edges):
            for sub_idx in range(subdivision_points_count):
                edge.subdivision_points[sub_idx].position = new_positions[edge_idx, sub_idx, :]

