import numpy as np

def compute_direction_vector(source, target, normalized=False):
    vec = (target.x - source.x, target.y - source.y)
    if normalized:
        size = np.sqrt(np.sum(np.power(vec, 2)))
        vec[0] /= size
        vec[1] /= size

    return vec

class Node():
    def __init__(self, id: int, x: float, y: float):
        self.id = id
        self.x = x
        self.y = y


class Edge():
    def __init__(self, id: int, source: Node, target: Node):
        self.id = id
        self.source = source
        self.target = target
        self.first_subdivision_point = None
        self.subdivision_points = []

    def get_direction_vector(self, normalized=False) -> tuple:
        """Returns the direction vector of the edge"""

        return compute_direction_vector(self.source, self.target, normalized)

        """
        vec = (self.target.x - self.source.x, self.target.y - self.source.y)
        if normalized:
            size = np.sqrt(np.sum(np.power(vec, 2)))
            vec[0] /= size
            vec[1] /= size

        return vec
        """

    def add_subdivisions(self):
        """Subdivides the edge into more parts"""
        
        # Create the new first subdivision point
        direction = compute_direction_vector()
        subdivision_point = SubdivisionPoint()
        self.subdivision_points.append(SubdivisionPoint())

        dir

class SubdivisionPoint():
    def __init__(self, x: int, y: int, previous_neighbor, next_neighbor):
        self.x = x
        self.y = y
        self.previous_neighbor = previous_neighbor
        self.next_neighbor = next_neighbor
