import numpy as np
from typing import Set

def compute_direction_vector(source, target, normalized=False):
    vec = np.array([target.x - source.x, target.y - source.y])
    if normalized:
        vec /= np.linalg.norm(vec)

    return vec

class Position():
    def __init__(self, x: float, y: float):
        self.position = np.array([x, y])

    @property
    def x(self):
        return self.position[0]

    @x.setter
    def x(self, new_x):
        self.position[0] = new_x

    @property
    def y(self):
        return self.position[1]

    @y.setter
    def y(self, new_y):
        self.position[1] = new_y


class Node(Position):
    def __init__(self, id: int, size: int, x: float, y: float):
        super().__init__(x, y)
        self.id = id
        self.size = size
        self.connected_edges: Set[Edge] = set()


class Edge():
    def __init__(self, id: int, source: Node, target: Node):
        self.id = id
        self.source = source
        self.target = target
        self.first_subdivision_point = None
        self.subdivision_points = []

        self._direction_vector = compute_direction_vector(self.source, self.target)
        self._direction_vector_norm = self._direction_vector / np.linalg.norm(self._direction_vector)

    def get_direction_vector(self, normalized=False) -> tuple:
        """Returns the direction vector of the edge"""
        if normalized:
            return self._direction_vector_norm
        return self._direction_vector

        # return compute_direction_vector(self.source, self.target, normalized)
 
    def add_subdivisions(self):
        """Subdivides the edge into more parts"""

        # Is it neccessary to keep track of previous neighbours?
        
        # Create the new first subdivision point
        if self.first_subdivision_point is not None:
            direction = compute_direction_vector(self.source, self.first_subdivision_point)
            next_neighbour = self.first_subdivision_point
        else:
            direction = compute_direction_vector(self.source, self.target)
            next_neighbour = self.target

        subdivision_point = SubdivisionPoint(
            x=self.source.x + direction[0]/2, y=self.source.y + direction[1]/2,
            previous_neighbour=self.source, next_neighbour=next_neighbour
        )

        """
        subdivision_point.x += np.random.uniform(10)
        subdivision_point.y += np.random.uniform(10)
        """

        self.first_subdivision_point = subdivision_point
        self.subdivision_points.append(subdivision_point)

        current_point = next_neighbour
        while current_point != self.target:
            direction = compute_direction_vector(current_point, current_point.next_neighbour)
            subdivision_point = SubdivisionPoint(
                x=current_point.x + direction[0]/2, y=current_point.y + direction[1]/2,
                previous_neighbour=current_point, next_neighbour=current_point.next_neighbour
            )

            self.subdivision_points.append(subdivision_point)

            """
            subdivision_point.x += np.random.uniform(100)
            subdivision_point.y += np.random.uniform(10)
            """

            current_point.next_neighbour = subdivision_point
            if subdivision_point.next_neighbour != self.target:
                subdivision_point.next_neighbour.previous_neighbour = subdivision_point

            current_point = subdivision_point.next_neighbour

    def get_subdivisions(self):
        return np.array([point.position for point in self.subdivision_points])

class SubdivisionPoint(Position):
    def __init__(self, x: int, y: int, previous_neighbour, next_neighbour):
        super().__init__(x, y)
        self.previous_neighbour = previous_neighbour
        self.next_neighbour = next_neighbour

    def distance_from(self, other):
        return np.linalg.norm(compute_direction_vector(self, other))

    def get_direction_to(self, other):
        return compute_direction_vector(self, other, normalized=True)
