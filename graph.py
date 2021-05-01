import numpy as np
from numba import jit
from numba.experimental import jitclass

@jit(nopython=True)
def compute_direction_vector(source, target, normalized=False):
<<<<<<< HEAD
    # vec = (target.x - source.x, target.y - source.y)
    vec = target - source
    if normalized:
        size = np.linalg.norm(vec)
        vec = vec / size
=======
    vec = np.array([target.x - source.x, target.y - source.y])
    if normalized:
        vec /= np.linalg.norm(vec)
>>>>>>> main

    return vec

@jit(nopython=True)
def distance_from(source, target):
    return np.linalg.norm(compute_direction_vector(source, target))

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

        self._direction_vector = compute_direction_vector(self.source, self.target)
        self._direction_vector_norm = self._direction_vector / np.linalg.norm(self._direction_vector)

    def get_direction_vector(self, normalized=False) -> tuple:
        """Returns the direction vector of the edge"""
        if normalized:
            return self._direction_vector_norm
        return self._direction_vector

<<<<<<< HEAD
        return compute_direction_vector(
            np.array([self.source.x, self.source.y]),
            np.array([self.target.x, self.target.y]),
            normalized
        )
=======
        # return compute_direction_vector(self.source, self.target, normalized)
>>>>>>> main
 
    def add_subdivisions(self):
        """Subdivides the edge into more parts"""

        # Is it neccessary to keep track of previous neighbours?
        
        # Create the new first subdivision point
        if self.first_subdivision_point is not None:
            direction = compute_direction_vector(
                np.array([self.source.x, self.source.y]),
                np.array([self.first_subdivision_point.x, self.first_subdivision_point.y])
            )
            next_neighbour = self.first_subdivision_point
        else:
            direction = compute_direction_vector(
                np.array([self.source.x, self.source.y]),
                np.array([self.target.x, self.target.y])
            )
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
            direction = compute_direction_vector(
                np.array([current_point.x, current_point.y]),
                np.array([current_point.next_neighbour.x, current_point.next_neighbour.y])
            )
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

class SubdivisionPoint():
    def __init__(self, x: int, y: int, previous_neighbour, next_neighbour):
        self.x = x
        self.y = y
        self.previous_neighbour = previous_neighbour
        self.next_neighbour = next_neighbour

    def distance_from(self, other):
        """
        return np.linalg.norm(compute_direction_vector(
            np.array([self.x, self.y]),
            np.array([other.x, other.y])
        ))
        """

        return distance_from(
            np.array([self.x, self.y]),
            np.array([other.x, other.y])
        )

    def get_direction_to(self, other):
        return compute_direction_vector(
            np.array([self.x, self.y]),
            np.array([other.x, other.y]),
            normalized=True
        )
