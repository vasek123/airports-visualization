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

        # Is it neccessary to keep track of previous neighbours?
        
        # Create the new first subdivision point
        if self.first_subdivision_point is not None:
            direction = compute_direction_vector(self.source, self.first_subdivision_point)
            next_neighbour = self.first_subdivision_point
        else:
            direction = compute_direction_vector(self.source, self.target)
            next_neighbour = self.target

        subdivision_point = SubdivisionPoint(
            x=self.source.x + direction[0]/2, y=self.target.y + direction[1]/2,
            previous_neighbour=self.source, next_neighbour=next_neighbour
        )

        self.subdivision_points.append(subdivision_point)

        current_point = next_neighbour
        while current_point != self.target:
            direction = compute_direction_vector(current_point, current_point.next_neighbour)
            subdivision_point = SubdivisionPoint(
                x=current_point.x + direction[0]/2, y=current_point.y + direction[1]/2,
                previous_neighbour=current_point, next_neighbour=current_point.next_neighbour
            )

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
