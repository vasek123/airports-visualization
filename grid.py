import numpy as np
from typing import List
from graph import Position, Edge, SubdivisionPoint


class Grid():
    def __init__(self, edges: List[Edge], cell_size: int):
        self.edges = edges
        self.cells = []

class Cell(Position):
    def __init__(self, x: float, y: float, size: float):
        super().__init__(x, y)
        self.size = size
        self.points: List[SubdivisionPoint] = []

    
