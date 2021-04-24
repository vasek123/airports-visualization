
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
