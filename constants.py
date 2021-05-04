from enum import IntEnum

class ObjectType(IntEnum):
    Node = 0
    Edge = 1
    State = 2

class Property(IntEnum):
    ObjectType = 0
    NodeId = 1
    EdgeId = 2
    ConnectedEdges = 3
