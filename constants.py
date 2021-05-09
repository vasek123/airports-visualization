from enum import IntEnum

NODES_Z_VALUE = 3.0

class ObjectType(IntEnum):
    Node = 0
    Edge = 1
    State = 2

class Property(IntEnum):
    ObjectType = 0
    Node = 1
    Edge = 2
