from enum import IntEnum

class ObjectType(IntEnum):
    Node = 0
    Edge = 1
    State = 2

class Property(IntEnum):
    ObjectType = 0
    Node = 1
    Edge = 2
