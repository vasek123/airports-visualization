from PySide6.QtGui import QColor
from enum import IntEnum

NODES_Z_VALUE = 3.0
NO_AIRPORT_SELECTED_LABEL = "No airport is selected"

class ObjectType(IntEnum):
    Node = 0
    Edge = 1
    State = 2

class Property(IntEnum):
    ObjectType = 0
    Node = 1
    Edge = 2

GOOGLE_COLORS = {
    "green": QColor.fromRgb(186, 228, 184),
    "lightBlue": QColor.fromRgb(156, 209, 254),
    "darkGreen": QColor.fromRgb(132, 202, 149),
    "blue": QColor.fromRgb(13, 74, 228)
}

COLORS = {
    "green": QColor.fromRgb(195, 224, 200),
    "gray": QColor.fromRgb(115, 115, 115),
    # "blue": QColor.fromRgb(106, 189, 223),
    # "blue": QColor.fromRgb(17, 30, 108),
    "blue": QColor.fromRgb(13, 74, 228),
    "lightBlue": QColor.fromRgb(195, 214, 243)
}
