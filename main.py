# Copyright (c) 2021 Ladislav Čmolík
#
# Permission to use, copy, modify, and/or distribute this software for any purpose with or without fee is 
# hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE 
# INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE 
# FOR ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING 
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import sys, random, math
import networkx as nx
from graph import Node, Edge
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import QApplication, QMainWindow, QGraphicsScene, QGraphicsView, QSizePolicy
from PySide6.QtGui import QBrush, QPen, QTransform, QPainter

class VisGraphicsScene(QGraphicsScene):
    def __init__(self):
        super(VisGraphicsScene, self).__init__()
        self.selection = None
        self.wasDragg = False
        self.pen = QPen(Qt.black)
        self.selected = QPen(Qt.red)

    def mouseReleaseEvent(self, event): 
        if self.wasDragg:
            return
        # If something has been previously selected, set its outline back to black
        if self.selection:
            self.selection.setPen(self.pen)
        # Try to get the new item
        item = self.itemAt(event.scenePos(), QTransform())
        if item:
            # Sets its outline to the "selected" color and store it in self.selection
            item.setPen(self.selected)
            self.selection = item

class VisGraphicsView(QGraphicsView):
    def __init__(self, scene, parent):
        super(VisGraphicsView, self).__init__(scene, parent)
        self.startX = 0.0
        self.startY = 0.0
        self.distance = 0.0
        self.myScene = scene
        self.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)

    def wheelEvent(self, event):
        zoom = 1 + event.angleDelta().y()*0.001;
        self.scale(zoom, zoom)
        
    def mousePressEvent(self, event):
        self.startX = event.pos().x()
        self.startY = event.pos().y()
        self.myScene.wasDragg = False
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        endX = event.pos().x()
        endY = event.pos().y()
        deltaX = endX - self.startX
        deltaY = endY - self.startY
        distance = math.sqrt(deltaX*deltaX + deltaY*deltaY)
        if distance > 5:
            self.myScene.wasDragg = True
        super().mouseReleaseEvent(event)

class MainWindow(QMainWindow):
    def __init__(self, input_file):
        super(MainWindow, self).__init__()
        self.setWindowTitle('VIZ Qt for Python Example')
        self.createGraphicView()

        self.nodes = []
        self.edges = []
        self.loadAndMapGraph(input_file)

        # self.generateAndMapDataOld()
        # self.setMinimumSize(800, 600)
        self.show()

    def createGraphicView(self):
        self.scene = VisGraphicsScene()
        self.brush = [QBrush(Qt.yellow), QBrush(Qt.green), QBrush(Qt.blue)]
        self.view = VisGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)
        self.view.setGeometry(0, 0, 800, 600)

    def loadAndMapGraph(self, input_file):
        graph = nx.read_graphml(input_file)

        self.nodes = [None] * len(graph.nodes())
        for node_id, node in graph.nodes(data=True):
            self.nodes[int(node_id)] = Node(id=int(node_id), x=float(node["x"]), y=float(node["y"]))

        for source, target in graph.edges():
            self.edges.append(Edge(id=len(self.edges), source=source, target=target))


    def generateAndMapDataOld(self):
        #Generate random data
        count = 100;
        x = []
        y = []
        r = []
        c = []
        for i in range(0, count):
            x.append(random.random()*600)
            y.append(random.random()*400)
            r.append(random.random()*50)
            c.append(random.randint(0, 2))

        #Map data to graphical elements
        for i in range(0, count):
            d = 2*r[i]
            ellipse = self.scene.addEllipse(x[i], y[i], d, d, self.scene.pen, self.brush[c[i]])

def main():
    app = QApplication(sys.argv)
    ex = MainWindow(sys.argv[1])
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
