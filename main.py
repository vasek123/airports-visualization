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

from fdeb import FDEB
import sys, random, math
import time
import argparse
import networkx as nx
import numpy as np
import shapefile
import geojson
import json
import csv
from typing import List
from graph import Node, Edge
from constants import ObjectType, Property
from PySide6.QtCore import QPointF, Qt, QSize
from PySide6.QtWidgets import QApplication, QColorDialog, QMainWindow, QGraphicsScene, QGraphicsView, QSizePolicy
from PySide6.QtGui import QBrush, QColor, QKeySequence, QPainterPath, QPen, QPolygon, QPolygonF, QTransform, QPainter, QKeyEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class VisGraphicsScene(QGraphicsScene):
    def __init__(self):
        super(VisGraphicsScene, self).__init__()
        self.selection = None
        self.selectedNode = None
        self.wasDragg = False
        colorGreen = QColor(Qt.green)
        # colorGreen.setAlphaF(0.5)
        colorRed = QColor(Qt.red)
        colorRed.setAlphaF(0.5)
        # self.pen = QPen(Qt.black)
        self.pen = QPen(colorGreen)
        self.selected = QPen(colorRed)
        self.selectedOrigColor = None

        self.edgeColor = QColor(Qt.blue)
        self.edgeColor.setAlphaF(0.4)

        self.nodePen = QPen(QColor(Qt.green))

        self.paths = {}

    def addEdge(self, edge_id, path_item):
        self.paths[edge_id] = path_item

    def colorConnectedEdges(self, node, color):
        for edge_id in node.data(Property.ConnectedEdges):
            self.paths[edge_id].setPen(color)

    def mouseReleaseEvent(self, event): 
        if self.wasDragg:
            return

        # If something has been previously selected, set its outline back to black
        if self.selection:
            self.selection.setPen(self.pen if self.selectedOrigColor is None else self.selectedOrigColor)

        if self.selectedNode:
            self.selectedNode.setPen(self.nodePen)
            self.colorConnectedEdges(self.selectedNode, self.edgeColor)

        # Try to get the new item
        item = self.itemAt(event.scenePos(), QTransform())
        if item:

            if item.data(Property.ObjectType) is ObjectType.Node:
                # print("Selected node", item.data(Property.NodeId))
                item.setPen(self.selected)
                self.colorConnectedEdges(item, self.selected)
                self.selectedNode = item
            else:
                return

            # Sets its outline to the "selected" color and store it in self.selection
            # self.selectedOrigColor = item.pen()
            # item.setPen(self.selected)
            # self.selection = item

class VisGraphicsView(QGraphicsView):
    def __init__(self, scene, parent):
        super(VisGraphicsView, self).__init__(scene, parent)
        self.startX = 0.0
        self.startY = 0.0
        self.distance = 0.0
        self.myScene = scene
        self.setViewport(QOpenGLWidget())
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

    RADIUS = 4

    def __init__(self, airlines_file_path, map_shape_file_path, compatibility_measure_file_path, max_number):
        super(MainWindow, self).__init__()
        self.setWindowTitle('VIZ Qt for Python Example')
        self.createGraphicView()

        self.max_number = max_number
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.step = 1

        self.pathItems = []

        self.loadTopology(map_shape_file_path)
        self.generateMap()

        self.loadGraph(airlines_file_path)
        self.generateGraph()

        for edge in self.edges:
            edge.add_subdivisions()

        self.fdeb = FDEB(self.nodes, self.edges, compatibility_measure_file_path)

        """
        x = -2
        y = 4
        for a, b in [(x, y)]:
            i_0, i_1 = self.fdeb.get_intersection_points(self.edges[a], self.edges[b])
            self.scene.addEllipse(i_0[0], i_0[1], 10, 10, brush=QBrush(Qt.red))
            self.scene.addEllipse(i_1[0], i_1[1], 10, 10, brush=QBrush(Qt.red))
            self.scene.addEllipse((i_0[0] + i_1[0]) / 2, (i_0[1] + i_1[1]) / 2, 10, 10, brush=QBrush(Qt.red))
            self.scene.addLine(self.edges[a].source.x, self.edges[a].source.y, i_0[0], i_0[1])
            self.scene.addLine(self.edges[a].target.x, self.edges[a].target.y, i_1[0], i_1[1])
            self.pathItems[a].setPen(QPen(Qt.red))
            self.pathItems[b].setPen(QPen(Qt.red))

        # print("Visibility compatibility:", self.fdeb.visibility_compatibility(self.edges[x], self.edges[y]))
        """
            
        self.show()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() == Qt.Key_X:
            self.fdeb.step_size /= 2
            for edge in self.edges:
                edge.add_subdivisions()

            self.fdeb.k = self.fdeb.K / len(self.edges[0].subdivision_points)

            print("Added subdivisions, new count:", len(self.edges[0].subdivision_points))
            print("Step size:", self.fdeb.step_size, ", k=", self.fdeb.k)

            return super().keyPressEvent(event)

        self.fdeb.iteration_step(self.step)
        self.step += 1
        self.updateEdgePaths()
        print("step:", self.step)
        print("PathItems count:", len(self.pathItems))
        print("Edges count:", len(self.edges))
        return super().keyPressEvent(event)

    def updateEdgePaths(self):
        if not self.edges:
            return

        start = time.time()
        for idx, edge in enumerate(self.edges):
            path = self.generateEdgePath(edge)
            self.pathItems[idx].setPath(path)
        end = time.time()

        print("Updating edges took {}s".format(end - start))

    def createEdgesPath(self):
        if not self.edges:
            return
        
        color = QColor(Qt.blue)
        color.setAlphaF(0.4)
        pen = QPen(color)

        for edge in self.edges:
            path = self.generateEdgePath(edge)
            pathItem = self.scene.addPath(path)
            pathItem.setPen(pen)
            pathItem.setData(Property.EdgeId, edge.id)
            pathItem.setData(Property.ObjectType, ObjectType.Edge)
            self.pathItems.append(pathItem)
            self.scene.addEdge(edge.id, pathItem)

    def createGraphicView(self):
        self.scene = VisGraphicsScene()
        self.brush = [QBrush(Qt.yellow), QBrush(Qt.green), QBrush(Qt.blue)]
        self.view = VisGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)
        self.view.setGeometry(0, 0, 800, 600)

    def calculateDegree(self, graph):
        degrees = [0] * len(graph.nodes())
        processed_edges = []
        for edge in graph.edges():
            if (edge[0], edge[1]) not in processed_edges and (edge[1], edge[0]) not in processed_edges:
                degrees[int(edge[0])] += 1
                degrees[int(edge[1])] += 1
                processed_edges.append((edge[0], edge[1]))
        return degrees

    def loadGraph(self, input_file_path):
        graph = nx.read_graphml(input_file_path)
        degrees = self.calculateDegree(graph)

        NUM = self.max_number 
        self.nodes = [None] * len(graph.nodes())
        self.nodes = [None] * min(NUM, len(graph.nodes()))
        min_x = float("inf")
        min_y = float("inf")
        max_x = -float("inf")
        max_y = -float("inf")
        max_size = -float("inf")
        min_size = float("inf")
        for node_id, node in graph.nodes(data=True):
            if int(node_id) >= NUM:
                continue
            node = Node(id=int(node_id), size=degrees[int(node_id)], x=float(node["x"]) - self.RADIUS/2, y=float(node["y"]) - self.RADIUS/2)
            min_x = min(min_x, node.x)
            min_y = min(min_y, node.y)
            max_x = max(max_x, node.x)
            max_y = max(max_y, node.y)

            max_size = max(max_size, graph.degree[node_id])
            min_size = min(min_size, graph.degree[node_id])

            self.nodes[int(node_id)] = node

        print(min_x, min_y)
        print(max_x, max_y)
        print(min_size, max_size)

        added = []
        for source, target, attr in graph.edges(data=True):
            # Use only edges that are connected the some of the first NUM aiports
            if int(source) < NUM and int(target) < NUM:
                # Ignore the edge if it's reverse has already been added
                if (int(target), int(source)) in added or (int(source), int(target)) in added:
                    continue

                edge = Edge(id=int(attr["id"]), source=self.nodes[int(source)], target=self.nodes[int(target)])
                self.edges.append(edge)
                added.append((min(int(source), int(target)), max(int(source), int(target))))

                # Add the edge id to the node
                self.nodes[edge.source.id].connected_edges.add(edge.id)
                self.nodes[edge.target.id].connected_edges.add(edge.id)

        print("Total number of edges:", len(added))

    def loadTopology(self, input_file_path):
        with open(input_file_path, "r") as f:
            topology = geojson.load(f)
            self.topology = topology

    def generateMap(self):

        # mapBrush = QBrush(QColor("#FFF2AF"))
        IGNORED_STATES = ["Alaska", "Hawaii"]

        def generatePolygon(geometry):
            polygon = QPolygonF()
            for point in geometry:
                polygon.append(QPointF(*point))

            return polygon

        for feature in self.topology.features:
            if feature["properties"]["name"] in IGNORED_STATES:
                continue

            if feature["geometry"]["type"] == "Polygon":
                geometry = feature["geometry"]["coordinates"][0]
                polygon = generatePolygon(geometry)
                item = self.scene.addPolygon(polygon, pen=self.scene.pen)
            elif feature["geometry"]["type"] == "MultiPolygon":
                for geometry in feature["geometry"]["coordinates"]:
                    polygon = generatePolygon(geometry[0])
                    item = self.scene.addPolygon(polygon, pen=self.scene.pen)

            item.setData(Property.ObjectType, ObjectType.State)
        

    def generateGraph(self):
        self.createEdgesPath()
        for node in self.nodes:
            nodeItem = self.scene.addEllipse(node.x, node.y, self.RADIUS, self.RADIUS, self.scene.pen, self.brush[0])
            nodeItem.setData(Property.ObjectType, ObjectType.Node)
            nodeItem.setData(Property.NodeId, node.id)
            nodeItem.setData(Property.ConnectedEdges, node.connected_edges)

    """
    def randomPathChange(self, times=1, change=1):
        for t in range(times):
            for path, item in zip(self.paths, self.pathItems):
                for idx in range(1, path.elementCount() - 1):
                    change_x = np.random.uniform(-change, change)
                    change_y = np.random.uniform(-change, change)

                    # print(path.elementAt(idx).x, path.elementAt(idx).y, end=" ")
                    path.setElementPositionAt(
                        idx, path.elementAt(idx).x + change_x, path.elementAt(idx).y + change_y)
                    # print(path.elementAt(idx).x, path.elementAt(idx).y)

                item.setPath(path)
    """

    def generateEdgePath(self, edge):
        path = QPainterPath(QPointF(edge.source.x + self.RADIUS/2, edge.source.y + self.RADIUS/2))
        
        # If there are no subdivision points, draw a line to the target node
        if not edge.subdivision_points:
            path.lineTo(edge.target.x + self.RADIUS/2, edge.target.y + self.RADIUS/2)
            return path

        current_point = edge.first_subdivision_point
        while True:
            if current_point == edge.target:
                path.lineTo(current_point.x + self.RADIUS/2, current_point.y + self.RADIUS/2)
                break
            path.lineTo(current_point.x, current_point.y)
            current_point = current_point.next_neighbour

        return path

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

    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", "-g", type=str, required=False, default="data/airlines-projected.graphml")
    parser.add_argument("--map", "-m", type=str, required=False, default="data/us-states.json")
    parser.add_argument("--compatibility", "-c", type=str, required=False, default="data/compatibility-measures.npy")
    parser.add_argument("--number", "-n", type=int, required=False, default=300)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ex = MainWindow(args.graph, args.map, args.compatibility, args.number)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
