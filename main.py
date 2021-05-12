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

from fdeb_interpolation import FDEBInterpolation
import sys, random, math
import time
import argparse
import networkx as nx
import numpy as np
from numpy.lib.function_base import gradient
import pandas as pd
import geojson
from typing import List
from graph import Node, Edge
from fdeb import FDEB
from constants import ObjectType, Property, NODES_Z_VALUE, NO_AIRPORT_SELECTED_LABEL, COLORS, GOOGLE_COLORS
from PySide6.QtCore import QPointF, QRect, Qt
from PySide6.QtWidgets import QApplication, QBoxLayout, QGraphicsRectItem, QGridLayout, QHBoxLayout, QMainWindow, QGraphicsScene, QGraphicsView, QGraphicsTextItem, QPushButton, QSlider, QToolBar, QLabel
from PySide6.QtGui import QBrush, QColor, QLinearGradient, QPainterPath, QPen, QPolygonF, QTextItem, QTransform, QPainter, QKeyEvent, QGradient, QPixmap
from PySide6.QtOpenGLWidgets import QOpenGLWidget


class VisGraphicsScene(QGraphicsScene):
    def __init__(self, selection_change_handler):
        super(VisGraphicsScene, self).__init__()
        self.selection = None
        self.selectedNode = None
        self.wasDragg = False
        self.selection_change_handler = selection_change_handler
        self.setBackgroundBrush(QBrush(GOOGLE_COLORS["lightBlue"]))
        colorGreen = QColor(Qt.green)
        # colorGreen.setAlphaF(0.5)
        colorRed = QColor(Qt.red)
        colorRed.setAlphaF(0.7)
        # self.pen = QPen(Qt.black)
        self.pen = QPen(colorGreen)
        self.pen.setWidthF(0.25)

        self.selected = QPen(colorRed)
        self.selected.setWidthF(1)
        self.selectedOrigColor = None

        self.edgeColor = COLORS["blue"]
        self.edgeColor.setAlphaF(0.3)

        self.nodePen = QPen(COLORS["blue"])
        self.nodePen.setWidthF(0.35)

        self.paths = {}

    def addEdge(self, edge_id, path_item):
        self.paths[edge_id] = path_item

    def colorConnectedEdges(self, node, color, z):
        for edge_id in node.data(Property.Node).connected_edges:
            self.paths[edge_id].setPen(color)
            self.paths[edge_id].setZValue(z)

    def mouseReleaseEvent(self, event): 
        if self.wasDragg:
            return

        # If something has been previously selected, set its outline back to black
        if self.selection:
            self.selection.setPen(self.pen if self.selectedOrigColor is None else self.selectedOrigColor)

        if self.selectedNode:
            self.selectedNode.setPen(self.nodePen)
            self.colorConnectedEdges(self.selectedNode, self.edgeColor, 0)
            self.selection_change_handler(None)

        # Try to get the new item
        item = self.itemAt(event.scenePos(), QTransform())
        if item:

            if item.data(Property.ObjectType) is ObjectType.Node:
                item.setPen(self.selected)
                self.colorConnectedEdges(item, self.selected, 1)
                self.selectedNode = item
                self.selection_change_handler(item.data(Property.Node))
            else:
                return

            
class VisGraphicsView(QGraphicsView):
    def __init__(self, scene: VisGraphicsScene, parent):
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

    RADIUS = 8 

    def __init__(self, airlines_file_path, map_shape_file_path, airport_names_file_path,
                 compatibility_measure_file_path, max_number, precomputed_positions_path):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Force-Directed Edge Bundling - US Flight Connections")
        self.createToolbar()
        self.createGraphicView()
        self.resize(1100,700)

        self.max_number = max_number
        self.nodes: List[Node] = []
        self.edges: List[Edge] = []
        self.step = 0

        self.pathItems = []

        self.loadTopology(map_shape_file_path)
        self.generateMap()

        self.loadGraph(airlines_file_path, airport_names_file_path)
        self.generateGraph()

        for edge in self.edges:
            edge.add_subdivisions()
            edge.add_subdivisions()
            edge.add_subdivisions()
            edge.add_subdivisions()
            edge.add_subdivisions()


        # self.fdeb = FDEB(self.nodes, self.edges, compatibility_measures_file_path=compatibility_measure_file_path)
        max_precomputed_K = np.max(np.load("{}_k.npy".format(precomputed_positions_path)))
        self.K_max = max_precomputed_K * 1.4
        self.fdeb_interpolation = FDEBInterpolation(precomputed_positions_path, self.edges, K_max=self.K_max)
        self.fdeb_interpolation.update_positions(self.K_max)
        self.updateEdgePaths()


        self.show()

    def createGraphicView(self):
        self.scene = VisGraphicsScene(self.selectionChangedHandler)
        self.brush = [QBrush(Qt.yellow), QBrush(Qt.green), QBrush(Qt.blue)]
        self.view = VisGraphicsView(self.scene, self)
        self.setCentralWidget(self.view)
        self.view.setGeometry(0, 0, 1000, 750)


    def createToolbar(self):
        self.toolbar = QToolBar("Toolbar")
        self.addToolBar(Qt.BottomToolBarArea, self.toolbar)
        self.toolbar.setFixedHeight(40)
        self.toolbar.setMovable(False)
        
        layout = QHBoxLayout()
        layout.setSpacing(48)
        self.toolbar.setLayout(layout)

        pixLabel = QLabel()
        pix = QPixmap("data/geeks.png")
        pixLabel.setPixmap(pix)

        gradientColorBar = QGraphicsRectItem(0, 0, 4000, 4000)
        gradientColorBar.setBrush(QBrush(self.createGradientColor()))
        toolbarScene = QGraphicsScene()
        toolbarView = QGraphicsView(toolbarScene, self.toolbar)
        toolbarView.setGeometry(0, 0, 40, 10)
        #toolbarScene.addItem(gradientColorBar)
        self.toolbar.addWidget(pixLabel)
        toolbarView.show()
        #self.toolbar.addWidget(toolbarView)

        self.slider = QSlider(orientation=Qt.Orientation.Horizontal)
        self.slider.setMaximumSize(400, 30)
        self.slider.setMaximum(100)
        self.slider_label = QLabel("0")
        self.slider.valueChanged.connect(self.sliderChangeHandler)
        self.toolbar.addWidget(self.slider)
        self.toolbar.addWidget(self.slider_label)

        self.toolbar.addSeparator()

        self.selected_airport_label = QLabel(NO_AIRPORT_SELECTED_LABEL)
        self.toolbar.addWidget(self.selected_airport_label)
        

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """
        if event.key() == Qt.Key_X:
            self.fdeb.step_size /= 2
            for edge in self.edges:
                edge.add_subdivisions()

            self.fdeb.k = self.fdeb.K / (len(self.edges[0].subdivision_points) + 1)

            print("Added subdivisions, new count:", len(self.edges[0].subdivision_points))
            print("Step size:", self.fdeb.step_size, ", k=", self.fdeb.k)

            self.checkNeighbourCorrectness()

            return super().keyPressEvent(event)

        self.fdeb.iteration_step(self.step)
        self.step += 1
        self.updateEdgePaths()
        print("step:", self.step)
        print("PathItems count:", len(self.pathItems))
        print("Edges count:", len(self.edges))
        print("k={}".format(self.fdeb.k), ", {} subdivision points".format(len(self.edges[0].subdivision_points)))
        """
        return super().keyPressEvent(event)

    def selectionChangedHandler(self, node: Node):
        if node is None:
            self.selected_airport_label.setText(NO_AIRPORT_SELECTED_LABEL)
        else:
            self.selected_airport_label.setText("({}) {}, size: {}".format(node.code, node.name, node.size))

    def sliderChangeHandler(self, slider_value: float):
        self.slider_label.setText("{}".format(slider_value))
        self.fdeb_interpolation.update_positions(self.K_max - self.K_max * slider_value / 100)
        self.updateEdgePaths()

    def updateEdgePaths(self):
        if not self.edges:
            return

        start = time.time()
        for idx, edge in enumerate(self.edges):
            path = self.generateEdgePath(edge)
            self.pathItems[idx].setPath(path)
        end = time.time()

        # print("Updating edges took {}s".format(end - start))

    def checkNeighbourCorrectness(self):
        mistakes_count = 0
        for edge in self.edges:
            current_point = edge.first_subdivision_point
            if current_point.previous_neighbour != edge.source:
                mistakes_count += 1
            while current_point.next_neighbour != edge.target:
                if current_point.next_neighbour.previous_neighbour != current_point: 
                    mistakes_count += 1
                current_point = current_point.next_neighbour

        if mistakes_count > 0:
            print("Number of incorrectly linked points:", mistakes_count)


    def createEdgesPath(self):
        if not self.edges:
            return
        
        color = self.scene.edgeColor
        pen = QPen(color)

        for edge in self.edges:
            path = self.generateEdgePath(edge)
            pathItem = self.scene.addPath(path)
            pathItem.setPen(pen)

            pathItem.setData(Property.ObjectType, ObjectType.Edge)
            pathItem.setData(Property.Edge, edge)

            self.pathItems.append(pathItem)
            self.scene.addEdge(edge.id, pathItem)

    def generateNodeColor(self):
        # TODO: Instead of using the concrete degree value of the node,
        # map it's color based on the ordered position of its degree value
        maxSize = max(np.log(node.size) for node in self.nodes)
        minSize = min(np.log(node.size) for node in self.nodes)
        assignedColors = {}
        for node in self.nodes:
            x = 1/(maxSize - minSize)
            l = 0.3 + 0.7 * x * np.log(node.size) - minSize
            h = ((np.log(node.size) - minSize)/(maxSize-minSize))/4
            #l = 0.5
            #h = ((((np.log(node.size) - minSize) / (maxSize - minSize)) / 2) + 0.67) % 1
            color = QColor()
            color.setHslF(h, 0.75, l, 1)
            assignedColors[node] = color

        return assignedColors

    def createGradientColor(self):
        gradient = QLinearGradient()
        gradient.setColorAt(0.0, QColor.fromHslF(0, 0.75, 0, 1))
        gradient.setColorAt(1.0, QColor.fromHslF(1/4, 0.75, 1, 1))
        gradient.setCoordinateMode(QGradient.ObjectMode)
        return gradient
    
    def calculateDegree(self, graph):
        degrees = [0] * len(graph.nodes())
        processed_edges = []
        for edge in graph.edges():
            if (edge[0], edge[1]) not in processed_edges and (edge[1], edge[0]) not in processed_edges:
                degrees[int(edge[0])] += 1
                degrees[int(edge[1])] += 1
                processed_edges.append((edge[0], edge[1]))
        
        return degrees

    def loadGraph(self, input_file_path, airports_file_path):
        graph = nx.read_graphml(input_file_path)
        airports = pd.read_csv(airports_file_path)
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

        for node_id, _node in graph.nodes(data=True):
            if int(node_id) >= NUM:
                continue

            airport_code = _node["tooltip"][:3]

            if (airports["iata_code"] == airport_code).sum() == 0:
                print("Missing", airport_code)
            airport_name = str(airports.loc[airports['iata_code'] == airport_code]['name'].values[0])


            node = Node(id=int(node_id), size=degrees[int(node_id)],
                        code=airport_code, name=airport_name,
                        x=float(_node["x"]), y=float(_node["y"]))

            min_x = min(min_x, node.x)
            min_y = min(min_y, node.y)
            max_x = max(max_x, node.x)
            max_y = max(max_y, node.y)

            max_size = max(max_size, graph.degree[node_id])
            min_size = min(min_size, graph.degree[node_id])

            self.nodes[int(node_id)] = node

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

        # print("Total number of edges:", len(added))

    def loadTopology(self, input_file_path):
        with open(input_file_path, "r") as f:
            topology = geojson.load(f)
            self.topology = topology

    def generateMap(self):

        # mapBrush = QBrush(QColor("#FFF2AF"))
        IGNORED_STATES = ["Alaska", "Hawaii"]

        stateBrush = QBrush(GOOGLE_COLORS["green"])
        statePen = QPen(GOOGLE_COLORS["darkGreen"])

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
                item = self.scene.addPolygon(polygon, pen=statePen, brush=stateBrush)
            elif feature["geometry"]["type"] == "MultiPolygon":
                for geometry in feature["geometry"]["coordinates"]:
                    polygon = generatePolygon(geometry[0])
                    item = self.scene.addPolygon(polygon, pen=statePen, brush=stateBrush)

            item.setData(Property.ObjectType, ObjectType.State)
        

    def generateGraph(self):
        self.createEdgesPath()
        assignedColors = self.generateNodeColor()
        for node in self.nodes:
            nodeItem = self.scene.addEllipse(node.x, node.y, self.RADIUS, self.RADIUS, self.scene.nodePen, assignedColors[node])
            nodeItem.setData(Property.ObjectType, ObjectType.Node)
            nodeItem.setData(Property.Node, node)
            nodeItem.setZValue(NODES_Z_VALUE)


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
    parser.add_argument("--airport-names", "-a", type=str, required=False, default="data/airports-large-edited.csv")
    parser.add_argument("--compatibility", "-c", type=str, required=False, default="data/compatibility-measures.npy")
    parser.add_argument("--precomputed", "-p", type=str, required=False, default="precomputed/positions_new")
    parser.add_argument("--number", "-n", type=int, required=False, default=300)
    parser.add_argument("--interpolation", "-i", type=bool, required=False, default=True)
    args = parser.parse_args()

    app = QApplication(sys.argv)
    ex = MainWindow(args.graph, args.map, args.airport_names, args.compatibility, args.number, args.precomputed)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
