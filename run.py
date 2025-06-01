import sys
import random
import copy
from time import sleep
from enum import IntEnum
from PyQt5.QtWidgets import QLabel, QWidget, QApplication, QGridLayout, QMessageBox, QLineEdit, QMainWindow, QPushButton, QComboBox, QFrame
from PyQt5.QtGui import QFont, QPalette
from PyQt5.QtCore import Qt
from PyQt5 import QtCore, QtGui, QtWidgets
from numpy import empty
from ui_object.Block import Block
from algorithm.bfs.bfs import BFSAgent
from algorithm.ids.ids import IDSAgent
from algorithm.A_asterisk.A_asterisk import AASTERISK, AASTERISKMisTiles, AASTERISKWeighMHT, GreedyBestFirstSearch, AASTERISKLinearConflict, GreedyLinearConflict
import math
import heapq
import time
import numpy as np
import tensorflow as tf

# Using enumeration class to represent direction.
class Direction(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

class NPuzzleSolver:
    def __init__(self, size=5, goal_state=None, model_path='C:/Users/DUYEN/OneDrive/Documents/GitHub/NhapmonAI-Project/model/keras-1024-1024-512-128-64-amse08.h5'):
        self.size = size
        self.n = size * size
        if goal_state is None:
            self.goal_state = [i % self.n for i in range(1, self.n + 1)]
        else:
            self.goal_state = goal_state

        # Load Keras model
        self.model = tf.keras.models.load_model(
            model_path,
            compile=False
        )

        # Validate model input size (assuming model expects 4x4 puzzle: 16 * 16 = 256)
        expected_size = 4  # Model trained for 4x4
        expected_input_size = (expected_size * expected_size) ** 2  # e.g., 256 for 4x4
        if self.n != expected_size * expected_size:
            raise ValueError(f"Model expects a {expected_size}x{expected_size} puzzle (input size {expected_input_size}), but got {size}x{size} (input size {self.n * self.n})")

        self.heuristic_count = 0

    def heuristic(self, state):
        self.heuristic_count += 1
        flat_state = [num for row in state for num in row]
        x_encoded = np.eye(self.n)[flat_state].ravel()
        output = self.model.predict(x_encoded.reshape(1, -1), verbose=0)
        estimated_cost = max(output.item(), 0)
        return estimated_cost

    def find_zero(self, state):
        for i in range(self.size):
            for j in range(self.size):
                if state[i][j] == 0:
                    return i, j

    def get_neighbors(self, state):
        neighbors = []
        x, y = self.find_zero(state)
        directions = [(-1,0), (1,0), (0,-1), (0,1)]
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.size and 0 <= ny < self.size:
                new_state = [row[:] for row in state]
                new_state[x][y], new_state[nx][ny] = new_state[nx][ny], new_state[x][y]
                neighbors.append(new_state)
        return neighbors

    def state_to_tuple(self, state):
        return tuple(tuple(row) for row in state)

    def is_goal(self, state):
        flat = [num for row in state for num in row]
        return flat == self.goal_state

    def solve(self, start_state):
        start_time = time.time()
        pq = []
        visited = set()
        heapq.heappush(pq, (self.heuristic(start_state), 0, start_state, []))  # (f, g, state, path)

        while pq:
            f, g, current, path = heapq.heappop(pq)
            state_key = self.state_to_tuple(current)

            if self.is_goal(current):
                total_time = time.time() - start_time
                moves = []
                prev_state = path[0] if path else start_state
                for state in path[1:] + [current]:
                    px, py = self.find_zero(prev_state)
                    cx, cy = self.find_zero(state)
                    if px > cx:
                        moves.append('U')
                    elif px < cx:
                        moves.append('D')
                    elif py > cy:
                        moves.append('L')
                    elif py < cy:
                        moves.append('R')
                    prev_state = state
                return total_time, g, moves

            if state_key in visited:
                continue
            visited.add(state_key)

            for neighbor in self.get_neighbors(current):
                if self.state_to_tuple(neighbor) not in visited:
                    h = self.heuristic(neighbor)
                    heapq.heappush(pq, (g + 1 + h, g + 1, neighbor, path + [current]))

        return None, None, []

class NumberNPuzzle(QMainWindow):
    """ N-puzzle main program """
    def __init__(self):
        super(NumberNPuzzle, self).__init__()
        self.blocks = []
        self.num_suffle = 100
        self.zero_row = 0
        self.zero_column = 0
        self.num_row = 5
        self.way = list()
        self.start_blocks = [[6,1,4,9,3],[11,2,7,10,0],[16,12,20,13,5],[17,18,8,19,15], [21,22,23,24,14]]
        self.gltMain = QGridLayout()
        self.initUI()

    def initUI(self):      
        self.setObjectName("Main")
        self.resize(1030, 963)  # Increased height to accommodate new button
        self.gltMain.setSpacing(20)
        
        self.widget = QtWidgets.QWidget(self)
        self.widget.setGeometry(QtCore.QRect(0, 50, 720, 720))
        self.widget.setObjectName("widget")
        self.widget.setLayout(self.gltMain)
        self.gltMain.setSpacing(20)
        self.widget.setLayout(self.gltMain)
        self.setWindowTitle('N-puzzle Game.')
        self.widget.setStyleSheet("background-color:gray;")

        self.pushButton_1 = QtWidgets.QPushButton(self)
        self.pushButton_1.setGeometry(QtCore.QRect(730, 50, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_1.setFont(font)
        self.pushButton_1.setObjectName("pushButton_1")
        self.pushButton_2 = QtWidgets.QPushButton(self)
        self.pushButton_2.setGeometry(QtCore.QRect(730, 140, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_2.setFont(font)
        self.pushButton_2.setObjectName("pushButton_2")
        self.pushButton_3 = QtWidgets.QPushButton(self)
        self.pushButton_3.setGeometry(QtCore.QRect(730, 230, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setObjectName("pushButton_3")
        self.pushButton_4 = QtWidgets.QPushButton(self)
        self.pushButton_4.setGeometry(QtCore.QRect(730, 320, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.pushButton_5 = QtWidgets.QPushButton(self)
        self.pushButton_5.setGeometry(QtCore.QRect(730, 410, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.pushButton_6 = QtWidgets.QPushButton(self)
        self.pushButton_6.setGeometry(QtCore.QRect(730, 500, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.pushButton_7 = QtWidgets.QPushButton(self)
        self.pushButton_7.setGeometry(QtCore.QRect(730, 590, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.pushButton_8 = QtWidgets.QPushButton(self)
        self.pushButton_8.setGeometry(QtCore.QRect(730, 680, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.pushButton_9 = QtWidgets.QPushButton(self)
        self.pushButton_9.setGeometry(QtCore.QRect(730, 770, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_9.setFont(font)
        self.pushButton_9.setObjectName("pushButton_9")
        self.pushButton_10 = QtWidgets.QPushButton(self)
        self.pushButton_10.setGeometry(QtCore.QRect(730, 860, 281, 31))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_10.setFont(font)
        self.pushButton_10.setObjectName("pushButton_10")
        self.labelCombobox = QtWidgets.QLabel(self)
        self.labelCombobox.setGeometry(QtCore.QRect(20, 10, 200, 21))
        self.labelShuffle = QtWidgets.QLabel(self)
        self.labelShuffle.setGeometry(QtCore.QRect(500, 10, 200, 21))
        self.textShuffle = QLineEdit(self)
        self.textShuffle.move(590, 10)
        self.textShuffle.resize(50,21)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.labelCombobox.setFont(font)
        self.labelCombobox.setObjectName("labelCombobox")
        self.labelShuffle.setFont(font)
        self.labelShuffle.setObjectName("labelShuffle")
        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.setGeometry(QtCore.QRect(196, 12, 69, 22))
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.resetBtn = QtWidgets.QPushButton(self)
        self.resetBtn.setGeometry(QtCore.QRect(830, 10, 171, 31))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.resetBtn.setFont(font)
        self.resetBtn.setObjectName("resetBtn")
        self.time_1 = QtWidgets.QLabel(self)
        self.time_1.setGeometry(QtCore.QRect(730, 90, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_1.setFont(font)
        self.time_1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_1.setWordWrap(True)
        self.time_1.setObjectName("time_1")
        self.num_of_steps_1 = QtWidgets.QLabel(self)
        self.num_of_steps_1.setGeometry(QtCore.QRect(730, 110, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_1.setFont(font)
        self.num_of_steps_1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_1.setWordWrap(True)
        self.num_of_steps_1.setObjectName("num_of_steps_1")
        self.time_value_1 = QtWidgets.QLabel(self)
        self.time_value_1.setGeometry(QtCore.QRect(870, 90, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_1.setFont(font)
        self.time_value_1.setText("")
        self.time_value_1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_1.setWordWrap(True)
        self.time_value_1.setObjectName("time_value_1")
        self.num_of_steps_value_1 = QtWidgets.QLabel(self)
        self.num_of_steps_value_1.setGeometry(QtCore.QRect(870, 110, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_1.setFont(font)
        self.num_of_steps_value_1.setText("")
        self.num_of_steps_value_1.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_1.setWordWrap(True)
        self.num_of_steps_value_1.setObjectName("num_of_steps_value_1")
        self.line = QtWidgets.QFrame(self)
        self.line.setGeometry(QtCore.QRect(730, 80, 281, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.line_2 = QtWidgets.QFrame(self)
        self.line_2.setGeometry(QtCore.QRect(730, 121, 281, 16))
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.line_3 = QtWidgets.QFrame(self)
        self.line_3.setGeometry(QtCore.QRect(730, 88, 3, 40))
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.line_4 = QtWidgets.QFrame(self)
        self.line_4.setGeometry(QtCore.QRect(1010, 90, 3, 40))
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.line_5 = QtWidgets.QFrame(self)
        self.line_5.setGeometry(QtCore.QRect(730, 178, 3, 40))
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.line_6 = QtWidgets.QFrame(self)
        self.line_6.setGeometry(QtCore.QRect(1010, 180, 3, 40))
        self.line_6.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.line_7 = QtWidgets.QFrame(self)
        self.line_7.setGeometry(QtCore.QRect(730, 170, 281, 16))
        self.line_7.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.num_of_steps_2 = QtWidgets.QLabel(self)
        self.num_of_steps_2.setGeometry(QtCore.QRect(730, 200, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_2.setFont(font)
        self.num_of_steps_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_2.setWordWrap(True)
        self.num_of_steps_2.setObjectName("num_of_steps_2")
        self.time_2 = QtWidgets.QLabel(self)
        self.time_2.setGeometry(QtCore.QRect(730, 180, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_2.setFont(font)
        self.time_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_2.setWordWrap(True)
        self.time_2.setObjectName("time_2")
        self.line_8 = QtWidgets.QFrame(self)
        self.line_8.setGeometry(QtCore.QRect(730, 211, 281, 16))
        self.line_8.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.time_value_2 = QtWidgets.QLabel(self)
        self.time_value_2.setGeometry(QtCore.QRect(870, 180, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_2.setFont(font)
        self.time_value_2.setText("")
        self.time_value_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_2.setWordWrap(True)
        self.time_value_2.setObjectName("time_value_2")
        self.num_of_steps_value_2 = QtWidgets.QLabel(self)
        self.num_of_steps_value_2.setGeometry(QtCore.QRect(870, 200, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_2.setFont(font)
        self.num_of_steps_value_2.setText("")
        self.num_of_steps_value_2.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_2.setWordWrap(True)
        self.num_of_steps_value_2.setObjectName("num_of_steps_value_2")
        self.line_9 = QtWidgets.QFrame(self)
        self.line_9.setGeometry(QtCore.QRect(730, 268, 3, 40))
        self.line_9.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.line_10 = QtWidgets.QFrame(self)
        self.line_10.setGeometry(QtCore.QRect(1010, 270, 3, 40))
        self.line_10.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.line_11 = QtWidgets.QFrame(self)
        self.line_11.setGeometry(QtCore.QRect(730, 260, 281, 16))
        self.line_11.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.num_of_steps_3 = QtWidgets.QLabel(self)
        self.num_of_steps_3.setGeometry(QtCore.QRect(730, 290, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_3.setFont(font)
        self.num_of_steps_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_3.setWordWrap(True)
        self.num_of_steps_3.setObjectName("num_of_steps_3")
        self.time_3 = QtWidgets.QLabel(self)
        self.time_3.setGeometry(QtCore.QRect(730, 270, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_3.setFont(font)
        self.time_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_3.setWordWrap(True)
        self.time_3.setObjectName("time_3")
        self.line_12 = QtWidgets.QFrame(self)
        self.line_12.setGeometry(QtCore.QRect(730, 301, 281, 16))
        self.line_12.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.time_value_3 = QtWidgets.QLabel(self)
        self.time_value_3.setGeometry(QtCore.QRect(870, 270, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_3.setFont(font)
        self.time_value_3.setText("")
        self.time_value_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_3.setWordWrap(True)
        self.time_value_3.setObjectName("time_value_3")
        self.num_of_steps_value_3 = QtWidgets.QLabel(self)
        self.num_of_steps_value_3.setGeometry(QtCore.QRect(870, 290, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_3.setFont(font)
        self.num_of_steps_value_3.setText("")
        self.num_of_steps_value_3.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_3.setWordWrap(True)
        self.num_of_steps_value_3.setObjectName("num_of_steps_value_3")
        self.line_13 = QtWidgets.QFrame(self)
        self.line_13.setGeometry(QtCore.QRect(730, 358, 3, 40))
        self.line_13.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.line_14 = QtWidgets.QFrame(self)
        self.line_14.setGeometry(QtCore.QRect(1010, 360, 3, 40))
        self.line_14.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.line_15 = QtWidgets.QFrame(self)
        self.line_15.setGeometry(QtCore.QRect(730, 350, 281, 16))
        self.line_15.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.num_of_steps_4 = QtWidgets.QLabel(self)
        self.num_of_steps_4.setGeometry(QtCore.QRect(730, 380, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_4.setFont(font)
        self.num_of_steps_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_4.setWordWrap(True)
        self.num_of_steps_4.setObjectName("num_of_steps_4")
        self.time_4 = QtWidgets.QLabel(self)
        self.time_4.setGeometry(QtCore.QRect(730, 360, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_4.setFont(font)
        self.time_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_4.setWordWrap(True)
        self.time_4.setObjectName("time_4")
        self.line_16 = QtWidgets.QFrame(self)
        self.line_16.setGeometry(QtCore.QRect(730, 391, 281, 16))
        self.line_16.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_16.setObjectName("line_16")
        self.time_value_4 = QtWidgets.QLabel(self)
        self.time_value_4.setGeometry(QtCore.QRect(870, 360, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_4.setFont(font)
        self.time_value_4.setText("")
        self.time_value_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_4.setWordWrap(True)
        self.time_value_4.setObjectName("time_value_4")
        self.num_of_steps_value_4 = QtWidgets.QLabel(self)
        self.num_of_steps_value_4.setGeometry(QtCore.QRect(870, 380, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_4.setFont(font)
        self.num_of_steps_value_4.setText("")
        self.num_of_steps_value_4.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_4.setWordWrap(True)
        self.num_of_steps_value_4.setObjectName("num_of_steps_value_4")
        self.line_17 = QtWidgets.QFrame(self)
        self.line_17.setGeometry(QtCore.QRect(730, 448, 3, 40))
        self.line_17.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_17.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_17.setObjectName("line_17")
        self.line_18 = QtWidgets.QFrame(self)
        self.line_18.setGeometry(QtCore.QRect(1010, 450, 3, 40))
        self.line_18.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_18.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_18.setObjectName("line_18")
        self.line_19 = QtWidgets.QFrame(self)
        self.line_19.setGeometry(QtCore.QRect(730, 440, 281, 16))
        self.line_19.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_19.setObjectName("line_19")
        self.num_of_steps_5 = QtWidgets.QLabel(self)
        self.num_of_steps_5.setGeometry(QtCore.QRect(730, 470, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_5.setFont(font)
        self.num_of_steps_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_5.setWordWrap(True)
        self.num_of_steps_5.setObjectName("num_of_steps_5")
        self.time_5 = QtWidgets.QLabel(self)
        self.time_5.setGeometry(QtCore.QRect(730, 450, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_5.setFont(font)
        self.time_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_5.setWordWrap(True)
        self.time_5.setObjectName("time_5")
        self.line_20 = QtWidgets.QFrame(self)
        self.line_20.setGeometry(QtCore.QRect(730, 481, 281, 16))
        self.line_20.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_20.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_20.setObjectName("line_20")
        self.time_value_5 = QtWidgets.QLabel(self)
        self.time_value_5.setGeometry(QtCore.QRect(870, 450, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_5.setFont(font)
        self.time_value_5.setText("")
        self.time_value_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_5.setWordWrap(True)
        self.time_value_5.setObjectName("time_value_5")
        self.num_of_steps_value_5 = QtWidgets.QLabel(self)
        self.num_of_steps_value_5.setGeometry(QtCore.QRect(870, 470, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_5.setFont(font)
        self.num_of_steps_value_5.setText("")
        self.num_of_steps_value_5.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_5.setWordWrap(True)
        self.num_of_steps_value_5.setObjectName("num_of_steps_value_5")
        self.line_21 = QtWidgets.QFrame(self)
        self.line_21.setGeometry(QtCore.QRect(730, 538, 3, 40))
        self.line_21.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_21.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_21.setObjectName("line_21")
        self.line_22 = QtWidgets.QFrame(self)
        self.line_22.setGeometry(QtCore.QRect(1010, 540, 3, 40))
        self.line_22.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_22.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_22.setObjectName("line_22")
        self.line_23 = QtWidgets.QFrame(self)
        self.line_23.setGeometry(QtCore.QRect(730, 530, 281, 16))
        self.line_23.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_23.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_23.setObjectName("line_23")
        self.num_of_steps_6 = QtWidgets.QLabel(self)
        self.num_of_steps_6.setGeometry(QtCore.QRect(730, 560, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_6.setFont(font)
        self.num_of_steps_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_6.setWordWrap(True)
        self.num_of_steps_6.setObjectName("num_of_steps_6")
        self.time_6 = QtWidgets.QLabel(self)
        self.time_6.setGeometry(QtCore.QRect(730, 540, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_6.setFont(font)
        self.time_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_6.setWordWrap(True)
        self.time_6.setObjectName("time_6")
        self.line_24 = QtWidgets.QFrame(self)
        self.line_24.setGeometry(QtCore.QRect(730, 571, 281, 16))
        self.line_24.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_24.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_24.setObjectName("line_24")
        self.time_value_6 = QtWidgets.QLabel(self)
        self.time_value_6.setGeometry(QtCore.QRect(870, 540, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_6.setFont(font)
        self.time_value_6.setText("")
        self.time_value_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_6.setWordWrap(True)
        self.time_value_6.setObjectName("time_value_6")
        self.num_of_steps_value_6 = QtWidgets.QLabel(self)
        self.num_of_steps_value_6.setGeometry(QtCore.QRect(870, 560, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_6.setFont(font)
        self.num_of_steps_value_6.setText("")
        self.num_of_steps_value_6.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_6.setWordWrap(True)
        self.num_of_steps_value_6.setObjectName("num_of_steps_value_6")
        self.line_25 = QtWidgets.QFrame(self)
        self.line_25.setGeometry(QtCore.QRect(730, 628, 3, 40))
        self.line_25.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_25.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_25.setObjectName("line_25")
        self.line_26 = QtWidgets.QFrame(self)
        self.line_26.setGeometry(QtCore.QRect(1010, 630, 3, 40))
        self.line_26.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_26.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_26.setObjectName("line_26")
        self.line_27 = QtWidgets.QFrame(self)
        self.line_27.setGeometry(QtCore.QRect(730, 620, 281, 16))
        self.line_27.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_27.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_27.setObjectName("line_27")
        self.num_of_steps_7 = QtWidgets.QLabel(self)
        self.num_of_steps_7.setGeometry(QtCore.QRect(730, 650, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_7.setFont(font)
        self.num_of_steps_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_7.setWordWrap(True)
        self.num_of_steps_7.setObjectName("num_of_steps_7")
        self.time_7 = QtWidgets.QLabel(self)
        self.time_7.setGeometry(QtCore.QRect(730, 630, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_7.setFont(font)
        self.time_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_7.setWordWrap(True)
        self.time_7.setObjectName("time_7")
        self.line_28 = QtWidgets.QFrame(self)
        self.line_28.setGeometry(QtCore.QRect(730, 661, 281, 16))
        self.line_28.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_28.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_28.setObjectName("line_28")
        self.time_value_7 = QtWidgets.QLabel(self)
        self.time_value_7.setGeometry(QtCore.QRect(870, 630, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_7.setFont(font)
        self.time_value_7.setText("")
        self.time_value_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_7.setWordWrap(True)
        self.time_value_7.setObjectName("time_value_7")
        self.num_of_steps_value_7 = QtWidgets.QLabel(self)
        self.num_of_steps_value_7.setGeometry(QtCore.QRect(870, 650, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_7.setFont(font)
        self.num_of_steps_value_7.setText("")
        self.num_of_steps_value_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_7.setWordWrap(True)
        self.num_of_steps_value_7.setObjectName("num_of_steps_value_7")
        self.line_29 = QtWidgets.QFrame(self)
        self.line_29.setGeometry(QtCore.QRect(730, 718, 3, 40))
        self.line_29.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_29.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_29.setObjectName("line_29")
        self.line_30 = QtWidgets.QFrame(self)
        self.line_30.setGeometry(QtCore.QRect(1010, 720, 3, 40))
        self.line_30.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_30.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_30.setObjectName("line_30")
        self.line_31 = QtWidgets.QFrame(self)
        self.line_31.setGeometry(QtCore.QRect(730, 710, 281, 16))
        self.line_31.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_31.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_31.setObjectName("line_31")
        self.num_of_steps_8 = QtWidgets.QLabel(self)
        self.num_of_steps_8.setGeometry(QtCore.QRect(730, 740, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_8.setFont(font)
        self.num_of_steps_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_8.setWordWrap(True)
        self.num_of_steps_8.setObjectName("num_of_steps_8")
        self.time_8 = QtWidgets.QLabel(self)
        self.time_8.setGeometry(QtCore.QRect(730, 720, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_8.setFont(font)
        self.time_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_8.setWordWrap(True)
        self.time_8.setObjectName("time_8")
        self.line_32 = QtWidgets.QFrame(self)
        self.line_32.setGeometry(QtCore.QRect(730, 751, 281, 16))
        self.line_32.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_32.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_32.setObjectName("line_32")
        self.time_value_8 = QtWidgets.QLabel(self)
        self.time_value_8.setGeometry(QtCore.QRect(870, 720, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_8.setFont(font)
        self.time_value_8.setText("")
        self.time_value_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_8.setWordWrap(True)
        self.time_value_8.setObjectName("time_value_8")
        self.num_of_steps_value_8 = QtWidgets.QLabel(self)
        self.num_of_steps_value_8.setGeometry(QtCore.QRect(870, 740, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_8.setFont(font)
        self.num_of_steps_value_8.setText("")
        self.num_of_steps_value_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_8.setWordWrap(True)
        self.num_of_steps_value_8.setObjectName("num_of_steps_value_8")
        self.line_33 = QtWidgets.QFrame(self)
        self.line_33.setGeometry(QtCore.QRect(730, 808, 3, 40))
        self.line_33.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_33.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_33.setObjectName("line_33")
        self.line_34 = QtWidgets.QFrame(self)
        self.line_34.setGeometry(QtCore.QRect(1010, 810, 3, 40))
        self.line_34.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_34.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_34.setObjectName("line_34")
        self.line_35 = QtWidgets.QFrame(self)
        self.line_35.setGeometry(QtCore.QRect(730, 800, 281, 16))
        self.line_35.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_35.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_35.setObjectName("line_35")
        self.num_of_steps_9 = QtWidgets.QLabel(self)
        self.num_of_steps_9.setGeometry(QtCore.QRect(730, 830, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_9.setFont(font)
        self.num_of_steps_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_9.setWordWrap(True)
        self.num_of_steps_9.setObjectName("num_of_steps_9")
        self.time_9 = QtWidgets.QLabel(self)
        self.time_9.setGeometry(QtCore.QRect(730, 810, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_9.setFont(font)
        self.time_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_9.setWordWrap(True)
        self.time_9.setObjectName("time_9")
        self.line_36 = QtWidgets.QFrame(self)
        self.line_36.setGeometry(QtCore.QRect(730, 841, 281, 16))
        self.line_36.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_36.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_36.setObjectName("line_36")
        self.time_value_9 = QtWidgets.QLabel(self)
        self.time_value_9.setGeometry(QtCore.QRect(870, 810, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_9.setFont(font)
        self.time_value_9.setText("")
        self.time_value_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_9.setWordWrap(True)
        self.time_value_9.setObjectName("time_value_9")
        self.num_of_steps_value_9 = QtWidgets.QLabel(self)
        self.num_of_steps_value_9.setGeometry(QtCore.QRect(870, 830, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_9.setFont(font)
        self.num_of_steps_value_9.setText("")
        self.num_of_steps_value_9.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_9.setWordWrap(True)
        self.num_of_steps_value_9.setObjectName("num_of_steps_value_9")
        self.line_37 = QtWidgets.QFrame(self)
        self.line_37.setGeometry(QtCore.QRect(730, 898, 3, 40))
        self.line_37.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_37.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_37.setObjectName("line_37")
        self.line_38 = QtWidgets.QFrame(self)
        self.line_38.setGeometry(QtCore.QRect(1010, 900, 3, 40))
        self.line_38.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_38.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_38.setObjectName("line_38")
        self.line_39 = QtWidgets.QFrame(self)
        self.line_39.setGeometry(QtCore.QRect(730, 890, 281, 16))
        self.line_39.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_39.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_39.setObjectName("line_39")
        self.num_of_steps_10 = QtWidgets.QLabel(self)
        self.num_of_steps_10.setGeometry(QtCore.QRect(730, 920, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.num_of_steps_10.setFont(font)
        self.num_of_steps_10.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_10.setWordWrap(True)
        self.num_of_steps_10.setObjectName("num_of_steps_10")
        self.time_10 = QtWidgets.QLabel(self)
        self.time_10.setGeometry(QtCore.QRect(730, 900, 201, 16))
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.time_10.setFont(font)
        self.time_10.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_10.setWordWrap(True)
        self.time_10.setObjectName("time_10")
        self.line_40 = QtWidgets.QFrame(self)
        self.line_40.setGeometry(QtCore.QRect(730, 931, 281, 16))
        self.line_40.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_40.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_40.setObjectName("line_40")
        self.time_value_10 = QtWidgets.QLabel(self)
        self.time_value_10.setGeometry(QtCore.QRect(870, 900, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.time_value_10.setFont(font)
        self.time_value_10.setText("")
        self.time_value_10.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.time_value_10.setWordWrap(True)
        self.time_value_10.setObjectName("time_value_10")
        self.num_of_steps_value_10 = QtWidgets.QLabel(self)
        self.num_of_steps_value_10.setGeometry(QtCore.QRect(870, 920, 201, 16))
        font = QtGui.QFont()
        font.setBold(False)
        font.setWeight(50)
        self.num_of_steps_value_10.setFont(font)
        self.num_of_steps_value_10.setText("")
        self.num_of_steps_value_10.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignTop)
        self.num_of_steps_value_10.setWordWrap(True)
        self.num_of_steps_value_10.setObjectName("num_of_steps_value_10")

        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

        self.onInit()
        self.show()

    def reset(self):
        """ Reset the puzzle to a new initial state """
        if self.textShuffle.text():
            try:
                self.num_suffle = int(self.textShuffle.text())
            except ValueError:
                self.num_suffle = 100
        
        self.num_row = int(self.comboBox.currentText())
        
        for i in reversed(range(self.gltMain.count())):
            widget = self.gltMain.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        
        self.blocks = []
        self.start_blocks = []
        self.way = []
        
        for i in range(1, 11):  # Updated to include ANN2
            getattr(self, f"time_value_{i}").setText("")
            getattr(self, f"num_of_steps_value_{i}").setText("")
        
        self.onInit()
        self.start_blocks = copy.deepcopy(self.blocks)
        
    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "N-puzzle Game"))

        def BFS():
            cells = [x for xs in self.blocks for x in xs]
            bfs = BFSAgent(cells, math.isqrt(len(cells)))
            time, num_steps, path = bfs.findMinimumSteps()
            reversed_path = []
            reverse_dir = {"R": "L", "L": "R", "U": "D", "D": "U"}
            for step in path:
                reversed_step = reverse_dir.get(step, step)  
                reversed_path.append(reversed_step)
            path = reversed_path
            print(f"BFS: num_steps={num_steps}, path_length={len(path)}, path={path}")
            a = str(round(time, 5))
            self.time_value_1.setText(_translate("Form", str(a)))
            b = str(num_steps)
            self.num_of_steps_value_1.setText(_translate("Form", b))
            self.way = path
            if path:
                self.simulatePath(path)

        def IDS():
            cells = [x for xs in self.blocks for x in xs]
            ids = IDSAgent(cells, math.isqrt(len(cells)))
            time, num_steps, path = ids.findMinimumSteps()
            reversed_path = []
            reverse_dir = {"R": "L", "L": "R", "U": "D", "D": "U"}
            for step in path:
                reversed_step = reverse_dir.get(step, step)  
                reversed_path.append(reversed_step)
            path = reversed_path
            print(f"IDS: num_steps={num_steps}, path_length={len(path)}, path={path}")
            a = str(round(time, 5))
            self.time_value_2.setText(_translate("Form", str(a)))
            b = str(num_steps)
            self.num_of_steps_value_2.setText(_translate("Form", b))
            self.way = path
            if path:
                self.simulatePath(path)

        def Greedy():
            agent = GreedyBestFirstSearch(self.blocks, len(self.blocks[0]))
            time, num_steps, path = agent.findMinimumSteps()
            print(f"Greedy: num_steps={num_steps}, path_length={len(path)}, path={path}")
            a = str(round(time, 5))
            self.time_value_3.setText(_translate("Form", str(a)))
            b = str(num_steps)
            self.num_of_steps_value_3.setText(_translate("Form", b))
            self.way = path
            if path:
                self.simulatePath(path)

        def AStarMT():
            a_star = AASTERISKMisTiles(self.blocks, len(self.blocks[0]))
            time, num_steps, path = a_star.findMinimumSteps()
            print(f"AStarMT: num_steps={num_steps}, path_length={len(path)}, path={path}")
            a = str(round(time, 5))
            self.time_value_4.setText(_translate("Form", str(a)))
            b = str(num_steps)
            self.num_of_steps_value_4.setText(_translate("Form", b))
            self.way = path
            if path:
                self.simulatePath(path)

        def AStarMHT():
            a_star = AASTERISK(self.blocks, len(self.blocks[0]))
            time, num_steps, path = a_star.findMinimumSteps()
            print(f"AStarMHT: num_steps={num_steps}, path_length={len(path)}, path={path}")
            a = str(round(time, 5))
            self.time_value_5.setText(_translate("Form", str(a)))
            b = str(num_steps)
            self.num_of_steps_value_5.setText(_translate("Form", b))
            self.way = path
            if path:
                self.simulatePath(path)

        def AStarWMHT():
            a_star = AASTERISKWeighMHT(self.blocks, len(self.blocks[0]))
            time, num_steps, path = a_star.findMinimumSteps()
            print(f"AStarWMHT: num_steps={num_steps}, path_length={len(path)}, path={path}")
            a = str(round(time, 5))
            self.time_value_6.setText(_translate("Form", str(a)))
            b = str(num_steps)
            self.num_of_steps_value_6.setText(_translate("Form", b))
            self.way = path
            if path:
                self.simulatePath(path)

        def AStarLC():
            a_star = AASTERISKLinearConflict(self.blocks, len(self.blocks[0]))
            time, num_steps, path = a_star.findMinimumSteps()
            print(f"AStarLC: num_steps={num_steps}, path_length={len(path)}, path={path}")
            a = str(round(time, 5))
            self.time_value_7.setText(_translate("Form", str(a)))
            b = str(num_steps)
            self.num_of_steps_value_7.setText(_translate("Form", b))
            self.way = path
            if path:
                self.simulatePath(path)

        def GreedyLC():
            agent = GreedyLinearConflict(self.blocks, len(self.blocks[0]))
            time, num_steps, path = agent.findMinimumSteps()
            print(f"GreedyLC: num_steps={num_steps}, path_length={len(path)}, path={path}")
            a = str(round(time, 5))
            self.time_value_8.setText(_translate("Form", str(a)))
            b = str(num_steps)
            self.num_of_steps_value_8.setText(_translate("Form", b))
            self.way = path
            if path:
                self.simulatePath(path)

        def ANN():
            if self.num_row != 4:
                QMessageBox.warning(self, "Invalid Puzzle Size", "The ANN solver only supports 4x4 puzzles. Please select 4 rows in the combo box.")
                self.time_value_9.setText(_translate("Form", "N/A"))
                self.num_of_steps_value_9.setText(_translate("Form", "N/A"))
                self.way = []
                return
            try:
                solver = NPuzzleSolver(size=self.num_row, model_path='C:/Users/DUYEN/OneDrive/Documents/GitHub/NhapmonAI-Project/model/keras-1024-1024-512-128-64-amse04.h5')
                time, num_steps, path = solver.solve(self.blocks)

                reversed_path = []
                reverse_dir = {"R": "L", "L": "R", "U": "D", "D": "U"}
                for step in path:
                    reversed_step = reverse_dir.get(step, step)  
                    reversed_path.append(reversed_step)
                path = reversed_path
                
                if time is None or num_steps is None:
                    print("ANN: No solution found")
                    self.time_value_9.setText(_translate("Form", "N/A"))
                    self.num_of_steps_value_9.setText(_translate("Form", "N/A"))
                    self.way = []
                else:
                    print(f"ANN: num_steps={num_steps}, path_length={len(path)}, path={path}")
                    a = str(round(time, 5))
                    self.time_value_9.setText(_translate("Form", str(a)))
                    b = str(num_steps)
                    self.num_of_steps_value_9.setText(_translate("Form", b))
                    self.way = path
                    if path:
                        self.simulatePath(path)
            except ValueError as e:
                QMessageBox.warning(self, "Model Error", str(e))
                self.time_value_9.setText(_translate("Form", "N/A"))
                self.num_of_steps_value_9.setText(_translate("Form", "N/A"))
                self.way = []

        def ANN2():
            if self.num_row != 4:
                QMessageBox.warning(self, "Invalid Puzzle Size", "The ANN2 solver only supports 4x4 puzzles. Please select 4 rows in the combo box.")
                self.time_value_10.setText(_translate("Form", "N/A"))
                self.num_of_steps_value_10.setText(_translate("Form", "N/A"))
                self.way = []
                return
            try:
                solver = NPuzzleSolver(size=self.num_row, model_path='C:/Users/DUYEN/OneDrive/Documents/GitHub/NhapmonAI-Project/model/keras-1024-1024-512-128-64-amse08.h5')
                time, num_steps, path = solver.solve(self.blocks)

                reversed_path = []
                reverse_dir = {"R": "L", "L": "R", "U": "D", "D": "U"}
                for step in path:
                    reversed_step = reverse_dir.get(step, step)  
                    reversed_path.append(reversed_step)
                path = reversed_path
                
                if time is None or num_steps is None:
                    print("ANN2: No solution found")
                    self.time_value_10.setText(_translate("Form", "N/A"))
                    self.num_of_steps_value_10.setText(_translate("Form", "N/A"))
                    self.way = []
                else:
                    print(f"ANN2: num_steps={num_steps}, path_length={len(path)}, path={path}")
                    a = str(round(time, 5))
                    self.time_value_10.setText(_translate("Form", str(a)))
                    b = str(num_steps)
                    self.num_of_steps_value_10.setText(_translate("Form", b))
                    self.way = path
                    if path:
                        self.simulatePath(path)
            except ValueError as e:
                QMessageBox.warning(self, "Model Error", str(e))
                self.time_value_10.setText(_translate("Form", "N/A"))
                self.num_of_steps_value_10.setText(_translate("Form", "N/A"))
                self.way = []

        self.pushButton_1.setText(_translate("Form", "BFS"))
        self.pushButton_1.clicked.connect(BFS)
        self.pushButton_2.setText(_translate("Form", "IDS"))
        self.pushButton_2.clicked.connect(IDS)
        self.pushButton_3.setText(_translate("Form", "Greedy (Manhattan)"))
        self.pushButton_3.clicked.connect(Greedy)
        self.pushButton_4.setText(_translate("Form", "A* (Misplaced Tiles)"))
        self.pushButton_4.clicked.connect(AStarMT)
        self.pushButton_5.setText(_translate("Form", "A* (Manhattan)"))
        self.pushButton_5.clicked.connect(AStarMHT)
        self.pushButton_6.setText(_translate("Form", "A* (Weighted Manhattan)"))
        self.pushButton_6.clicked.connect(AStarWMHT)
        self.pushButton_7.setText(_translate("Form", "A* (Linear Conflict)"))
        self.pushButton_7.clicked.connect(AStarLC)
        self.pushButton_8.setText(_translate("Form", "Greedy (Linear Conflict)"))
        self.pushButton_8.clicked.connect(GreedyLC)
        self.pushButton_9.setText(_translate("Form", "ANN"))
        self.pushButton_9.clicked.connect(ANN)
        self.pushButton_10.setText(_translate("Form", "ANN2"))
        self.pushButton_10.clicked.connect(ANN2)

        self.labelCombobox.setText(_translate("Form", "Number of rows:"))
        self.labelShuffle.setText(_translate("Form", "Shuffle:"))
        self.comboBox.setItemText(0, _translate("Form", "2"))
        self.comboBox.setItemText(1, _translate("Form", "3"))
        self.comboBox.setItemText(2, _translate("Form", "4"))
        self.comboBox.setItemText(3, _translate("Form", "5"))
        self.comboBox.setItemText(4, _translate("Form", "6"))
        self.comboBox.setItemText(5, _translate("Form", "7"))
        self.comboBox.setItemText(6, _translate("Form", "8"))
        self.comboBox.setItemText(7, _translate("Form", "9"))
        self.comboBox.setItemText(8, _translate("Form", "10"))

        self.comboBox.setCurrentText(str(self.num_row))
        self.resetBtn.setText(_translate("Form", "Reset"))

        def reset():
            if self.textShuffle.text():
                try:
                    self.num_suffle = int(self.textShuffle.text())
                except ValueError:
                    self.num_suffle = 100
            
            self.num_row = int(self.comboBox.currentText())
            
            for i in reversed(range(self.gltMain.count())):
                widget = self.gltMain.itemAt(i).widget()
                if widget is not None:
                    widget.setParent(None)
            
            self.blocks = []
            self.start_blocks = []
            self.way = []
            
            for i in range(1, 11):  # Updated to include ANN2
                getattr(self, f"time_value_{i}").setText("")
                getattr(self, f"num_of_steps_value_{i}").setText("")
            
            self.onInit()
            self.start_blocks = copy.deepcopy(self.blocks)
        
        self.resetBtn.clicked.connect(reset)

        self.num_of_steps_1.setText(_translate("Form", "  Number of steps: "))
        self.time_1.setText(_translate("Form", "  Time: "))
        self.num_of_steps_2.setText(_translate("Form", "  Number of steps: "))
        self.time_2.setText(_translate("Form", "  Time: "))
        self.num_of_steps_3.setText(_translate("Form", "  Number of steps: "))
        self.time_3.setText(_translate("Form", "  Time: "))
        self.num_of_steps_4.setText(_translate("Form", "  Number of steps: "))
        self.time_4.setText(_translate("Form", "  Time: "))
        self.num_of_steps_5.setText(_translate("Form", "  Number of steps: "))
        self.time_5.setText(_translate("Form", "  Time: "))
        self.num_of_steps_6.setText(_translate("Form", "  Number of steps: "))
        self.time_6.setText(_translate("Form", "  Time: "))
        self.num_of_steps_7.setText(_translate("Form", "  Number of steps: "))
        self.time_7.setText(_translate("Form", "  Time: "))
        self.num_of_steps_8.setText(_translate("Form", "  Number of steps: "))
        self.time_8.setText(_translate("Form", "  Time: "))
        self.num_of_steps_9.setText(_translate("Form", "  Number of steps: "))
        self.time_9.setText(_translate("Form", "  Time: "))
        self.num_of_steps_10.setText(_translate("Form", "  Number of steps: "))
        self.time_10.setText(_translate("Form", "  Time: "))

    def onInit(self):
        self.numbers = list(range(1, self.num_row * self.num_row))
        self.numbers.append(0)
        self.blocks = []
        for row in range(self.num_row):
            self.blocks.append([])
            for column in range(self.num_row):
                temp = self.numbers[row * self.num_row + column]
                if temp == 0:
                    self.zero_row = row
                    self.zero_column = column
                self.blocks[row].append(temp)
        if self.start_blocks and len(self.start_blocks) == self.num_row and all(len(row) == self.num_row for row in self.start_blocks):
            flat_start = [x for row in self.start_blocks for x in row]
            if sorted(flat_start) == list(range(self.num_row * self.num_row)):
                self.blocks = copy.deepcopy(self.start_blocks)
                for row in range(self.num_row):
                    for column in range(self.num_row):
                        if self.blocks[row][column] == 0:
                            self.zero_row = row
                            self.zero_column = column
                self.start_blocks = []
            else:
                QMessageBox.warning(self, "Invalid Input", "start_blocks contains invalid numbers.")
        for i in range(self.num_suffle):
            random_num = random.randint(0, 3)
            self.move(Direction(random_num))
        self.start_blocks = copy.deepcopy(self.blocks)
        self.updatePanel()

    def resetStartBlock(self):
        self.start_blocks = copy.deepcopy(self.blocks)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Up or key == Qt.Key_W:
            self.move(Direction.DOWN)
            self.resetStartBlock()
        if key == Qt.Key_Down or key == Qt.Key_S:
            self.move(Direction.UP)
            self.resetStartBlock()
        if key == Qt.Key_Left or key == Qt.Key_A:
            self.move(Direction.RIGHT)
            self.resetStartBlock()
        if key == Qt.Key_Right or key == Qt.Key_D:
            self.move(Direction.LEFT)
            self.resetStartBlock()
        if key == Qt.Key_B:
            self.simulateOneStep()
        if key == Qt.Key_U:
            self.blocks = copy.deepcopy(self.start_blocks)
            for i in range(self.num_row):
                for j in range(self.num_row):
                    if self.blocks[i][j] == 0:
                        self.zero_row = i
                        self.zero_column = j
        self.updatePanel()
        if self.checkResult():
            if QMessageBox.Ok == QMessageBox.information(self, 'Challenge Results', 'Congratulations on completing the challenge!'):
                self.reset()

    def simulatePath(self, path):
        self.blocks = copy.deepcopy(self.start_blocks)
        for i in range(self.num_row):
            for j in range(self.num_row):
                if self.blocks[i][j] == 0:
                    self.zero_row = i
                    self.zero_column = j
        self.updatePanel()
        QApplication.processEvents()
        for move in path:
            if move == 'D':
                self.move(Direction.DOWN)
            elif move == 'U':
                self.move(Direction.UP)
            elif move == 'R':
                self.move(Direction.RIGHT)
            elif move == 'L':
                self.move(Direction.LEFT)
            self.updatePanel()
            QApplication.processEvents()
            sleep(0.5)

    def simulateOneStep(self):
        if self.way:
            move = self.way.pop(0)
            if move == 'D':
                self.move(Direction.DOWN)
            elif move == 'U':
                self.move(Direction.UP)
            elif move == 'R':
                self.move(Direction.RIGHT)
            elif move == 'L':
                self.move(Direction.LEFT)
            self.updatePanel()

    def move(self, direction):
        if direction == Direction.UP:
            if self.zero_row != self.num_row - 1:
                self.blocks[self.zero_row][self.zero_column] = self.blocks[self.zero_row + 1][self.zero_column]
                self.blocks[self.zero_row + 1][self.zero_column] = 0
                self.zero_row += 1
        if direction == Direction.DOWN:
            if self.zero_row != 0:
                self.blocks[self.zero_row][self.zero_column] = self.blocks[self.zero_row - 1][self.zero_column]
                self.blocks[self.zero_row - 1][self.zero_column] = 0
                self.zero_row -= 1
        if direction == Direction.LEFT:
            if self.zero_column != self.num_row - 1:
                self.blocks[self.zero_row][self.zero_column] = self.blocks[self.zero_row][self.zero_column + 1]
                self.blocks[self.zero_row][self.zero_column + 1] = 0
                self.zero_column += 1
        if direction == Direction.RIGHT:
            if self.zero_column != 0:
                self.blocks[self.zero_row][self.zero_column] = self.blocks[self.zero_row][self.zero_column - 1]
                self.blocks[self.zero_row][self.zero_column - 1] = 0
                self.zero_column -= 1

    def updatePanel(self):
        for i in reversed(range(self.gltMain.count())):
            widget = self.gltMain.itemAt(i).widget()
            if widget is not None:
                widget.setParent(None)
        for row in range(self.num_row):
            for column in range(self.num_row):
                self.gltMain.addWidget(Block(self.blocks[row][column], int(720 / self.num_row) - 15), row, column)
        self.widget.setLayout(self.gltMain)

    def checkResult(self):
        if self.blocks[self.num_row - 1][self.num_row - 1] != 0:
            return False
        for row in range(self.num_row):
            for column in range(self.num_row):
                if row == self.num_row - 1 and column == self.num_row - 1:
                    continue
                if self.blocks[row][column] != row * self.num_row + column + 1:
                    return False
        return True

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = NumberNPuzzle()
    sys.exit(app.exec_())