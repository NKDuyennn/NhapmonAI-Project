from collections import deque
import math
import time

class BFSAgent:
    def __init__(self, cells, width):
        self.cells = cells
        self.width = width

    def findMinimumSteps(self):
        start = time.time()
        BFSqueue = deque([Node(cells=self.cells, width=self.width)])
        visited = set()
        visited.add(str(BFSqueue[0].cells))

        while BFSqueue:
            node = BFSqueue.popleft()  # FIFO for BFS
            node_ord_step = node.ordinal_step
            if node.isSolved():
                end = time.time()
                duration = end - start
                temp = node
                path = []
                while temp.p_action is not None:
                    path.append(temp.p_action)
                    temp = temp.parent
                path.reverse()  # Reverse to get correct order
                print(f"BFS: num_steps={node_ord_step}, path_length={len(path)}, path={path}")
                if len(path) != node_ord_step:
                    print("BFS Error: Path length does not match num_steps")
                return duration, node_ord_step, path
            for next_state, action in node.getNextStates():
                child = Node(cells=next_state, width=self.width, parent=node,
                            p_action=action, ordinal_step=node_ord_step + 1)
                if str(child.cells) not in visited:
                    BFSqueue.append(child)
                    visited.add(str(child.cells))
        end = time.time()
        duration = end - start
        return duration, float('inf'), []

class Node:
    def __init__(self, cells, width, parent=None, p_action=None, ordinal_step=0):
        self.cells = cells
        self.parent = parent
        self.p_action = p_action
        self.ordinal_step = ordinal_step
        self.width = int(width)

    def isSolved(self):
        # Goal state: [1, 2, ..., n*n-1, 0]
        goal = list(range(1, self.width * self.width)) + [0]
        return self.cells == goal

    def swapCell(self, r, c, i, j):
        clone_cells = list(self.cells)
        idx1 = r * self.width + c
        idx2 = i * self.width + j
        clone_cells[idx1], clone_cells[idx2] = clone_cells[idx2], clone_cells[idx1]
        return clone_cells

    def getNextStates(self):
        empty_space = self.cells.index(0)
        empty_space_row = empty_space // self.width
        empty_space_col = empty_space % self.width
        next_states = []

        # Define moves: moving the empty tile (opposite of tile movement)
        actions = {
            "U": (empty_space_row - 1, empty_space_col),  # Move empty tile up
            "D": (empty_space_row + 1, empty_space_col),  # Move empty tile down
            "L": (empty_space_row, empty_space_col - 1),  # Move empty tile left
            "R": (empty_space_row, empty_space_col + 1)   # Move empty tile right
        }

        for action, (row, col) in actions.items():
            if 0 <= row < self.width and 0 <= col < self.width:
                next_state = self.swapCell(empty_space_row, empty_space_col, row, col)
                next_states.append((next_state, action))

        return next_states