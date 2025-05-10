from collections import deque
import math, time

class BFSAgent:
    def __init__(self, cells, width) -> None:
        self.cells = cells
        self.width = width
        self.minimum_steps = float('inf')
        self.minimum_steps_node = None

    def findMinimumSteps(self, timeout=None):
        start = time.time()
        BFSqueue = deque([Node(cells=self.cells, width=self.width)])
        visited = set()
        visited.add(str(BFSqueue[0].cells))

        while BFSqueue:
            # Nếu có timeout và đã quá thời gian => dừng
            if timeout is not None and (time.time() - start > timeout):
                return {"timeout": True}

            node = BFSqueue.pop()
            node_ord_step = node.ordinal_step
            if node.isSolved():
                if node_ord_step < self.minimum_steps:
                    self.minimum_steps = node_ord_step
                    self.minimum_steps_node = node
                    break

            for next_state, action in node.getNextStates():
                child = Node(cells=next_state, width=self.width, parent=node,
                             p_action=action, ordinal_step=node_ord_step + 1)
                if str(child.cells) not in visited:
                    BFSqueue.appendleft(child)
                    visited.add(str(child.cells))

        end = time.time()
        duration = end - start

        if self.minimum_steps_node:
            move_sequence = self.minimum_steps_node.getPath()
            final_state = self.minimum_steps_node.cells
            return {
                "move_sequence": move_sequence,
                "total_steps": len(move_sequence),
                "solving_time": duration,
                "final_state": final_state
            }
        else:
            return None

class Node:
    def __init__(self, cells, width:int, parent=None, p_action=None, ordinal_step=0) -> None:
        self.cells = cells
        self.parent = parent
        self.p_action = p_action
        self.ordinal_step = ordinal_step
        self.width = int(width)

    def isSolved(self):
        correct = list(range(1, len(self.cells))) + [0]
        return self.cells == correct

    def swapCell(self, r:int, c:int, i:int, j:int):
        clone_cells = list(self.cells)
        clone_cells[r * self.width + c], clone_cells[i * self.width + j] = \
            clone_cells[i * self.width + j], clone_cells[r * self.width + c]
        return clone_cells

    def getNextStates(self):
        empty_space = self.cells.index(0)
        row, col = divmod(empty_space, self.width)
        next_states = []

        actions = {
            "R": (row, col + 1),
            "L": (row, col - 1),
            "U": (row - 1, col),
            "D": (row + 1, col)
        }

        for action, (r, c) in actions.items():
            if 0 <= r < self.width and 0 <= c < self.width:
                next_cells = self.swapCell(row, col, r, c)
                next_states.append((next_cells, action))
        return next_states

    def getPath(self):
        path = []
        node = self
        while node.parent is not None:
            path.append(node.p_action)
            node = node.parent
        return path[::-1]

# =======================
# RUN TEST CONFIGURATIONS
# =======================

cell_configs = [
    [0, 5, 7, 2, 1, 4, 3, 8, 6],
    [5, 8, 2, 1, 7, 4, 6, 3, 0],
    [6, 4, 0, 7 , 1, 5, 2, 3, 8],
    [1, 4, 3, 6, 5, 2, 0, 7, 9, 10, 12, 8, 13, 14, 11, 15],
    [6, 5, 2, 3, 0, 7, 11, 4, 9, 1, 10, 8, 15, 14, 13, 12],
    [0, 7, 2, 4, 6, 12, 1, 15, 13, 5, 9, 8, 10, 11, 3, 14],
    [2, 0, 3, 4, 5, 1, 6, 8, 9, 10, 11, 7, 12, 14, 15, 21, 17, 13, 24, 19, 22, 16, 18, 23, 20],
    [6, 1, 2, 14, 4, 16, 8, 3, 9, 5, 17, 7, 11, 24, 18, 13, 12, 10, 15, 19, 21, 22, 23, 20, 0],
    [6, 1, 4, 9, 3, 11, 2, 7, 10, 0, 16, 12, 20, 13, 5, 17, 18, 8, 19, 15, 21, 22, 23, 24, 14],
]

# Optional: set timeout in seconds
optional_timeout = 100  # or None if you want to disable timeout

for cells in cell_configs:
    print("Testing configuration:", cells)
    BFS = BFSAgent(cells, math.isqrt(len(cells)))
    result = BFS.findMinimumSteps(timeout=optional_timeout)

    if result is None:
        print("No solution found.")
    elif "timeout" in result:
        print("Timeout: Search exceeded time limit.")
    else:
        print("Move Sequence:", result["move_sequence"])
        print("Total Steps:", result["total_steps"])
        print("Solving Time (seconds):", result["solving_time"])
        print("Final State:", result["final_state"])
    print("-" * 50)
