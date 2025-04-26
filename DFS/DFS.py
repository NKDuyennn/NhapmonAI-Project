from collections import deque
from cmath import inf
import math, time

class DFSAgent:
    def __init__(self, cells, width) -> None:
        # Khởi tạo agent với trạng thái ban đầu và kích thước bàn cờ
        self.cells = cells
        self.width = width
        self.minimum_steps = inf
        self.minimum_steps_node = None

    def findMinimumSteps(self):
        # Tìm đường đi ngắn nhất bằng DFS
        start = time.time()
        DFSstack = list([Node(cells = self.cells, width = self.width)])
        # if using list, error: unhashable type: 'list'
        visited = set()

        while DFSstack:
            node = DFSstack.pop()
            visited.add(tuple(node.cells))

            node_ord_step = node.ordinal_step
            if node.isSolved():
                if node_ord_step < self.minimum_steps:
                    self.minimum_steps = node_ord_step
                    self.minimum_steps_node = node
                    break
            neighbors = reversed(node.getNextStates())
            for next_state, action in neighbors:
                child = Node(cells = next_state, width = self.width, parent = node, 
                    ordinal_step = node_ord_step + 1)
                if tuple(child.cells) not in visited:
                    DFSstack.append(child)
                    visited.add(tuple(child.cells))


        end = time.time()
        duration = end - start
        # print(self.minimum_steps)
        return duration, self.minimum_steps #, self.minimum_steps_node.getPath()

class Node:
    def __init__(self, cells, width:int, parent = None, p_action = None, ordinal_step = 0) -> None:
        # Lưu trạng thái của một nút trong cây tìm kiếm
        self.cells = cells
        self.parent = parent
        self.p_action = p_action
        self.ordinal_step = ordinal_step
        self.width = int(width)

    def isSolved(self):
        # Kiểm tra trạng thái hiện tại đã phải trạng thái đích chưa
        i, num_cells = 0, len(self.cells)
        for i, v in enumerate(self.cells, 1):
            if (i != v):
                break
        if i == num_cells:
            return True
        return False

    def swapCell(self, r:int, c:int, i:int, j:int):
        # Đổi chỗ hai ô trong ma trận (trạng thái mới sau một hành động)
        clone_cells = list(self.cells)
        #print(self.cells)
        
        #print(r, c, i, j)
        clone_cells[r * self.width + c], clone_cells[i * self.width + j] \
                    = clone_cells[i * self.width + j], clone_cells[r * self.width + c]
        return clone_cells
    
    def getPath(self):
        # Truy vết lại đường đi từ nút hiện tại về gốc
        path = []
        node = self
        while node.parent:
            path.append(node.cells)
            node = node.parent
        path.append(node.cells)
        return path[::-1]

    def getNextStates(self):
        # Sinh ra các trạng thái kế tiếp từ trạng thái hiện tại
        empty_space = self.cells.index(0)
        empty_space_row = empty_space // self.width
        empty_space_col = empty_space % self.width
        next_states = list()

        actions = {"R":(empty_space_row, empty_space_col + 1),
                             "L":(empty_space_row, empty_space_col - 1),
                             "U":(empty_space_row - 1, empty_space_col),
                             "D":(empty_space_row + 1, empty_space_col)}

        for action, (row, col) in actions.items():
            if row >= 0 and row < self.width and col >= 0 and col < self.width:
                #print(self.width, empty_space_row, empty_space_col, row, col)
                move = self.swapCell(empty_space_row, empty_space_col, row, col), action
                next_states.append(move)
        
        #print(next_states)
        return next_states

# ----------- Chạy thử thuật toán IDS với 1 ví dụ 3x3 -------------
# print("\nTest case : Gốc ban đầu")
# cells = [6,3,8,0,1,5,7,2,4]
# DFS = DFSAgent(cells, math.isqrt(len(cells)))
# t, steps = DFS.findMinimumSteps()
# print("Solution Path:")
# for state in DFS.minimum_steps_node.getPath():
#     for i in range(0, len(state), DFS.width):
#         print(state[i:i+DFS.width])
#     print("---")
# print(f"Time: {t:.4f}s, Steps: {steps}")