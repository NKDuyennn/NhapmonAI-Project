from collections import deque
from cmath import inf
import math, time
from platform import node
import numpy as np
class IDSAgent:
    def __init__(self, cells, width) -> None:
        """
        Khởi tạo agent dùng thuật toán IDS (Iterative Deepening Search).
        - cells: danh sách các giá trị trên bảng puzzle.
        - width: chiều rộng của bảng puzzle (ví dụ: 3 cho 3x3).
        """
        self.cells = cells
        self.width = width
        self.maximum_steps = 100000
        self.minimum_steps = inf
        self.minimum_steps_node = None

        self.h(self.cells)
        # i, num_cells = 0, len(self.cells)
    
    def h(self, cells):
        """
        Tính toán heuristic ban đầu bằng khoảng cách Manhattan giữa vị trí hiện tại và vị trí đích của từng ô.
        Giá trị này không dùng trực tiếp trong tìm kiếm IDS nhưng có thể dùng để xác định giới hạn độ sâu ban đầu.
        """
        cnt = 0
        w = self.width
        for i, v in enumerate(cells, 1):
            if i == w * w:
                continue
            i1 = i % w
            i2 = (i-i1)/w + 1
            if i1 == 0:
                i1 = w
                i2 = i/w 

            v1 = v % w
            v2 = (v-v1)/w + 1
            if v1 == 0:
                v1 = w
                v2 = v/w
            cnt += abs(i1 - v1) + abs(i2 - v2)
        self.numberBeginSteps = cnt

    def findMinimumSteps(self):
        """
        Áp dụng thuật toán IDS để tìm lời giải ngắn nhất cho bài toán puzzle.
        Trả về thời gian chạy và số bước để giải.
        """
        start = time.time()
        for i in range(int(self.numberBeginSteps), int(self.maximum_steps)):
            IDSstack = list([Node(cells = self.cells, width = self.width)])
            visited = set()
            while IDSstack:
                node = IDSstack.pop()
                visited.add(str(node.cells) + str(node.ordinal_step))
                node_ord_step = node.ordinal_step

                if node.isSolved():
                    if node_ord_step < self.minimum_steps:
                        self.minimum_steps = node_ord_step
                        self.minimum_steps_node = node
                    break

                if node_ord_step == i:
                    continue
                neighbors = reversed(node.getNextStates())
                for next_state, action in neighbors:
                    child = Node(cells = next_state, width = self.width, parent = node, 
                        ordinal_step = node_ord_step + 1)
                    if str(child.cells) + str(child.ordinal_step) not in visited:
                        IDSstack.append(child)
                        visited.add(str(child.cells)+ str(child.ordinal_step) )                       
            if self.minimum_steps != inf:
                # print("Number of steps: " + str(self.minimum_steps))
                break
        end = time.time()
        duration = end - start   
        return duration, self.minimum_steps #, self.minimum_steps_node.getPath()
    


class Node:
    def __init__(self, cells, width:int, parent = None, p_action = None, ordinal_step = 0, cost = 0) -> None:
        """
        Đại diện cho một trạng thái của bảng puzzle.
        - cells: cấu trúc bảng hiện tại.
        - width: chiều rộng của bảng.
        - parent: node cha (dùng để truy ngược đường đi).
        - p_action: hành động dẫn đến node này.
        - ordinal_step: số bước từ trạng thái gốc đến đây.
        """
        self.cells = cells
        self.parent = parent
        self.p_action = p_action
        self.ordinal_step = ordinal_step
        self.width = int(width)

    def isSolved(self):
        """
        Kiểm tra trạng thái hiện tại có phải là trạng thái đích không.
        Trạng thái đích là dãy tăng dần từ 1 đến n-1, ô trống là 0 ở cuối.
        """
        i, num_cells = 0, len(self.cells)
        for i, v in enumerate(self.cells, 1):
            if (i != v):
                break
        if i == num_cells:
            return True
        return False

    def swapCell(self, r:int, c:int, i:int, j:int):
        """
        Đổi chỗ ô tại vị trí (r, c) với ô (i, j).
        Trả về bảng mới sau khi đổi chỗ.
        """
        clone_cells = list(self.cells)
        #print(self.cells)
        
        #print(r, c, i, j)
        clone_cells[r * self.width + c], clone_cells[i * self.width + j] \
                    = clone_cells[i * self.width + j], clone_cells[r * self.width + c]
        return clone_cells

    def getNextStates(self):
        """
        Trả về danh sách trạng thái có thể đạt được từ trạng thái hiện tại,
        bằng cách di chuyển ô trống (0) theo 4 hướng nếu hợp lệ.
        """
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
    
    def getPath(self):
        """
        Truy ngược lại đường đi từ trạng thái gốc đến trạng thái hiện tại.
        Trả về danh sách các trạng thái theo thứ tự thời gian.
        """
        path = []
        node = self
        while node.parent:
            path.append(node.cells)
            node = node.parent
        path.append(node.cells)
        return path[::-1]

# ----------- Chạy thử thuật toán IDS với 1 ví dụ 3x3 -------------
# cells = [6, 3, 8, 0, 1, 5, 7, 2, 4]
# IDS = IDSAgent(cells, math.isqrt(len(cells)))
# t, steps = IDS.findMinimumSteps()
# print("Solution Path:")
# for state in IDS.minimum_steps_node.getPath():
#     for i in range(0, len(state), IDS.width):
#         print(state[i:i+IDS.width])
#     print("---")
# print(f"Time: {t:.4f}s, Steps: {steps}")