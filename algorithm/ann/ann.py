import heapq
import time
import numpy as np
import tensorflow as tf

class NPuzzleSolver:
    def __init__(self, size=3, goal_state=None):
        self.size = size
        self.n = size * size
        if goal_state is None:
            self.goal_state = [i % self.n for i in range(1, self.n + 1)]
        else:
            self.goal_state = goal_state

        # Tải 1 mô hình Keras
        self.model = tf.keras.models.load_model(
            'C:/Users/DUYEN/OneDrive/Documents/GitHub/NhapmonAI-Project/model/ann.h5',
            compile=False
        )

        self.heuristic_count = 0  # Đếm số lần gọi heuristic

    def heuristic(self, state):
        """Dự đoán chi phí bằng mô hình Keras"""
        self.heuristic_count += 1
        flat_state = [num for row in state for num in row]
        x_encoded = np.eye(16)[flat_state].ravel()  # shape (256,)

        output = self.model.predict(x_encoded.reshape(1, -1), verbose=0)  # shape (1, 1)
        estimated_cost = max(output.item(), 0)

        print(f"Board {self.heuristic_count} → Estimated cost: {estimated_cost:.2f}")
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

    def print_state(self, state):
        for row in state:
            print(' '.join(str(num).rjust(2) if num != 0 else '  ' for num in row))
        print()

    def is_goal(self, state):
        flat = [num for row in state for num in row]
        return flat == self.goal_state

    def solve(self, start_state):
        """Chạy A* để giải bài toán"""
        start_time = time.time()
        pq = []
        visited = set()
        heapq.heappush(pq, (self.heuristic(start_state), 0, start_state, []))  # (f, g, state, path)

        while pq:
            f, g, current, path = heapq.heappop(pq)
            state_key = self.state_to_tuple(current)

            if self.is_goal(current):
                total_time = time.time() - start_time
                print("✅ Đã tìm thấy lời giải sau", g, "bước:")
                for idx, s in enumerate(path + [current]):
                    print(f"Step {idx}:")
                    self.print_state(s)
                print(f"🕒 Thời gian chạy: {total_time:.4f} giây")
                return

            if state_key in visited:
                continue
            visited.add(state_key)

            for neighbor in self.get_neighbors(current):
                if self.state_to_tuple(neighbor) not in visited:
                    h = self.heuristic(neighbor)
                    heapq.heappush(pq, (g + 1 + h, g + 1, neighbor, path + [current]))

        print("❌ Không tìm được lời giải.")

# Trạng thái ban đầu cho 15-puzzle
start_state = [
    [5, 1, 2, 3],
    [0, 6, 7, 4],
    [9,10,11,8],
    [13,14,15,12] 
]

# Chạy solver
solver = NPuzzleSolver(size=4)
solver.solve(start_state)
