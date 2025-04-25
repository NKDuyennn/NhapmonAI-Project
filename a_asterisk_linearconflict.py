import time
import heapq
import collections

class Block_Puzzle:
    def __init__(self, blocks):
        """
        Hàm khởi tạo. Nhận một từ điển các cặp số nguyên-tuple mô tả vị trí các ô.
        Hoặc nhận một danh sách 2D các số nguyên sau đó được chuyển đổi.
        """
        if type(blocks) is list:
            self.blocks = self.__list_to_dict__(blocks)
        else:
            self.blocks = blocks
        self.width = self.__get_width__()

    def __eq__(self, other):
        """Hàm so sánh bằng. Kiểm tra xem hai từ điển ô có giống nhau không."""
        if other is None:
            return False
        return self.blocks == other.blocks

    def __lt__(self, other):
        """Hàm so sánh nhỏ hơn. So sánh điểm f. Nếu điểm f bằng nhau, so sánh điểm g."""
        if other is None:
            return False
        if fScore[self] == fScore[other]:
            return gScore[self] < gScore[other]
        return fScore[self] < fScore[other]

    def __hash__(self):
        """Hàm băm. Băm một frozenset của các ô."""
        return hash(frozenset(self.blocks.items()))

    def __list_to_dict__(self, blockList):
        """Chuyển đổi danh sách 2D các số nguyên thành từ điển."""
        dictionary = {}
        for row in range(len(blockList)):
            for col in range(len(blockList[0])):
                dictionary[blockList[row][col]] = (col, row)
        return dictionary

    def __get_width__(self):
        """Trả về chiều rộng của câu đố này."""
        maxWidth = 0
        for value in self.blocks.values():
            maxWidth = max(value[0], maxWidth)
        return maxWidth + 1

    def create_solved_puzzle(self):
        """Trả về một câu đố đã giải dựa trên kích thước của câu đố này."""
        cellArr = []
        for row in range(self.width):
            boardRow = []
            for col in range(self.width):
                boardRow.append((row * self.width) + col + 1)
            cellArr.append(boardRow)
        cellArr[self.width - 1][self.width - 1] = 0
        puzzle = Block_Puzzle(cellArr)
        return puzzle

    def get_move(self, other):
        """Diễn giải một câu đố khác thành một nước đi 'U', 'D', 'L' hoặc 'R' từ câu đố này."""
        thisEmpty = self.blocks[0]
        thatEmpty = other.blocks[0]
        move = (thisEmpty[0] - thatEmpty[0], thisEmpty[1] - thatEmpty[1])
        moveList = {(0,1): "D", (0,-1): "U", (1,0): "R", (-1,0): "L"}
        return moveList[move]

    def heuristic_estimate_manhattan(self, other):
        """
        Finds the heuristic estimation of the cost to reach another state from this one.
        This heuristic is based on "manhattan distance."
        Returns the sum of each tile's orthogonal movement to reach its twin on the other board.
        """
        estimate = 0

        for index in range(len(self.blocks)):
            estimate += abs(other.blocks[index][0] - self.blocks[index][0]) + abs(other.blocks[index][1] - self.blocks[index][1])

        return estimate
    
    def heuristic_estimate_linear_conflict(self, other):
        """
        Tính toán ước lượng heuristic sử dụng khoảng cách Manhattan cộng với xung đột tuyến tính.
        Thêm 2 vào ước lượng cho mỗi cặp ô trong cùng hàng hoặc cột có xung đột.
        https://algorithmsinsight.wordpress.com/graph-theory-2/a-star-in-general/implementing-a-star-to-solve-n-puzzle/
        """
        estimate = 0

        for index in range(len(self.blocks)):
            x_cur = self.blocks[index][0]
            y_cur = self.blocks[index][1]
            x_goal = other.blocks[index][0]
            y_goal = other.blocks[index][1]
            if x_cur == x_goal and y_cur == y_goal:
                continue

            estimate += abs(x_goal - x_cur) + abs(y_goal - y_cur)

            if x_cur == x_goal:
                index_begin = x_cur * self.width
                index_end = index_begin + y_goal - 1
                for i in range(index_begin, index_end):
                    if i in self.blocks and self.blocks[i][0] == x_cur and self.blocks[i][1] > y_cur:
                        estimate += 2

            elif y_cur == y_goal:
                index_begin = y_cur
                index_end = (x_goal - 1) * self.width + y_cur
                for i in range(index_begin, index_end, self.width):
                    if i in self.blocks and self.blocks[i][1] == y_cur and self.blocks[i][0] > x_cur:
                        estimate += 2

        return estimate

    def get_neighbors(self, previous):
        """Lấy tất cả các trạng thái lân cận của trạng thái hiện tại, trừ trạng thái trước đó."""
        neighbors = []
        # Các nước đi có thể: lên, xuống, trái, phải
        moves = ((-1,0), (1,0), (0,-1), (0,1))
        zeroLoc = self.blocks[0]
        
        for move in moves:
            newZeroLoc = (zeroLoc[0] + move[0], zeroLoc[1] + move[1])
            # Kiểm tra xem vị trí mới có hợp lệ không
            # và không phải là vị trí của ô trước đó
            if newZeroLoc[0] < 0 or newZeroLoc[1] < 0 or newZeroLoc[0] > self.width-1 or newZeroLoc[1] > self.width-1:
                continue

            if previous and previous.blocks[0] == newZeroLoc:
                continue


            newBlocks = dict(self.blocks)
            # Hoán đổi vị trí của ô 0 với ô ở vị trí mới
            newBlocks[0] = newZeroLoc
            # Hoán đổi vị trí của ô ở vị trí mới với ô 0

            for face, location in newBlocks.items():
                if face != 0 and location == newZeroLoc:
                    newBlocks[face] = zeroLoc
            
            # Tạo một trạng thái mới với các ô đã hoán đổi
            # và thêm vào danh sách các trạng thái lân cận
            neighbor = Block_Puzzle(newBlocks)
            neighbors.append(neighbor)

        return neighbors

class AASTERISK:
    def __init__(self, board, width) -> None:
        self.board = board
        self.width = width

    def run_Astar(self, start, goal):
        """
        A star search algorithm. Takes a start state and an end state.
        While there are available moves, loops through them and exits if the end is found.
        Returns the list of states that are the "quickest" way to the end.
        """
        # The set of states already evaluated
        closedSet = set()

        # For each node, which node it can most efficiently be reached from.
        # If a node can be reached from many start, cameFrom will eventually contain the
        # most efficient previous step.
        cameFrom = {}

        # For each node, the total cost of getting from the start node to the goal
        # by passing by that node. That value is partly known, partly heuristic.
        # This variable is global for the Block_Puzzle object's __lt__ method.
        global fScore
        fScore = collections.defaultdict(lambda: float("inf"))

        # For each node, the cost of getting from the start node to that node.
        # This variable is global for the Block_Puzzle object's __lt__ method.
        global gScore
        gScore = collections.defaultdict(lambda: float("inf"))

        # The cost of going from start to start is zero.
        gScore[start] = 0

        # The heap of currently discovered state that are not evaluated yet.
        # Initially, only the start state is known.
        openHeap = [start]
        heapq.heapify(openHeap)

        # For the first node, that value is completely heuristic.
        fScore[start] = start.heuristic_estimate_manhattan(goal)
        # print(goal.blocks)

        # While there are yet nodes to inspect,
        while openHeap:
            # Get the lowest f-score state not yet evaluated.
            current = heapq.heappop(openHeap)

            # print(len(openHeap))

            # Skip this state if it's a duplicate of one that's already been evaluated.
            if current in closedSet:
                continue

            # If we've reached the goal:
            if current == goal:
                # return the list of states it took to get there.
                end = time.time()
                path = []
                path.append(current)
                cnt = 0
                step = current
                while(cameFrom.get(step)):
                    path.append(cameFrom[step])
                    step = cameFrom[step]
                    cnt = cnt + 1
                ##path.reverse()
                return cnt, end, path

            # make sure we don't visit this state again.
            closedSet.add(current)

            # For each possible neighbor of our current state,
            for neighbor in current.get_neighbors(cameFrom.get(current)):
                # Skip it if it's already been evaluated
                if neighbor in closedSet:
                    continue

                tentative_gScore = gScore[current] + 1
                # If this path costs less than previous paths here...
                if tentative_gScore < gScore[neighbor]:
                    # Update the values for this state.
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = gScore[neighbor] + neighbor.heuristic_estimate_manhattan(goal)
                
                # Finally, add it to our open heap
                heapq.heappush(openHeap, neighbor)

    #### SOLVER FUNCTION ####

    def findMinimumSteps(self):
        # from the cell list provided, create a Block Puzzle.
        start = Block_Puzzle(self.board)

        # find the solution puzzle for the given puzzle.
        goal = start.create_solved_puzzle()

        # run A* search algorithm.
        begin = time.time()
        num_steps, end, path = self.run_Astar(start, goal)

        # convert into single-letter moves.
        moves = []
        for index in range(len(path)-1):
            moves.append(path[index].get_move(path[index+1]))
        return end - begin, num_steps, moves
    
class AASTERISKLinearConflict(AASTERISK):
    def __init__(self, board, width) -> None:
        # self.board = board
        # self.width = width
        super().__init__(board, width)

    def run_Astar(self, start, goal):
        closedSet = set()
        cameFrom = {}

        global fScore
        fScore = collections.defaultdict(lambda: float("inf"))

        global gScore
        gScore = collections.defaultdict(lambda: float("inf"))

        gScore[start] = 0

        openHeap = [start]
        heapq.heapify(openHeap)

        fScore[start] = start.heuristic_estimate_linear_conflict(goal)

        while openHeap:
            current = heapq.heappop(openHeap)

            if current in closedSet:
                continue

            if current == goal:
                end = time.time()
                path = []
                path.append(current)
                cnt = 0
                step = current
                while(cameFrom.get(step)):
                    path.append(cameFrom[step])
                    step = cameFrom[step]
                    cnt = cnt + 1
                #path.reverse()
                return cnt, end, path

            closedSet.add(current)

            for neighbor in current.get_neighbors(cameFrom.get(current)):
                if neighbor in closedSet:
                    continue

                tentative_gScore = gScore[current] + 1
                if tentative_gScore < gScore[neighbor]:
                    cameFrom[neighbor] = current
                    gScore[neighbor] = tentative_gScore
                    fScore[neighbor] = gScore[neighbor] + neighbor.heuristic_estimate_linear_conflict(goal)
                
                heapq.heappush(openHeap, neighbor)


# Các trường hợp kiểm tra
test_boards = [
    [[6,5,2,3],
     [0,7,11,4],
     [9,1,10,8],
     [15,14,13,12]],  # Khó
    [[1,2,3,4],
     [0,7,11,5],
     [9,6,10,8],
     [15,14,13,12]],  # Trung bình
    # [[1,2,3,4],
    #  [0,5,6,7],
    #  [9,10,11,8]],    # Dễ
    [[8,5,1],
     [6,7,0],
     [3,2,4]],        # 3x3
    [[0,2,4,8],
     [3,1,6,12],
     [5,9,10,7],
     [13,14,11,15]],  # 4x4
    [[9,7,5,4],
     [6,1,0,8],
     [10,3,14,11],
     [13,15,2,12]]    # 4x4
]

# Chạy kiểm tra
for i, board in enumerate(test_boards):
    print(f"\nTrường hợp kiểm tra {i+1}:")
    a = AASTERISKLinearConflict(board, len(board[0]))
    duration, num_steps, moves = a.findMinimumSteps()
    print(f"Thời gian: {duration:.4f} giây")
    print(f"Số bước: {num_steps}")
    print(f"Nước đi: {moves}")