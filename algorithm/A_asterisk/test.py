def heuristic_estimate_linear_conflict(self, other):
    """
    Linear Conflict Heuristic - Phiên bản đã sửa lỗi
    Tính Manhattan Distance + Linear Conflict penalty
    """
    estimate = 0
    
    # Bước 1: Tính Manhattan Distance cho tất cả tiles (trừ tile 0)
    for tile_value in self.blocks:
        if tile_value == 0:  # Bỏ qua ô trống
            continue
        x_cur, y_cur = self.blocks[tile_value]
        x_goal, y_goal = other.blocks[tile_value]
        estimate += abs(x_goal - x_cur) + abs(y_goal - y_cur)
    
    # Bước 2: Thêm penalty cho Linear Conflicts
    
    # Kiểm tra conflicts trong từng hàng
    for row in range(self.width):
        # Tìm tất cả tiles ở hàng này trong cả trạng thái hiện tại và đích
        tiles_in_row = []
        for tile_value in self.blocks:
            if tile_value == 0:
                continue
            x_cur, y_cur = self.blocks[tile_value]
            x_goal, y_goal = other.blocks[tile_value]
            
            # Nếu tile hiện tại ở hàng này VÀ đích cũng ở hàng này
            if y_cur == row and y_goal == row:
                tiles_in_row.append((tile_value, x_cur, x_goal))
        
        # Đếm conflicts trong hàng này
        estimate += self._count_conflicts_in_line(tiles_in_row)
    
    # Kiểm tra conflicts trong từng cột
    for col in range(self.width):
        # Tìm tất cả tiles ở cột này trong cả trạng thái hiện tại và đích
        tiles_in_col = []
        for tile_value in self.blocks:
            if tile_value == 0:
                continue
            x_cur, y_cur = self.blocks[tile_value]
            x_goal, y_goal = other.blocks[tile_value]
            
            # Nếu tile hiện tại ở cột này VÀ đích cũng ở cột này
            if x_cur == col and x_goal == col:
                tiles_in_col.append((tile_value, y_cur, y_goal))
        
        # Đếm conflicts trong cột này
        estimate += self._count_conflicts_in_line(tiles_in_col)
    
    return estimate

def _count_conflicts_in_line(self, tiles):
    """
    Đếm số linear conflicts trong một hàng hoặc cột
    tiles: list of (tile_value, current_position, goal_position)
    Trả về: số conflicts * 2 (vì mỗi conflict cần ít nhất 2 moves để giải quyết)
    """
    conflicts = 0
    n = len(tiles)
    
    # So sánh từng cặp tiles
    for i in range(n):
        for j in range(i + 1, n):
            tile1_val, tile1_cur, tile1_goal = tiles[i]
            tile2_val, tile2_cur, tile2_goal = tiles[j]
            
            # Linear conflict xảy ra khi thứ tự hiện tại khác với thứ tự đích
            if ((tile1_cur < tile2_cur and tile1_goal > tile2_goal) or
                (tile1_cur > tile2_cur and tile1_goal < tile2_goal)):
                conflicts += 1
    
    return conflicts * 2  # Mỗi conflict cần ít nhất 2 moves để giải quyết


# Ví dụ về cách hoạt động:
def example_linear_conflict():
    """
    Ví dụ minh họa Linear Conflict
    """
    # Trạng thái hiện tại: hàng đầu là [2, 1, 3]
    # Trạng thái đích: hàng đầu là [1, 2, 3]
    
    current_state = {
        1: (1, 0),  # Tile 1 ở vị trí (1,0)
        2: (0, 0),  # Tile 2 ở vị trí (0,0) 
        3: (2, 0),  # Tile 3 ở vị trí (2,0)
        0: (0, 1)   # Ô trống
    }
    
    goal_state = {
        1: (0, 0),  # Tile 1 cần ở vị trí (0,0)
        2: (1, 0),  # Tile 2 cần ở vị trí (1,0)
        3: (2, 0),  # Tile 3 cần ở vị trí (2,0)
        0: (2, 1)   # Ô trống
    }
    
    # Manhattan Distance:
    # Tile 1: |1-0| + |0-0| = 1
    # Tile 2: |0-1| + |0-0| = 1  
    # Tile 3: |2-2| + |0-0| = 0
    # Tổng Manhattan = 2
    
    # Linear Conflict:
    # Trong hàng 0: tiles 1,2,3 đều ở hàng 0 trong cả hai trạng thái
    # tiles_in_row = [(1, 1, 0), (2, 0, 1), (3, 2, 2)]
    # 
    # So sánh từng cặp:
    # - Tile 1 vs Tile 2: current (1 > 0) nhưng goal (0 < 1) → CONFLICT!
    # - Tile 1 vs Tile 3: current (1 < 2) và goal (0 < 2) → OK
    # - Tile 2 vs Tile 3: current (0 < 2) và goal (1 < 2) → OK
    #
    # Có 1 conflict → +2 penalty
    # Tổng heuristic = 2 + 2 = 4
    
    print("Ví dụ Linear Conflict:")
    print("Manhattan Distance: 2")
    print("Linear Conflicts: 1 conflict × 2 = 2")  
    print("Tổng heuristic: 4")
    
    return current_state, goal_state