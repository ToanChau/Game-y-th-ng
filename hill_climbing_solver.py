import random
from typing import List, Tuple, Optional
from sokoban_common import SokobanState, MOVES

class HillClimbingSolver:
    def __init__(self, max_iterations: int = 1000, max_sideways: int = 100):
        self.max_iterations = max_iterations
        self.max_sideways = max_sideways

    def solve(self, initial_state: SokobanState) -> Optional[List[Tuple[int, int]]]:
        current_state = initial_state
        current_score = self.evaluate_state(current_state)
        path = []

        for _ in range(self.max_iterations):
            if current_state.is_goal():
                return path

            neighbors = self.get_neighbors(current_state)
            if not neighbors:
                return None

            # Lọc ra các hàng xóm có điểm số lớn hơn điểm số hiện tại
            better_neighbors = [(neighbor, move) for neighbor, move in neighbors if self.evaluate_state(neighbor) > current_score]

            if better_neighbors:
                # Nếu có hàng xóm có điểm lớn hơn, chọn ngẫu nhiên một trong số đó
                best_neighbor = random.choice(better_neighbors)
            else:
                # Nếu không có hàng xóm nào lớn hơn, lọc các hàng xóm có điểm số bằng
                best_neighbors = [(neighbor, move) for neighbor, move in neighbors if self.evaluate_state(neighbor) == current_score]
                if best_neighbors:
                    # Chọn ngẫu nhiên một trong các hàng xóm có điểm số bằng
                    best_neighbor = random.choice(best_neighbors)
                else:
                    # Nếu không có hàng xóm nào có điểm bằng hoặc lớn hơn, thiết lập lại ngẫu nhiên
                    current_state = self.get_random_state(initial_state)
                    current_score = self.evaluate_state(current_state)
                    path = []
                    continue

            # Cập nhật trạng thái hiện tại
            current_state = best_neighbor[0]
            current_score = self.evaluate_state(current_state)
            path.append(best_neighbor[1])

        return None



    def evaluate_state(self, state: SokobanState) -> float:
        if state.is_goal():
            return float('inf')

        score = 0
        for box in state.boxes:
            min_distance = min(abs(box[0] - target[0]) + abs(box[1] - target[1]) for target in state.targets)
            score -= min_distance

        return score

    def get_neighbors(self, state: SokobanState) -> List[Tuple[SokobanState, Tuple[int, int]]]:
        neighbors = []
        for move in MOVES:
            new_state = state.apply_move(move)
            if new_state != state:
                neighbors.append((new_state, move))
        return neighbors

    def get_random_state(self, initial_state: SokobanState) -> SokobanState:
        current_state = initial_state
        for _ in range(random.randint(1, 20)):
            neighbors = self.get_neighbors(current_state)
            if neighbors:
                current_state = random.choice(neighbors)[0]
            else:
                break
        return current_state

def solve_sokoban_hillclimbing(maze: List[List[int]], 
                               player_pos: Tuple[int, int],
                               boxes: List[Tuple[int, int]], 
                               targets: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    initial_state = SokobanState(tuple(tuple(row) for row in maze), player_pos, frozenset(boxes), frozenset(targets))
    solver = HillClimbingSolver()
    return solver.solve(initial_state)