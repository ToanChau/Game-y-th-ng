import heapq
from typing import List, Tuple, Optional, FrozenSet
from sokoban_common import SokobanState, MOVES

class OptimizedAStarSolver:
    def __init__(self, max_iterations: int = 1000000):
        self.max_iterations = max_iterations
    
    def solve(self, initial_state: SokobanState) -> Optional[List[Tuple[int, int]]]:
        start_node = (initial_state.heuristic(), 0, [], initial_state)
        frontier = [start_node]
        explored = set()
        deadlock_cache = {}

        for _ in range(self.max_iterations):
            if not frontier:
                return None

            _, cost, path, current_state = heapq.heappop(frontier)

            if current_state.is_goal():
                return path

            state_hash = self.state_to_hashable(current_state)
            if state_hash in explored:
                continue
            explored.add(state_hash)

            for move in MOVES:
                next_state = current_state.apply_move(move)
                if next_state != current_state and not self.is_deadlock(next_state, deadlock_cache):
                    next_cost = cost + 1
                    next_heuristic = next_state.heuristic()
                    next_node = (next_cost + next_heuristic, next_cost, path + [move], next_state)
                    heapq.heappush(frontier, next_node)

        return None

    @staticmethod
    def state_to_hashable(state: SokobanState) -> Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]:
        return (state.player_pos, state.boxes)

    def is_deadlock(self, state: SokobanState, deadlock_cache: dict) -> bool:
        state_hash = self.state_to_hashable(state)
        if state_hash in deadlock_cache:
            return deadlock_cache[state_hash]

        is_deadlock = self.check_deadlock(state)
        deadlock_cache[state_hash] = is_deadlock
        return is_deadlock

    def check_deadlock(self, state: SokobanState) -> bool:
        for box in state.boxes:
            if self.is_corner_deadlock(state, box):
                return True
        return False

    def is_corner_deadlock(self, state: SokobanState, box: Tuple[int, int]) -> bool:
        x, y = box
        if box in state.targets:
            return False
        
        walls = [
            (state.maze[y-1][x] == 1 and state.maze[y][x-1] == 1),
            (state.maze[y-1][x] == 1 and state.maze[y][x+1] == 1),
            (state.maze[y+1][x] == 1 and state.maze[y][x-1] == 1),
            (state.maze[y+1][x] == 1 and state.maze[y][x+1] == 1)
        ]
        
        return any(walls)

def solve_sokoban_astar(maze: List[List[int]], 
                                  player_pos: Tuple[int, int],
                                  boxes: List[Tuple[int, int]], 
                                  targets: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    initial_state = SokobanState(tuple(tuple(row) for row in maze), player_pos, frozenset(boxes), frozenset(targets))
    solver = OptimizedAStarSolver()
    return solver.solve(initial_state)