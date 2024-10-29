from typing import FrozenSet, List, Tuple, Optional, Set
from sokoban_common import SokobanState, MOVES
from collections import deque

class BFSSolver:
    def __init__(self, max_iterations: int = 1000000):
        self.max_iterations = max_iterations

    def solve(self, initial_state: SokobanState) -> Optional[List[Tuple[int, int]]]:
        queue = deque([(initial_state, [])])
        visited = set()

        for _ in range(self.max_iterations):
            if not queue:
                return None
            current_state, path = queue.popleft()
            if current_state.is_goal():
                return path

            state_hash = self.state_to_hashable(current_state)
            if state_hash in visited:
                continue
            visited.add(state_hash)

            for move in MOVES:
                next_state = current_state.apply_move(move)
                if next_state != current_state:
                    queue.append((next_state, path + [move]))

        return None

    @staticmethod
    def state_to_hashable(state: SokobanState) -> Tuple[Tuple[int, int], FrozenSet[Tuple[int, int]]]:
        return (state.player_pos, state.boxes)

def solve_sokoban_bfs(maze: List[List[int]], 
                      player_pos: Tuple[int, int],
                      boxes: List[Tuple[int, int]], 
                      targets: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    initial_state = SokobanState(tuple(tuple(row) for row in maze), player_pos, frozenset(boxes), frozenset(targets))
    solver = BFSSolver()
    return solver.solve(initial_state)