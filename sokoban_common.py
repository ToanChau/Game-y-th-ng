from typing import List, Tuple, Set, Dict, FrozenSet
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment

MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Left, Right, Up, Down

@dataclass(frozen=True)
class SokobanState:
    maze: Tuple[Tuple[int, ...], ...]
    player_pos: Tuple[int, int]
    boxes: FrozenSet[Tuple[int, int]]
    targets: FrozenSet[Tuple[int, int]]
    _zone_map: Dict[Tuple[int, int], int] = None 
    _deadlock_cache: Dict[Tuple[int, int], bool] = None

    def __post_init__(self):
        if self._zone_map is None:
            object.__setattr__(self, '_zone_map', self._create_zone_map())
        if self._deadlock_cache is None:
            object.__setattr__(self, '_deadlock_cache', {})

    def _create_zone_map(self) -> Dict[Tuple[int, int], int]:
        zone_map = {}
        current_zone = 0
        for y in range(len(self.maze)):
            for x in range(len(self.maze[0])):
                if (x, y) not in zone_map and self.maze[y][x] != 1:
                    self._flood_fill(x, y, current_zone, zone_map)
                    current_zone += 1
        return zone_map

    def _flood_fill(self, x: int, y: int, zone: int, zone_map: Dict[Tuple[int, int], int]):
        if not (0 <= x < len(self.maze[0]) and 0 <= y < len(self.maze)):
            return
        if self.maze[y][x] == 1 or (x, y) in zone_map:
            return
        zone_map[(x, y)] = zone
        for dx, dy in MOVES:
            self._flood_fill(x + dx, y + dy, zone, zone_map)

    def is_goal(self) -> bool:
        return self.boxes == self.targets
    
    def get_possible_moves(self) -> List[Tuple[int, int]]:
        return [move for move in MOVES if self._is_valid_move(*self._get_new_position(move))]

    def _get_new_position(self, move: Tuple[int, int]) -> Tuple[int, int]:
        return self.player_pos[0] + move[0], self.player_pos[1] + move[1]

    def _is_valid_move(self, x: int, y: int) -> bool:
        if not self._is_within_bounds(x, y) or self.maze[y][x] == 1:
            return False
        if (x, y) in self.boxes:
            box_new_x, box_new_y = x + (x - self.player_pos[0]), y + (y - self.player_pos[1])
            return self._can_push_box(box_new_x, box_new_y)
        return True

    def _is_within_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < len(self.maze[0]) and 0 <= y < len(self.maze)

    def _can_push_box(self, x: int, y: int) -> bool:
        return (self._is_within_bounds(x, y) and 
                self.maze[y][x] != 1 and 
                (x, y) not in self.boxes and
                not self._is_deadlock_position(x, y))

    def apply_move(self, move: Tuple[int, int]) -> 'SokobanState':
        new_x, new_y = self._get_new_position(move)
        
        if not self._is_valid_move(new_x, new_y):
            return self
        
        new_boxes = set(self.boxes)
        if (new_x, new_y) in new_boxes:
            box_new_x, box_new_y = new_x + move[0], new_y + move[1]
            new_boxes.remove((new_x, new_y))
            new_boxes.add((box_new_x, box_new_y))
        
        return SokobanState(
            self.maze,
            (new_x, new_y),
            frozenset(new_boxes),
            self.targets,
            self._zone_map,
            self._deadlock_cache
        )

    def _is_deadlock_position(self, x: int, y: int) -> bool:
        if (x, y) in self._deadlock_cache:
            return self._deadlock_cache[(x, y)]
        
        is_deadlock = (self._is_corner_deadlock(x, y) or 
                       self._is_line_deadlock(x, y) or 
                       self._is_zone_deadlock(x, y))
        
        self._deadlock_cache[(x, y)] = is_deadlock
        return is_deadlock

    def _is_corner_deadlock(self, x: int, y: int) -> bool:
        if (x, y) in self.targets:
            return False
        
        horizontal_wall = (not self._is_within_bounds(x-1, y) or self.maze[y][x-1] == 1 or
                           not self._is_within_bounds(x+1, y) or self.maze[y][x+1] == 1)
        vertical_wall = (not self._is_within_bounds(x, y-1) or self.maze[y-1][x] == 1 or
                         not self._is_within_bounds(x, y+1) or self.maze[y+1][x] == 1)
        
        return horizontal_wall and vertical_wall

    def _is_line_deadlock(self, x: int, y: int) -> bool:
        if (x, y) in self.targets:
            return False
        
        for axis in ['horizontal', 'vertical']:
            if self._check_line_deadlock(x, y, axis):
                return True
        
        return False

    def _check_line_deadlock(self, x: int, y: int, axis: str) -> bool:
        if axis == 'horizontal':
            if not self._is_within_bounds(x, y-1) or self.maze[y-1][x] != 1 or not self._is_within_bounds(x, y+1) or self.maze[y+1][x] != 1:
                return False
            directions = [(-1, 0), (1, 0)]
        else:  # vertical
            if not self._is_within_bounds(x-1, y) or self.maze[y][x-1] != 1 or not self._is_within_bounds(x+1, y) or self.maze[y][x+1] != 1:
                return False
            directions = [(0, -1), (0, 1)]

        for dx, dy in directions:
            blocked = True
            cx, cy = x, y
            while True:
                cx, cy = cx + dx, cy + dy
                if not self._is_within_bounds(cx, cy) or self.maze[cy][cx] == 1:
                    break
                if (cx, cy) in self.targets:
                    blocked = False
                    break
            if not blocked:
                return False
        return True

    def _is_zone_deadlock(self, x: int, y: int) -> bool:
        if (x, y) not in self._zone_map:
            return True
        
        box_zone = self._zone_map[(x, y)]
        return not any(self._zone_map.get(target, -1) == box_zone for target in self.targets)

    def heuristic(self) -> float:
        if self.is_goal():
            return 0
        
        boxes = list(self.boxes)
        targets = list(self.targets)
        cost_matrix = np.zeros((len(boxes), len(targets)))
        
        for i, box in enumerate(boxes):
            for j, target in enumerate(targets):
                manhattan_dist = abs(box[0] - target[0]) + abs(box[1] - target[1])
                deadlock_penalty = 1000 if self._is_deadlock_position(box[0], box[1]) else 0
                zone_penalty = 500 if self._zone_map.get(box) != self._zone_map.get(target) else 0
                cost_matrix[i][j] = manhattan_dist + deadlock_penalty + zone_penalty
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_ind, col_ind].sum()
        
        min_dist_to_unmatched_box = min(
            abs(self.player_pos[0] - box[0]) + abs(self.player_pos[1] - box[1])
            for box in boxes
        )
        
        return total_cost + min_dist_to_unmatched_box

    def __hash__(self) -> int:
        return hash((self.player_pos, self.boxes))

    def __eq__(self, other: 'SokobanState') -> bool:
        return (self.player_pos == other.player_pos and 
                self.boxes == other.boxes)