import threading
import pygame
import sys
import time
from typing import List, Tuple, Optional
from sokoban_common import SokobanState, MOVES
from astar_solver import solve_sokoban_astar
from hill_climbing_solver import solve_sokoban_hillclimbing
from bfs_solver import solve_sokoban_bfs

class SokobanGame:
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    GREEN =(34,139,34)
    RED = (220,20,60)
    LIGHT_BLUE = (173, 216, 230)
    DARK_BLUE = (0, 0, 139)
    BACKGROUND = (255, 160, 122)
# Khởi tạo
    def __init__(self, window_size: Tuple[int, int] = (1100, 600)):
        pygame.init()
        self.window_size = window_size
        self.screen = pygame.display.set_mode(window_size)
        pygame.display.set_caption("Sokoban Game")
 
        self.cell_size = 50
        self.load_assets()
        self.maps = self.load_maps('maps.txt')
        self.current_algorithm = 'astar'
        self.algorithms = ['astar', 'hillclimbing', 'bfs']
        self.player_direction = 'ghost' 
        self.game_time = 0
        self.start_time = None
        self.pause_time = 0
        self.is_paused = False
        self.solve_time = 0
        self.solve_steps = 0
        self.best_times = self.load_best_times()
        self.best_steps = self.load_best_steps()
        self.player_moves = []
        self.player_moves_by_space = []
# Tài nguyên
    def load_assets(self):
        assets = ['wall', 'ghost', 'box', 'target', 'trai', 'phai', 'tren', 'duoi']
        self.images = {}
        for asset in assets:
            image = pygame.image.load(f'img/{asset}.png')
            self.images[asset] = pygame.transform.scale(image, (self.cell_size, self.cell_size))  
# Thời gian
    # Lấy thời gian tốt nhất
    def load_best_times(self) -> dict:
        best_times = {}
        with open('BestTime.txt', 'r') as file:
            current_map = -1
            for line in file:
                line = line.strip()
                if line.startswith('#'):
                    current_map += 1
                elif line.startswith('Best Time:'):
                    try:
                        best_time = float(line.split(':')[1].strip())
                        best_times[current_map] = best_time
                    except ValueError:
                        pass
        return best_times
    # Lưu thời gain tốt nhất
    def save_best_times(self):
        lines = []
        with open('BestTime.txt', 'r') as file:
            lines = file.readlines()

        current_map = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                current_map += 1
                if current_map in self.best_times:
                    # Check if the next line is a Best Time line
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith('Best Time:'):
                        lines[i + 1] = f'Best Time: {self.best_times[current_map]:.5f}\n'
                    else:
                        lines.insert(i + 1, f'Best Time: {self.best_times[current_map]:.5f}\n')

        with open('BestTime.txt', 'w') as file:
            file.writelines(lines)
# Số bước giải
    # Lấy số lượng bước giải tốt nhất
    def load_best_steps(self) -> dict:
        best_steps = {}
        with open('BestSteps.txt', 'r') as file:
            current_map = -1
            for line in file:
                line = line.strip()
                if line.startswith('#'):
                    current_map += 1
                elif line.startswith('Best step:'):
                    try:
                        best_step = int(line.split(':')[1].strip())
                        best_steps[current_map] = best_step
                    except ValueError:
                        pass
        return best_steps
    # Lưu số lượng bước giải tốt nhất
    def save_best_steps(self):
        lines = []
        with open('BestSteps.txt', 'r') as file:
            lines = file.readlines()

        current_map = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('#'):
                current_map += 1
                if current_map in self.best_steps:
                    if i + 1 < len(lines) and lines[i + 1].strip().startswith('Best step:'):
                        lines[i + 1] = f'Best step: {self.best_steps[current_map]}\n'
                    else:
                        lines.insert(i + 1, f'Best step: {self.best_steps[current_map]}\n')

        with open('BestSteps.txt', 'w') as file:
            file.writelines(lines)
# Giải pháp
    # chuyển bước đi thành dạng kí tự
    def move_to_string(self, move: Tuple[int, int]) -> str:
        """Chuyển đổi tuple move thành ký tự hướng di chuyển"""
        move_map = {
            (-1, 0): 'L',
            (1, 0): 'R',
            (0, -1): 'U',
            (0, 1): 'D'
        }
        return move_map.get(move, '')
    # Lưu giải pháp
    def save_solution(self, map_index: int):
        """Lưu giải pháp vào file solution tương ứng"""
        if not self.player_moves and not self.player_moves_by_space:
            return

        # Tạo chuỗi cho cả hai loại giải pháp
        solutions_to_save = []
        
        if self.player_moves:
            solution_str = ' '.join(self.move_to_string(move) for move in self.player_moves)
            solutions_to_save.append(solution_str)
        
        if self.player_moves_by_space:
            solution_str_by_space = ' '.join(self.move_to_string(move) for move in self.player_moves_by_space)
            if solution_str_by_space not in solutions_to_save:
                solutions_to_save.append(solution_str_by_space)

        # Đọc các giải pháp hiện có
        existing_solutions = set()
        try:
            with open(f'solution/solution_{map_index + 1}.txt', 'r') as file:
                content = file.read().strip()
                if content:
                    existing_solutions = set(solution.strip() for solution in content.split('#') if solution.strip())
        except FileNotFoundError:
            pass
        new_solutions = [sol for sol in solutions_to_save if sol not in existing_solutions]
        
        if new_solutions:
            with open(f'solution/solution_{map_index + 1}.txt', 'a') as file:
                for solution in new_solutions:
                    if existing_solutions:  # Nếu file không trống, thêm dấu # trước
                        file.write('\n')
                    file.write(f'{solution}\n#')                
    # Sắp xếp giải pháp
    def sort_solutions(self,map_index):
        file_path = f'solution/solution_{map_index+1}.txt'
        try:
        # Đọc nội dung file
            with open(file_path, 'r') as file:
                content = file.read().strip()
                
                # Kiểm tra nếu file trống, bỏ qua
                if not content:
                    # Thay vì return, ta dùng pass để chỉ bỏ qua file này
                    pass  
                
                else:
                    # Tách các dãy kí tự bằng dấu '#' và loại bỏ khoảng trắng thừa
                    sequences = [seq.strip() for seq in content.split('#') if seq.strip()]
                    
                    # Sắp xếp các dãy theo độ dài (không tính khoảng trắng)
                    sorted_sequences = sorted(sequences, key=lambda x: len(x.replace(" ", "")))
                    
                    # Ghi các dãy đã sắp xếp trở lại file
                    with open(file_path, 'w') as file:
                        file.write('\n#\n'.join(sorted_sequences))  # Kết hợp các dãy đã sắp xếp với '#' giữa các dãy
                        file.write('\n#\n')  # Đảm bảo định dạng cuối file giống như ban đầu
        
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Đã xảy ra lỗi: {e}")
    # Đọc các giải pháp
    @staticmethod
    def read_solutions(map_index):
        try:
            with open(f'solution/solution_{map_index + 1}.txt', 'r') as file:
                solutions = file.read().split('#')
            return [solution.strip().split() for solution in solutions if solution.strip()]
        except FileNotFoundError:
            return []
    # Chuyển giải pháp về đúng dạng
    @staticmethod
    def parse_solution(solution):
        move_dict = {
            'L': (-1, 0),
            'R': (1, 0),
            'U': (0, -1),
            'D': (0, 1)
        }
        return [move_dict[move] for move in solution]
# Map
    # Đọc map
    @staticmethod
    def load_maps(filename: str) -> List[List[str]]:
        maps = []
        current_map = []
        with open(filename, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith('#'):
                    if current_map:
                        maps.append(current_map)
                        current_map = []
                elif line:
                    current_map.append(line)
            if current_map:
                maps.append(current_map)
        return maps
    # Chuyển map thành mê cung
    @staticmethod
    def map_to_game_state(map_data: List[str]) -> Tuple[List[List[int]], Tuple[int, int], List[Tuple[int, int]], List[Tuple[int, int]]]:
        maze = []
        player_pos = None
        boxes = []
        targets = []
        
        for y, row in enumerate(map_data):
            maze_row = []
            for x, cell in enumerate(row):
                if cell == 'W':
                    maze_row.append(1)
                elif cell == 'P':
                    maze_row.append(0)
                    player_pos = (x, y)
                elif cell == 'B':
                    maze_row.append(0)
                    boxes.append((x, y))
                elif cell == 'T':
                    maze_row.append(0)
                    targets.append((x, y))
                else:
                    maze_row.append(0)
            maze.append(maze_row)
        return maze, player_pos, boxes, targets
    # Vẽ mê cung
    def draw_maze(self, maze: List[List[int]], boxes: List[Tuple[int, int]], targets: List[Tuple[int, int]]):
        self.screen.fill(self.BACKGROUND)
        for y, row in enumerate(maze):
            for x, cell in enumerate(row):
                pos = (x * self.cell_size, y * self.cell_size)
                if cell == 1:
                    self.screen.blit(self.images['wall'], pos)
                
        for target in targets:
            pos = (target[0] * self.cell_size, target[1] * self.cell_size)
            self.screen.blit(self.images['target'], pos)
            
        for box in boxes:
            pos = (box[0] * self.cell_size, box[1] * self.cell_size)
            self.screen.blit(self.images['box'], pos)

        self.draw_info_panel()
    # Vẽ khung chức năng
    def draw_info_panel(self):
        panel_rect = pygame.Rect(800, 0, 300, self.window_size[1])
        pygame.draw.rect(self.screen, self.GRAY, panel_rect)

        font = pygame.font.Font(None, 32)
        
        # Vẽ thời gian 
        time_text = font.render(f"Time: {self.game_time:.5f}", True, self.BLACK)
        self.screen.blit(time_text, (820, 20))
        time_text = font.render(f"Steps: {self.solve_steps}", True, self.BLACK)
        self.screen.blit(time_text, (820, 50))    
        # Vẽ thời gian tốt nhất
        current_map = self.maps.index(self.current_map)
        best_time = self.best_times.get(current_map, float('inf'))
        best_step = self.best_steps.get(current_map, float('inf'))
        if best_time != float('inf'):
            best_time_text = font.render(f"Best time: {best_time:.5f}", True, self.BLACK)
            self.screen.blit(best_time_text, (820, 80))
        if best_step != float('inf'):
            best_step_text = font.render(f"Best step: {best_step}", True, self.BLACK)
            self.screen.blit(best_step_text, (820, 110))
        # Vẽ  thuât toán
        algorithm_text = font.render(f"Algorithm: {self.current_algorithm.upper()}", True, self.BLACK)
        self.screen.blit(algorithm_text, (820, 140))
        # Vẽ nút
        button_texts = ["Pause (P)" , "Reset (R)" , "Select (Esc)" , "Quit"]
        for i, text in enumerate(button_texts):
            button_rect = pygame.Rect(820, 220 + i*60, 260, 50)
            pygame.draw.rect(self.screen, self.LIGHT_BLUE, button_rect)
            button_text = font.render(text, True, self.BLACK)
            text_rect = button_text.get_rect(center=button_rect.center)
            self.screen.blit(button_text, text_rect)
    # Vẽ nút
    def draw_button(self, text: str, rect: pygame.Rect, color: Tuple[int, int, int], centered: bool = False):
        pygame.draw.rect(self.screen, color, rect)
        font = pygame.font.Font(None, 32)
        text_surface = font.render(text, True, self.BLACK)
        if centered:
            text_rect = text_surface.get_rect(center=rect.center)
        else:
            text_rect = text_surface.get_rect(topleft=(rect.x + 10, rect.y + 10))
        self.screen.blit(text_surface, text_rect)
    # Vẽ map (Xem trước)
    def draw_map_preview(self, map_data: List[str], preview_rect: pygame.Rect):
        map_width = len(map_data[0])
        map_height = len(map_data)
        cell_width = preview_rect.width // map_width
        cell_height = preview_rect.height // map_height

        for row_index, row in enumerate(map_data):
            for col_index, cell in enumerate(row):
                rect = pygame.Rect(
                    preview_rect.x + col_index * cell_width,
                    preview_rect.y + row_index * cell_height,
                    cell_width, cell_height
                )
                if cell == 'W':
                    pygame.draw.rect(self.screen, self.GRAY, rect)
                elif cell == 'P':
                    pygame.draw.rect(self.screen, self.LIGHT_BLUE, rect)
                elif cell == 'B':
                    pygame.draw.rect(self.screen, self.DARK_BLUE, rect)
                elif cell == 'T':
                    pygame.draw.circle(self.screen, self.BLACK, rect.center, min(cell_width, cell_height) // 2 - 1)
    # Vẽ phần chọn map
    def map_selection_screen(self) -> Tuple[int, str]:
        current_map = 0
        preview_width, preview_height = 400, 300
        preview_x = (self.window_size[0] - preview_width) // 2
        preview_y = 150
        preview_rect = pygame.Rect(preview_x, preview_y, preview_width, preview_height)

        buttons = {
            'left': pygame.Rect(preview_x - 60, preview_y + preview_height // 2 - 25, 50, 50),
            'right': pygame.Rect(preview_x + preview_width + 10, preview_y + preview_height // 2 - 25, 50, 50),
            'select': pygame.Rect(self.window_size[0] // 2 - 50, preview_y + preview_height + 20, 100, 50),
            'algorithm': pygame.Rect(self.window_size[0] // 2 - 100, preview_y + preview_height + 80, 200, 50) 
        }

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_pos = pygame.mouse.get_pos()
                    if buttons['left'].collidepoint(mouse_pos):
                        current_map = (current_map - 1) % len(self.maps)
                    elif buttons['right'].collidepoint(mouse_pos):
                        current_map = (current_map + 1) % len(self.maps)
                    elif buttons['select'].collidepoint(mouse_pos):
                        return current_map, self.current_algorithm
                    elif buttons['algorithm'].collidepoint(mouse_pos):
                        current_index = self.algorithms.index(self.current_algorithm)
                        next_index = (current_index + 1) % len(self.algorithms)
                        self.current_algorithm = self.algorithms[next_index]
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        current_map = (current_map - 1) % len(self.maps)
                    elif event.key == pygame.K_RIGHT:
                        current_map = (current_map + 1) % len(self.maps)
                    elif event.key == pygame.K_UP or event.key == pygame.K_DOWN:
                        current_index = self.algorithms.index(self.current_algorithm)
                        next_index = (current_index + 1) % len(self.algorithms)
                        self.current_algorithm = self.algorithms[next_index]
                    elif event.key == pygame.K_RETURN:
                        return current_map, self.current_algorithm
                    elif event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        sys.exit()

            self.screen.fill(self.WHITE)
            font = pygame.font.Font(None, 40)
            title = font.render(f"Select a Map ({current_map + 1}/{len(self.maps)})", True, self.BLACK)
            self.screen.blit(title, (self.window_size[0] // 2 - title.get_width() // 2, 50))

            pygame.draw.rect(self.screen, self.BLACK, preview_rect.inflate(4, 4), 2)
            self.draw_map_preview(self.maps[current_map], preview_rect)

            self.draw_button("<", buttons['left'], self.GRAY, True)
            self.draw_button(">", buttons['right'], self.GRAY, True)
            self.draw_button("Select", buttons['select'], self.LIGHT_BLUE, True)
            self.draw_button(f"{self.current_algorithm.upper()}", buttons['algorithm'], self.LIGHT_BLUE, True)

            best_time = self.best_times.get(current_map, float('inf'))
            best_step = self.best_steps.get(current_map, float('inf'))
            if best_time != float('inf'):
                best_time_text = font.render(f"Best Time: {best_time:.5f}", True, self.BLACK)
                self.screen.blit(best_time_text, (preview_width*2 + 30 , preview_y ))
            if best_step!= float('inf'):
                best_step_text = font.render(f"Best step: {best_step}", True, self.BLACK)
                self.screen.blit(best_step_text, (preview_width*2 + 30 , preview_y + 40))
            pygame.display.flip()
    # Thay đổi hướng nhân vật
    def update_player_direction(self, move):
        if move == (-1, 0):
            self.player_direction = 'trai'
        elif move == (1, 0):
            self.player_direction = 'phai'
        elif move == (0, -1):
            self.player_direction = 'tren'
        elif move == (0, 1):
            self.player_direction = 'duoi'
    # Hoạt ảnh giải pháp
    def animate_solution(self, initial_state: SokobanState, solution: List[Tuple[int, int]]):
        current_state = initial_state
        for move in solution:
            self.solve_steps += 1
            self.player_moves_by_space.append(move)
            self.player_moves.append(move)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.update_player_direction(move)  
            current_state = current_state.apply_move(move)
            self.game_time = time.time() - self.start_time - self.pause_time
            self.draw_maze(current_state.maze, list(current_state.boxes), list(current_state.targets))
            player_pos = (current_state.player_pos[0] * self.cell_size, 
                        current_state.player_pos[1] * self.cell_size)
            self.screen.blit(self.images[self.player_direction], player_pos)
            font = pygame.font.Font(None, 36)
            time_text_solve = font.render(f"Solve Time: {self.solve_time:.5f}", True, self.GREEN)
            self.screen.blit(time_text_solve, (820, 500))
            pygame.display.flip()
            pygame.time.wait(100)
            if current_state.is_goal():
                self.handle_level_complete()
                return "COMPLETE"
    # Hành động hoàn thành trò chơi
    def handle_level_complete(self):
        font = pygame.font.Font(None, 74)
        text = font.render('Level Complete!', True, self.GREEN)
        self.screen.blit(text, (200, 250))
        
        current_map = self.maps.index(self.current_map)
        if current_map not in self.best_times or self.game_time <= self.best_times[current_map]:
            self.best_times[current_map] = self.game_time
            self.save_best_times()
        if current_map not in self.best_steps or self.solve_steps < self.best_steps[current_map]:
            self.best_steps[current_map] = self.solve_steps
            self.save_best_steps() 
        self.save_solution(current_map)
        self.sort_solutions(current_map)

        instruction_font = pygame.font.Font(None, 36)
        instruction_text = instruction_font.render('Press SPACE to restart', True, self.WHITE)
        self.screen.blit(instruction_text, (270, 350))
        
        pygame.display.flip()
    # Dừng thời gian
    def toggle_pause(self):
        if self.is_paused:
            self.start_time += time.time() - self.pause_start_time
        else:
            self.pause_start_time = time.time()
        self.is_paused = not self.is_paused
    # Sự kiện click chuột
    def handle_mouse_click(self, pos: Tuple[int, int], first_move: bool) -> Optional[str]:
        if 820 <= pos[0] <= 1080:
            if 220 <= pos[1] <= 270:  # Pause button
                if first_move:
                    self.is_paused = not self.is_paused
                else:
                    self.toggle_pause()
            elif 280 <= pos[1] <= 330:  # Reset button
                return "RESET"
            elif 340 <= pos[1] <= 390:  # Select button
                return "SELECT"
            elif 400 <= pos[1] <= 450:  # Quit button
                return "QUIT"
        return None
    # Sự kiện nhấn nút
    def handle_key_press(self, event: pygame.event.Event, current_state: SokobanState, first_move: bool) -> Tuple[Optional[str], SokobanState, bool]:
        if event.key == pygame.K_r:
            return "RESET", current_state, first_move
        elif event.key == pygame.K_ESCAPE:
            return "SELECT", current_state, first_move
        elif event.key == pygame.K_p:
            self.toggle_pause()
            return None, current_state, first_move
        elif event.key == pygame.K_SPACE:
            self.player_moves_by_space = []
            result = self.handle_solution(current_state)
            if result == "COMPLETE":
                return "COMPLETE", current_state, first_move
            return None, current_state, first_move
        
        if not self.is_paused and not first_move:
            new_state, moved = self.move_player(event.key, current_state)
            if moved:
                self.solve_steps += 1
                return None, new_state, first_move
        return None, current_state, first_move
    # Di chuyển nhân vật
    def move_player(self, key: int, current_state: SokobanState) -> Tuple[SokobanState, bool]:
        direction = {
            pygame.K_LEFT: (-1, 0, 'trai'),
            pygame.K_RIGHT: (1, 0, 'phai'),
            pygame.K_UP: (0, -1, 'tren'),
            pygame.K_DOWN: (0, 1, 'duoi')
        }.get(key)
        
        if not direction:
            return current_state, False

        dx, dy, new_direction = direction
        new_x = current_state.player_pos[0] + dx
        new_y = current_state.player_pos[1] + dy
        
        if not (0 <= new_x < len(current_state.maze[0]) and 
                0 <= new_y < len(current_state.maze) and
                current_state.maze[new_y][new_x] != 1):
            return current_state, False

        new_boxes = set(current_state.boxes)
        
        if (new_x, new_y) in current_state.boxes:
            box_new_x = new_x + dx
            box_new_y = new_y + dy
            
            if not (0 <= box_new_x < len(current_state.maze[0]) and 
                    0 <= box_new_y < len(current_state.maze) and
                    current_state.maze[box_new_y][box_new_x] != 1 and
                    (box_new_x, box_new_y) not in current_state.boxes):
                return current_state, False
            
            new_boxes.remove((new_x, new_y))
            new_boxes.add((box_new_x, box_new_y))
        
        self.player_direction = new_direction  # Cập nhật hướng của nhân vật
        self.player_moves.append((dx, dy))
        return SokobanState(
            current_state.maze,
            (new_x, new_y),
            frozenset(new_boxes),
            current_state.targets,
            current_state._zone_map,
            current_state._deadlock_cache
        ), True
    # Giới hạn thời gian tìm kiếm
    def solve_with_timeout(self, solver_func, *args):
        solution = [None]
        def target():
            solution[0] = solver_func(*args)
        
        thread = threading.Thread(target=target)
        thread.start()
        thread.join(30)  # Đợi tối đa 30 giây
        
        if thread.is_alive():
            # Nếu thread vẫn đang chạy sau 30 giây
            return None
        return solution[0]
    # Tìm kếm giải pháp
    def handle_solution(self, current_state: SokobanState) -> Optional[str]:
        solve_start_time = time.time()
        font = pygame.font.Font(None, 74)
        text = font.render('Solving ...', True, self.WHITE)
        self.screen.blit(text, (300, 250))
        pygame.display.flip()
        if self.current_algorithm == 'astar':
            solution = self.solve_with_timeout(solve_sokoban_astar, current_state.maze, current_state.player_pos,
                                               list(current_state.boxes), list(current_state.targets))
        elif self.current_algorithm == 'hillclimbing':
            solution = self.solve_with_timeout(solve_sokoban_hillclimbing, current_state.maze, current_state.player_pos,
                                               list(current_state.boxes), list(current_state.targets))
        elif self.current_algorithm == 'bfs':
            solution = self.solve_with_timeout(solve_sokoban_bfs, current_state.maze, current_state.player_pos,
                                               list(current_state.boxes), list(current_state.targets))

        self.solve_time = time.time() - solve_start_time
        
        if solution:
            self.start_time = time.time()
            for move in solution:
                self.update_player_direction(move)  # Cập nhật hướng của nhân vật cho mỗi bước di chuyển
            result = self.animate_solution(current_state, solution)
            if result == "COMPLETE":
                return "COMPLETE"
        else:
            current_map = self.maps.index(self.current_map)
            file_solutions = self.read_solutions(current_map)
            for file_solution in file_solutions:
                parsed_solution = self.parse_solution(file_solution) 
                if self.is_valid_solution(current_state,parsed_solution):
                    self.start_time = time.time()
                    for move in parsed_solution:
                        self.update_player_direction(move)
                    result = self.animate_solution(current_state, parsed_solution)
                    if result == "COMPLETE":
                        return "COMPLETE"
            self.handle_no_solution(current_state)
        return None
    # Kiểm tra khả năng giải
    def is_valid_solution(self, state: SokobanState, solution: List[Tuple[int, int]]) -> bool:
        for move in solution:
            new_state = state.apply_move(move)
            if new_state == state:  # If the state didn't change, the move was invalid
                return False
            state = new_state
        return state.is_goal()
    # Hành động không hoàn thành trò chơi
    def handle_no_solution(self, current_state: SokobanState):
        self.game_time = 0
        self.draw_maze(current_state.maze, list(current_state.boxes), list(current_state.targets))
        player_pos = (current_state.player_pos[0] * self.cell_size, 
                     current_state.player_pos[1] * self.cell_size)
        self.screen.blit(self.images['ghost'], player_pos)
        font = pygame.font.Font(None, 74)
        text = font.render('No solution found!', True, self.RED)
        self.screen.blit(text, (300, 250))
        pygame.display.flip()
        pygame.time.wait(2000)
    # Cập nhật hình của màn chơi
    def update_game_state(self, current_state: SokobanState, first_move: bool):
        if not self.is_paused:
            self.draw_maze(current_state.maze, list(current_state.boxes), list(current_state.targets))
            player_pos = (current_state.player_pos[0] * self.cell_size, 
                        current_state.player_pos[1] * self.cell_size)
            self.screen.blit(self.images[self.player_direction], player_pos)
            
            if current_state.is_goal():
                self.handle_level_complete()
                return "COMPLETE"
            
            if not first_move:
                self.game_time = time.time() - self.start_time - self.pause_time
        else:
            font = pygame.font.Font(None, 74)
            text = font.render('PAUSED', True, self.WHITE)
            text_rect = text.get_rect(center=(self.window_size[0] // 2, self.window_size[1] // 2))
            self.screen.blit(text, text_rect)
        return None

    
    def play_game(self, initial_state: SokobanState) -> str:
        current_state = initial_state
        clock = pygame.time.Clock()
        first_move = True
        
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return "QUIT"
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        result = self.handle_mouse_click(event.pos, first_move)
                        if result:
                            return result
                elif event.type == pygame.KEYDOWN:
                    if first_move and event.key in [pygame.K_LEFT, pygame.K_RIGHT, pygame.K_UP, pygame.K_DOWN]:
                        self.start_time = time.time()
                        first_move = False
                    
                    result, current_state, first_move = self.handle_key_press(event, current_state, first_move)
                    if result:
                        return result
            
            result = self.update_game_state(current_state, first_move)
            if result:
                return result
            
            pygame.display.flip()
            clock.tick(60)
    
    def run(self):
        while True:
            selected_map, algorithm = self.map_selection_screen()
            self.current_map = self.maps[selected_map]
            maze, player_pos, boxes, targets = self.map_to_game_state(self.current_map)
            initial_state = SokobanState(tuple(tuple(row) for row in maze), player_pos, frozenset(boxes), frozenset(targets))
            
            font = pygame.font.Font(None, 36)
            text = font.render('Press SPACE to solve automatically', True, self.BLACK)
            self.screen.blit(text, (330, 100))
            pygame.display.flip()
            pygame.time.wait(100)

            self.game_time = 0
            self.start_time = None
            self.pause_time = 0
            self.is_paused = False
            self.solve_steps = 0
            self.player_moves = []
            self.player_moves_by_space = []

            while True:
                result = self.play_game(initial_state)
                if result == "QUIT":
                    self.save_best_times()
                    self.save_best_steps()
                    pygame.quit()
                    sys.exit()
                elif result == "RESET":
                    current_algorithm = self.current_algorithm  
                    self.__init__()
                    self.current_algorithm = current_algorithm
                    pygame.display.flip()      
                elif result == "SELECT": 
                    break
                elif result == "COMPLETE":
                    waiting_for_input = True
                    while waiting_for_input:
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                self.save_best_times()
                                self.save_best_steps()
                                pygame.quit()
                                sys.exit()
                            if event.type == pygame.KEYDOWN:
                                if event.key == pygame.K_SPACE:
                                    current_algorithm = self.current_algorithm
                                    self.save_best_times()
                                    self.save_best_steps()
                                    self.__init__()
                                    self.current_algorithm = current_algorithm
                                    waiting_for_input = False
                                    break
                    
if __name__ == "__main__":
    game = SokobanGame()
    game.run()