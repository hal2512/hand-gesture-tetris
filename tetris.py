import pygame
import time
from gesture_detector import GestureDetector
from collections import deque, Counter
import threading
import random
import cv2
import numpy as np
import mediapipe as mp

WINDOW_WIDTH = 1920
WINDOW_HEIGHT = 1080
PLAY_WIDTH = 300
BLOCK_SIZE = 30
GRID_HEIGHT = 20
GRID_WIDTH = 10
PLAY_HEIGHT = GRID_HEIGHT * BLOCK_SIZE
TOP_LEFT_X = (WINDOW_WIDTH - PLAY_WIDTH) // 2
TOP_LEFT_Y = 200  # lowered grid to add padding from the top and base

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
SHAPE_COLORS = [(0, 255, 255), (255, 255, 0), (128, 0, 128), (0, 255, 0), (255, 0, 0)]

SHAPES = [
    [[1, 1, 1, 1]],
    [[1, 1], [1, 1]],
    [[0, 1, 0], [1, 1, 1]],
    [[1, 1, 0], [0, 1, 1]],
    [[0, 1, 1], [1, 1, 0]],
]

class Piece:
    def __init__(self, shape):
        self.shape = shape
        self.color = SHAPE_COLORS[SHAPES.index(shape)]
        self.x = GRID_WIDTH // 2 - len(shape[0]) // 2
        self.y = 0

    def rotate(self):
        self.shape = list(zip(*self.shape[::-1]))

def create_grid(locked_pos={}):
    grid = [[BLACK for _ in range(GRID_WIDTH)] for _ in range(GRID_HEIGHT)]
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if (x, y) in locked_pos:
                grid[y][x] = locked_pos[(x, y)]
    return grid

def draw_grid(surface, grid):
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            pygame.draw.rect(surface, grid[y][x],
                             (TOP_LEFT_X + x * BLOCK_SIZE, TOP_LEFT_Y + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE), 0)
    for x in range(GRID_WIDTH):
        pygame.draw.line(surface, WHITE, (TOP_LEFT_X + x * BLOCK_SIZE, TOP_LEFT_Y), (TOP_LEFT_X + x * BLOCK_SIZE, TOP_LEFT_Y + PLAY_HEIGHT))
    for y in range(GRID_HEIGHT):
        pygame.draw.line(surface, WHITE, (TOP_LEFT_X, TOP_LEFT_Y + y * BLOCK_SIZE), (TOP_LEFT_X + PLAY_WIDTH, TOP_LEFT_Y + y * BLOCK_SIZE))

def valid_space(shape, grid, offset):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                if x + off_x < 0 or x + off_x >= GRID_WIDTH or y + off_y >= GRID_HEIGHT:
                    return False
                if y + off_y >= 0 and grid[y + off_y][x + off_x] != BLACK:
                    return False
    return True

def merge_shape(shape, locked, offset, color):
    off_x, off_y = offset
    for y, row in enumerate(shape):
        for x, cell in enumerate(row):
            if cell:
                locked[(x + off_x, y + off_y)] = color

def clear_lines(grid, locked):
    lines_to_clear = [y for y in range(GRID_HEIGHT) if all((x, y) in locked for x in range(GRID_WIDTH))]
    if not lines_to_clear:
        return 0

    for y in lines_to_clear:
        for x in range(GRID_WIDTH):
            del locked[(x, y)]

    for y in reversed(range(min(lines_to_clear))):
        for x in range(GRID_WIDTH):
            if (x, y) in locked:
                locked[(x, y + len(lines_to_clear))] = locked.pop((x, y))

    return len(lines_to_clear)

current_gesture = "None"
gesture_last_used = {}
gesture_cooldown = 0.3

def gesture_ready(gesture):
    now = time.time()
    if gesture not in gesture_last_used or (now - gesture_last_used[gesture]) > gesture_cooldown:
        gesture_last_used[gesture] = now
        return True
    return False

def gesture_polling(detector):
    global current_gesture
    while True:
        current_gesture = detector.get_current_gesture()
        time.sleep(0.1)

def draw_camera_overlay(frame, gesture_label):
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7) as hands:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                ih, iw, _ = frame.shape
                xmin, xmax = int(min(x_coords) * iw), int(max(x_coords) * iw)
                ymin, ymax = int(min(y_coords) * ih), int(max(y_coords) * ih)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                cv2.putText(frame, gesture_label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame

def show_start_menu(screen, detector):
    cap = cv2.VideoCapture(0)
    for _ in range(15):
        cap.read()
        time.sleep(0.05)
    cap.release()
    font = pygame.font.SysFont('Arial', 36)
    screen.fill(BLACK)
    text = font.render("Gesture Tetris", True, WHITE)
    text2 = font.render("Raise your hand to start", True, (0, 255, 255))
    screen.blit(text, (WINDOW_WIDTH // 2 - text.get_width() // 2, 200))
    screen.blit(text2, (WINDOW_WIDTH // 2 - text2.get_width() // 2, 260))
    pygame.display.update()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        gesture = detector.get_current_gesture()
        if gesture != "None":
            break

def draw_window(surface, grid, score, next_piece):
    surface.fill(BLACK)
    draw_grid(surface, grid)
    pygame.draw.rect(surface, WHITE, (TOP_LEFT_X, TOP_LEFT_Y, PLAY_WIDTH, PLAY_HEIGHT), 5)
    font = pygame.font.SysFont('Arial', 24)
    label = font.render(f"Score: {score}", True, WHITE)
    surface.blit(label, (TOP_LEFT_X - 200, TOP_LEFT_Y))
    label = font.render("Next:", True, WHITE)
    surface.blit(label, (TOP_LEFT_X - 200, TOP_LEFT_Y + 40))
    for y, row in enumerate(next_piece.shape):
        for x, cell in enumerate(row):
            if cell:
                pygame.draw.rect(surface, next_piece.color, (TOP_LEFT_X - 200 + x * BLOCK_SIZE, TOP_LEFT_Y + 70 + y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE))
    if hasattr(draw_window, 'cap') and draw_window.cap is not None:
        ret, frame = draw_window.cap.read()
        if ret:
            frame = draw_camera_overlay(frame, current_gesture)
            frame = cv2.resize(frame, (500, 375))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            surface.blit(camera_surface, (TOP_LEFT_X + PLAY_WIDTH + 20, TOP_LEFT_Y + 50))

def main():
    global current_gesture
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption('Gesture Tetris')
    clock = pygame.time.Clock()
    detector = GestureDetector()
    draw_window.cap = detector.cap
    show_start_menu(screen, detector)
    grid = create_grid()
    locked_positions = {}
    current_piece = Piece(random.choice(SHAPES))
    next_piece = Piece(random.choice(SHAPES))
    fall_time = 0
    fall_speed = 0.5
    score = 0
    run = True
    gesture_thread = threading.Thread(target=gesture_polling, args=(detector,), daemon=True)
    gesture_thread.start()
    while run:
        grid = create_grid(locked_positions)
        cleared = clear_lines(grid, locked_positions)
        score += cleared * 100
        fall_time += clock.get_rawtime()
        clock.tick()
        speed = 0.1 if current_gesture == "OK" else fall_speed
        if fall_time / 1000 >= speed:
            current_piece.y += 1
            if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                current_piece.y -= 1
                merge_shape(current_piece.shape, locked_positions, (current_piece.x, current_piece.y), current_piece.color)
                cleared = clear_lines(grid, locked_positions)
                score += cleared * 100
                current_piece = next_piece
                next_piece = Piece(random.choice(SHAPES))
                if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                    print("Game Over")
                    run = False
            fall_time = 0
        if current_gesture == "Peace" and gesture_ready("Peace"):
            current_piece.rotate()
            if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                for _ in range(3):
                    current_piece.rotate()
        elif current_gesture == "Index Finger" and gesture_ready("Index Finger"):
            current_piece.x -= 1
            if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                current_piece.x += 1
        elif current_gesture == "Pinkies" and gesture_ready("Pinkies"):
            current_piece.x += 1
            if not valid_space(current_piece.shape, grid, (current_piece.x, current_piece.y)):
                current_piece.x -= 1
        temp_locked = dict(locked_positions)
        merge_shape(current_piece.shape, temp_locked, (current_piece.x, current_piece.y), current_piece.color)
        draw_window(screen, create_grid(temp_locked), score, next_piece)
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
    pygame.quit()

if __name__ == '__main__':
    main()
