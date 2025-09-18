"""
Tic Tac Toe Environment for VLA Training

This module implements a Tic Tac Toe environment using pygame and gymnasium,
supporting human interaction, visual observations, and AI opponents of different difficulties.
"""

import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces
import random
from typing import Optional, Dict, Any, Tuple, Union
from enum import Enum


class Player(Enum):
    """Players in the game"""
    EMPTY = 0
    X = 1  # Human/Agent player
    O = 2  # AI opponent


class Difficulty(Enum):
    """AI opponent difficulty levels"""
    RANDOM = "random"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class TicTacToeEnv(gym.Env):
    """
    Tic Tac Toe environment with pygame rendering and gymnasium interface.
    
    Observation space: RGB image of the game board (300x300x3)
    Action space: Discrete(9) - positions 0-8 on the 3x3 grid
    
    Features:
    - Visual rendering with pygame
    - Human interaction via mouse clicks
    - AI opponents with different difficulty levels
    - Gymnasium interface compliance
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        difficulty: Difficulty = Difficulty.MEDIUM,
        human_player: bool = False,
        board_size: int = 300,
        use_internal_ai: bool = True,
    ):
        """
        Initialize the Tic Tac Toe environment.
        
        Args:
            render_mode: Rendering mode ("human" or "rgb_array")
            difficulty: AI opponent difficulty level
            human_player: Whether to enable human interaction
            board_size: Size of the game board in pixels
        """
        super().__init__()
        
        # Environment configuration
        self.render_mode = render_mode
        self.difficulty = difficulty
        self.human_player = human_player
        self.board_size = board_size
        self.cell_size = board_size // 3
        # Whether to let environment control O automatically. Set to False when an external agent (e.g., VLA) plays O.
        self.use_internal_ai = use_internal_ai
        
        # Gymnasium spaces
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(board_size, board_size, 3), dtype=np.uint8
        )
        self.action_space = spaces.Discrete(9)
        
        # Game state
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.game_over = False
        self.winner = None
        self.human_action = None
        
        # Pygame setup
        self.window = None
        self.clock = None
        
        # Colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'gray': (128, 128, 128),
            'blue': (0, 0, 255),
            'red': (255, 0, 0),
            'green': (0, 255, 0)
        }
        
        # Reset the environment
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Reset game state
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = Player.X
        self.game_over = False
        self.winner = None
        self.human_action = None
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: Union[int, None] = None) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (0-8 for grid positions, None for human input)
            
        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.game_over:
            return self._get_observation(), 0.0, True, False, self._get_info()
        
        # Handle human player input
        if self.human_player and self.current_player == Player.X:
            action = self._get_human_action()
            if action is None:
                # No action taken yet, return current state
                return self._get_observation(), 0.0, False, False, self._get_info()
        
        # Validate and execute action
        reward = 0.0
        if action is not None and self._is_valid_action(action):
            row, col = divmod(action, 3)
            self.board[row, col] = self.current_player.value
            
            # Check for game end
            if self._check_winner():
                self.game_over = True
                if self.winner == Player.X:
                    reward = 1.0  # Agent wins
                elif self.winner == Player.O:
                    reward = -1.0  # AI wins
            elif self._is_board_full():
                self.game_over = True
                reward = 0.0  # Draw
            else:
                # Switch player
                self.current_player = Player.O if self.current_player == Player.X else Player.X
                
                # If it's AI's turn, make AI move (only when internal AI is enabled)
                if self.use_internal_ai and self.current_player == Player.O and not self.game_over:
                    ai_action = self._get_ai_action()
                    if ai_action is not None:
                        ai_row, ai_col = divmod(ai_action, 3)
                        self.board[ai_row, ai_col] = Player.O.value
                        
                        # Check for game end after AI move
                        if self._check_winner():
                            self.game_over = True
                            if self.winner == Player.O:
                                reward = -1.0  # AI wins
                        elif self._is_board_full():
                            self.game_over = True
                            reward = 0.0  # Draw
                        else:
                            self.current_player = Player.X
        else:
            # Invalid action penalty
            reward = -0.1
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, self.game_over, False, info
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode is None:
            return None
            
        return self._render_frame()
    
    def close(self):
        """Close the environment."""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation (RGB image of the board)."""
        return self._render_frame()
    
    def _get_info(self) -> Dict:
        """Get additional information about the current state."""
        return {
            'board': self.board.copy(),
            'current_player': self.current_player.value,
            'game_over': self.game_over,
            'winner': self.winner.value if self.winner else None,
            'valid_actions': self._get_valid_actions()
        }
    
    def _is_valid_action(self, action: int) -> bool:
        """Check if an action is valid."""
        if not 0 <= action <= 8:
            return False
        row, col = divmod(action, 3)
        return self.board[row, col] == Player.EMPTY.value
    
    def _get_valid_actions(self) -> list:
        """Get list of valid actions."""
        valid_actions = []
        for i in range(9):
            if self._is_valid_action(i):
                valid_actions.append(i)
        return valid_actions
    
    def _check_winner(self) -> bool:
        """Check if there's a winner and update game state."""
        # Check rows
        for row in range(3):
            if (self.board[row, 0] == self.board[row, 1] == self.board[row, 2] != Player.EMPTY.value):
                self.winner = Player(self.board[row, 0])
                return True
        
        # Check columns
        for col in range(3):
            if (self.board[0, col] == self.board[1, col] == self.board[2, col] != Player.EMPTY.value):
                self.winner = Player(self.board[0, col])
                return True
        
        # Check diagonals
        if (self.board[0, 0] == self.board[1, 1] == self.board[2, 2] != Player.EMPTY.value):
            self.winner = Player(self.board[0, 0])
            return True
        
        if (self.board[0, 2] == self.board[1, 1] == self.board[2, 0] != Player.EMPTY.value):
            self.winner = Player(self.board[0, 2])
            return True
        
        return False
    
    def _is_board_full(self) -> bool:
        """Check if the board is full."""
        return np.all(self.board != Player.EMPTY.value)
    
    def _get_human_action(self) -> Optional[int]:
        """Get action from human player via mouse click."""
        if not pygame.get_init():
            return None
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return None
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    mouse_x, mouse_y = event.pos
                    col = mouse_x // self.cell_size
                    row = mouse_y // self.cell_size
                    if 0 <= row < 3 and 0 <= col < 3:
                        action = row * 3 + col
                        if self._is_valid_action(action):
                            return action
        
        return None
    
    def _get_ai_action(self) -> Optional[int]:
        """Get action from AI opponent based on difficulty level."""
        valid_actions = self._get_valid_actions()
        if not valid_actions:
            return None
        
        if self.difficulty == Difficulty.RANDOM:
            return random.choice(valid_actions)
        
        elif self.difficulty == Difficulty.EASY:
            # 70% random, 30% strategic
            if random.random() < 0.7:
                return random.choice(valid_actions)
            else:
                return self._get_strategic_action(valid_actions)
        
        elif self.difficulty == Difficulty.MEDIUM:
            # 30% random, 70% strategic
            if random.random() < 0.3:
                return random.choice(valid_actions)
            else:
                return self._get_strategic_action(valid_actions)
        
        elif self.difficulty == Difficulty.HARD:
            # Always use minimax algorithm
            return self._get_minimax_action()
        
        return random.choice(valid_actions)
    
    def _get_strategic_action(self, valid_actions: list) -> int:
        """Get a strategic action (check for win/block)."""
        # First, check if AI can win
        for action in valid_actions:
            row, col = divmod(action, 3)
            self.board[row, col] = Player.O.value
            if self._check_winner() and self.winner == Player.O:
                self.board[row, col] = Player.EMPTY.value
                self.winner = None
                return action
            self.board[row, col] = Player.EMPTY.value
            self.winner = None
        
        # Second, check if AI needs to block player
        for action in valid_actions:
            row, col = divmod(action, 3)
            self.board[row, col] = Player.X.value
            if self._check_winner() and self.winner == Player.X:
                self.board[row, col] = Player.EMPTY.value
                self.winner = None
                return action
            self.board[row, col] = Player.EMPTY.value
            self.winner = None
        
        # Otherwise, prefer center and corners
        preferences = [4, 0, 2, 6, 8, 1, 3, 5, 7]  # center, corners, edges
        for action in preferences:
            if action in valid_actions:
                return action
        
        return random.choice(valid_actions)
    
    def _get_minimax_action(self) -> int:
        """Get the best action using minimax algorithm."""
        best_score = float('-inf')
        best_action = None
        
        for action in self._get_valid_actions():
            row, col = divmod(action, 3)
            self.board[row, col] = Player.O.value
            score = self._minimax(False, 0)
            self.board[row, col] = Player.EMPTY.value
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action if best_action is not None else random.choice(self._get_valid_actions())
    
    def _minimax(self, is_maximizing: bool, depth: int) -> float:
        """Minimax algorithm implementation."""
        # Check terminal states
        if self._check_winner():
            if self.winner == Player.O:
                self.winner = None
                return 1.0 - depth * 0.1  # Prefer faster wins
            elif self.winner == Player.X:
                self.winner = None
                return -1.0 + depth * 0.1  # Prefer slower losses
        
        if self._is_board_full():
            return 0.0
        
        if is_maximizing:
            best_score = float('-inf')
            for action in self._get_valid_actions():
                row, col = divmod(action, 3)
                self.board[row, col] = Player.O.value
                score = self._minimax(False, depth + 1)
                self.board[row, col] = Player.EMPTY.value
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for action in self._get_valid_actions():
                row, col = divmod(action, 3)
                self.board[row, col] = Player.X.value
                score = self._minimax(True, depth + 1)
                self.board[row, col] = Player.EMPTY.value
                best_score = min(score, best_score)
            return best_score
    
    def _render_frame(self) -> np.ndarray:
        """Render the current game state and return as RGB array."""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.board_size, self.board_size))
            pygame.display.set_caption("Tic Tac Toe")
        
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        
        # Create surface for rendering
        canvas = pygame.Surface((self.board_size, self.board_size))
        canvas.fill(self.colors['white'])
        
        # Draw grid lines
        for i in range(1, 3):
            # Vertical lines
            pygame.draw.line(
                canvas,
                self.colors['black'],
                (i * self.cell_size, 0),
                (i * self.cell_size, self.board_size),
                3
            )
            # Horizontal lines
            pygame.draw.line(
                canvas,
                self.colors['black'],
                (0, i * self.cell_size),
                (self.board_size, i * self.cell_size),
                3
            )
        
        # Draw X's and O's
        for row in range(3):
            for col in range(3):
                center_x = col * self.cell_size + self.cell_size // 2
                center_y = row * self.cell_size + self.cell_size // 2
                
                if self.board[row, col] == Player.X.value:
                    # Draw X
                    offset = self.cell_size // 4
                    pygame.draw.line(
                        canvas,
                        self.colors['blue'],
                        (center_x - offset, center_y - offset),
                        (center_x + offset, center_y + offset),
                        5
                    )
                    pygame.draw.line(
                        canvas,
                        self.colors['blue'],
                        (center_x + offset, center_y - offset),
                        (center_x - offset, center_y + offset),
                        5
                    )
                elif self.board[row, col] == Player.O.value:
                    # Draw O
                    pygame.draw.circle(
                        canvas,
                        self.colors['red'],
                        (center_x, center_y),
                        self.cell_size // 4,
                        5
                    )
        
        # Draw game status
        if self.game_over:
            # Ensure pygame font module initialized before creating Font
            try:
                if not pygame.font.get_init():
                    pygame.font.init()
            except Exception:
                # In some headless or unusual environments pygame.font may not be available;
                # continue without text to avoid crashing the renderer.
                pygame_font_available = False
            else:
                pygame_font_available = True

            if pygame_font_available:
                try:
                    font = pygame.font.Font(None, 36)
                except Exception:
                    # Fallback to a default system font
                    try:
                        font = pygame.font.SysFont(None, 36)
                    except Exception:
                        font = None

                if font is not None:
                    if self.winner:
                        if self.winner == Player.X:
                            text = font.render("Player X Wins!", True, self.colors['green'])
                        else:
                            text = font.render("Player O Wins!", True, self.colors['green'])
                    else:
                        text = font.render("Draw!", True, self.colors['gray'])

                    text_rect = text.get_rect(center=(self.board_size // 2, 20))
                    canvas.blit(text, text_rect)
        
        if self.render_mode == "human":
            # Copy the surface to the display
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        
        # Convert surface to numpy array
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
        )


# Example usage and testing functions
def test_basic_functionality():
    """Test basic environment functionality."""
    print("Testing basic TicTacToe environment...")
    
    # Test environment creation
    env = TicTacToeEnv(render_mode="rgb_array", difficulty=Difficulty.EASY)
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial board:\n{info['board']}")
    
    # Test a few steps
    for i in range(3):
        valid_actions = info['valid_actions']
        if valid_actions:
            action = random.choice(valid_actions)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action {action}, Reward {reward}, Done {terminated}")
            print(f"Board:\n{info['board']}\n")
            
            if terminated:
                break
    
    env.close()
    print("Basic functionality test completed!\n")


def test_human_interaction():
    """Test human interaction mode."""
    print("Testing human interaction (close window to end)...")
    
    env = TicTacToeEnv(
        render_mode="human",
        difficulty=Difficulty.MEDIUM,
        human_player=True
    )
    
    obs, info = env.reset()
    
    print("Click on the board to make moves. Close the window to quit.")
    
    running = True
    while running and not info['game_over']:
        obs, reward, terminated, truncated, info = env.step(None)
        
        if terminated:
            print("Game over!")
            if info['winner']:
                winner = "X" if info['winner'] == 1 else "O"
                print(f"Winner: Player {winner}")
            else:
                print("It's a draw!")
            break
        
        # Check if window is still open
        try:
            pygame.event.peek()
        except:
            running = False
    
    env.close()
    print("Human interaction test completed!\n")


if __name__ == "__main__":
    # Run tests
    test_basic_functionality()
    
    # Uncomment to test human interaction
    # test_human_interaction()
