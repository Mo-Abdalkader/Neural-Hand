"""
Advanced Gesture Control System - Action Controller Module
Executes system actions based on detected gestures.

This module provides safe system integration for mouse control, keyboard input,
window management, and other system-level operations.
"""

import pyautogui
import logging
import time
from typing import Optional, Tuple, Dict
from enum import Enum


logger = logging.getLogger(__name__)


# Configure PyAutoGUI
pyautogui.FAILSAFE = False  # Disable corner fail-safe for production
pyautogui.PAUSE = 0.01  # Minimal pause between actions


class ActionType(Enum):
    """Types of system actions that can be executed."""
    MOVE_MOUSE = "move_mouse"
    LEFT_CLICK = "left_click"
    RIGHT_CLICK = "right_click"
    SCROLL = "scroll"
    DRAG_START = "drag_start"
    DRAG_END = "drag_end"
    MINIMIZE_WINDOW = "minimize_window"
    MAXIMIZE_WINDOW = "maximize_window"
    CLOSE_WINDOW = "close_window"
    VOLUME_UP = "volume_up"
    VOLUME_DOWN = "volume_down"


class ActionController:
    """
    System action execution controller.

    Provides safe execution of system-level actions with proper error handling,
    cooldowns, and safety mechanisms.
    """

    def __init__(self):
        """Initialize action controller with safety settings."""
        # Screen dimensions
        self.screen_width, self.screen_height = pyautogui.size()
        logger.info(f"Screen size detected: {self.screen_width}x{self.screen_height}")

        # Action cooldowns (seconds)
        self.action_cooldowns = {
            ActionType.LEFT_CLICK: 0.1,
            ActionType.RIGHT_CLICK: 0.1,
            ActionType.SCROLL: 0.05,
            ActionType.MINIMIZE_WINDOW: 1.0,
            ActionType.MAXIMIZE_WINDOW: 1.0,
            ActionType.CLOSE_WINDOW: 1.0,
            ActionType.VOLUME_UP: 0.1,
            ActionType.VOLUME_DOWN: 0.1,
        }

        # Last action times
        self.last_action_time: Dict[ActionType, float] = {}

        # Drag state
        self.is_dragging = False
        self.drag_start_position: Optional[Tuple[int, int]] = None

        # Emergency stop flag
        self.emergency_stop = False

        # Mouse movement smoothing
        self.last_mouse_position: Optional[Tuple[int, int]] = None
        self.mouse_smoothing = 0.3  # Lower = more responsive, higher = smoother

        logger.info("ActionController initialized")

    def can_execute_action(self, action_type: ActionType) -> bool:
        """
        Check if action can be executed based on cooldown.

        Args:
            action_type: Type of action to check

        Returns:
            bool: True if action can be executed
        """
        if self.emergency_stop:
            return False

        current_time = time.time()
        last_time = self.last_action_time.get(action_type, 0)
        cooldown = self.action_cooldowns.get(action_type, 0)

        return (current_time - last_time) > cooldown

    def move_mouse(self, normalized_x: float, normalized_y: float) -> bool:
        """
        Move mouse cursor to normalized screen position.

        Args:
            normalized_x: X coordinate (0-1)
            normalized_y: Y coordinate (0-1)

        Returns:
            bool: True if successful
        """
        if self.emergency_stop:
            return False

        try:
            # Convert normalized coordinates to screen pixels
            target_x = int(normalized_x * self.screen_width)
            target_y = int(normalized_y * self.screen_height)

            # Validate coordinates are within screen bounds
            target_x = max(0, min(target_x, self.screen_width - 1))
            target_y = max(0, min(target_y, self.screen_height - 1))

            # Apply smoothing if we have a previous position
            if self.last_mouse_position is not None:
                prev_x, prev_y = self.last_mouse_position
                target_x = int(prev_x + (target_x - prev_x) * (1 - self.mouse_smoothing))
                target_y = int(prev_y + (target_y - prev_y) * (1 - self.mouse_smoothing))

            # Move mouse
            pyautogui.moveTo(target_x, target_y, duration=0)

            # Update last position
            self.last_mouse_position = (target_x, target_y)

            return True

        except Exception as e:
            logger.error(f"Error moving mouse: {e}")
            return False

    def left_click(self) -> bool:
        """
        Perform left mouse click.

        Returns:
            bool: True if successful
        """
        if not self.can_execute_action(ActionType.LEFT_CLICK):
            return False

        try:
            pyautogui.click()
            self.last_action_time[ActionType.LEFT_CLICK] = time.time()
            logger.debug("Left click executed")
            return True

        except Exception as e:
            logger.error(f"Error performing left click: {e}")
            return False

    def right_click(self) -> bool:
        """
        Perform right mouse click.

        Returns:
            bool: True if successful
        """
        if not self.can_execute_action(ActionType.RIGHT_CLICK):
            return False

        try:
            pyautogui.rightClick()
            self.last_action_time[ActionType.RIGHT_CLICK] = time.time()
            logger.debug("Right click executed")
            return True

        except Exception as e:
            logger.error(f"Error performing right click: {e}")
            return False

    def scroll(self, amount: int) -> bool:
        """
        Scroll mouse wheel.

        Args:
            amount: Scroll amount (positive = up, negative = down)

        Returns:
            bool: True if successful
        """
        if not self.can_execute_action(ActionType.SCROLL):
            return False

        try:
            pyautogui.scroll(amount)
            self.last_action_time[ActionType.SCROLL] = time.time()
            logger.debug(f"Scroll executed: {amount}")
            return True

        except Exception as e:
            logger.error(f"Error scrolling: {e}")
            return False

    def start_drag(self) -> bool:
        """
        Start drag operation.

        Returns:
            bool: True if successful
        """
        if self.is_dragging:
            return False

        try:
            current_position = pyautogui.position()
            pyautogui.mouseDown()

            self.is_dragging = True
            self.drag_start_position = current_position

            logger.debug(f"Drag started at {current_position}")
            return True

        except Exception as e:
            logger.error(f"Error starting drag: {e}")
            return False

    def end_drag(self) -> bool:
        """
        End drag operation.

        Returns:
            bool: True if successful
        """
        if not self.is_dragging:
            return False

        try:
            pyautogui.mouseUp()

            end_position = pyautogui.position()
            logger.debug(f"Drag ended at {end_position}")

            self.is_dragging = False
            self.drag_start_position = None

            return True

        except Exception as e:
            logger.error(f"Error ending drag: {e}")
            return False

    def minimize_window(self) -> bool:
        """
        Minimize active window.

        Returns:
            bool: True if successful
        """
        if not self.can_execute_action(ActionType.MINIMIZE_WINDOW):
            return False

        try:
            # Windows: Alt+Space, then N for minimize
            pyautogui.hotkey('alt', 'space')
            time.sleep(0.1)
            pyautogui.press('n')

            self.last_action_time[ActionType.MINIMIZE_WINDOW] = time.time()
            logger.info("Window minimized")
            return True

        except Exception as e:
            logger.error(f"Error minimizing window: {e}")
            return False

    def maximize_window(self) -> bool:
        """
        Maximize active window.

        Returns:
            bool: True if successful
        """
        if not self.can_execute_action(ActionType.MAXIMIZE_WINDOW):
            return False

        try:
            # Windows: Alt+Space, then X for maximize
            pyautogui.hotkey('alt', 'space')
            time.sleep(0.1)
            pyautogui.press('x')

            self.last_action_time[ActionType.MAXIMIZE_WINDOW] = time.time()
            logger.info("Window maximized")
            return True

        except Exception as e:
            logger.error(f"Error maximizing window: {e}")
            return False

    def close_window(self) -> bool:
        """
        Close active window.

        Returns:
            bool: True if successful
        """
        if not self.can_execute_action(ActionType.CLOSE_WINDOW):
            return False

        try:
            # Windows: Alt+F4
            pyautogui.hotkey('alt', 'f4')

            self.last_action_time[ActionType.CLOSE_WINDOW] = time.time()
            logger.info("Window close command sent")
            return True

        except Exception as e:
            logger.error(f"Error closing window: {e}")
            return False

    def adjust_volume(self, direction: str) -> bool:
        """
        Adjust system volume.

        Args:
            direction: 'up' or 'down'

        Returns:
            bool: True if successful
        """
        action_type = ActionType.VOLUME_UP if direction == 'up' else ActionType.VOLUME_DOWN

        if not self.can_execute_action(action_type):
            return False

        try:
            if direction == 'up':
                pyautogui.press('volumeup')
            else:
                pyautogui.press('volumedown')

            self.last_action_time[action_type] = time.time()
            logger.debug(f"Volume {direction}")
            return True

        except Exception as e:
            logger.error(f"Error adjusting volume: {e}")
            return False

    def set_mouse_smoothing(self, smoothing: float):
        """
        Set mouse movement smoothing factor.

        Args:
            smoothing: Smoothing factor (0-1, lower = more responsive)
        """
        self.mouse_smoothing = max(0.0, min(1.0, smoothing))
        logger.info(f"Mouse smoothing set to {self.mouse_smoothing}")

    def enable_emergency_stop(self):
        """Enable emergency stop to prevent all actions."""
        self.emergency_stop = True
        logger.warning("Emergency stop ENABLED")

    def disable_emergency_stop(self):
        """Disable emergency stop to allow actions."""
        self.emergency_stop = False
        logger.info("Emergency stop DISABLED")

    def reset(self):
        """Reset controller state."""
        if self.is_dragging:
            self.end_drag()

        self.last_action_time.clear()
        self.last_mouse_position = None
        self.emergency_stop = False

        logger.info("ActionController reset")


if __name__ == "__main__":
    """Test the action controller module."""
    print("Action Controller Test")
    print("Testing basic functionality...")

    controller = ActionController()

    print(f"Screen size: {controller.screen_width}x{controller.screen_height}")
    print(f"Emergency stop: {controller.emergency_stop}")

    # Test mouse movement
    print("\nTesting mouse movement to center of screen...")
    success = controller.move_mouse(0.5, 0.5)
    print(f"Move mouse: {'✓' if success else '✗'}")

    time.sleep(1)

    # Test click cooldown
    print("\nTesting click cooldown...")
    click1 = controller.left_click()
    click2 = controller.left_click()  # Should fail due to cooldown
    print(f"First click: {'✓' if click1 else '✗'}")
    print(f"Second click (immediate): {'✗' if not click2 else '✓ (unexpected)'}")

    time.sleep(0.2)
    click3 = controller.left_click()  # Should succeed after cooldown
    print(f"Third click (after cooldown): {'✓' if click3 else '✗'}")

    print("\nAction Controller test completed")
    print("For full testing, run the complete application with gesture recognition")