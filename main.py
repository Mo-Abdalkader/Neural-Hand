"""
Advanced Gesture Control System - Ultimate Enhanced Version
Professional-grade gesture control with advanced features and stunning UI.
"""

import customtkinter as ctk
import cv2
import logging
import threading
import queue
import time
import numpy as np
from PIL import Image, ImageTk
from typing import Optional
import json
import os

from hand_tracker import HandTracker
from gesture_recognizer import GestureRecognizer, GestureType
from action_controller import ActionController

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gesture_control.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FloatingPreviewWindow(ctk.CTkToplevel):
    """Floating camera preview window that stays on top."""

    def __init__(self, parent):
        super().__init__(parent)

        self.title("Camera Preview - Floating")
        self.geometry("640x480+100+100")
        self.attributes('-topmost', True)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # Make window semi-transparent
        self.attributes('-alpha', 0.95)

        # Preview label
        self.preview_label = ctk.CTkLabel(self, text="")
        self.preview_label.pack(expand=True, fill="both", padx=5, pady=5)

        # Control bar at bottom
        control_bar = ctk.CTkFrame(self, height=40)
        control_bar.pack(fill="x", padx=5, pady=5)

        # Opacity slider
        ctk.CTkLabel(control_bar, text="Opacity:").pack(side="left", padx=5)
        self.opacity_slider = ctk.CTkSlider(
            control_bar, from_=0.3, to=1.0, width=100,
            command=self.change_opacity
        )
        self.opacity_slider.set(0.95)
        self.opacity_slider.pack(side="left", padx=5)

        # Pin button
        self.pin_button = ctk.CTkButton(
            control_bar, text="📌", width=40,
            command=self.toggle_pin
        )
        self.pin_button.pack(side="right", padx=5)

        self.is_pinned = True

    def change_opacity(self, value):
        """Change window opacity."""
        self.attributes('-alpha', value)

    def toggle_pin(self):
        """Toggle always-on-top."""
        self.is_pinned = not self.is_pinned
        self.attributes('-topmost', self.is_pinned)
        self.pin_button.configure(text="📌" if self.is_pinned else "📍")

    def on_close(self):
        """Handle close button."""
        self.withdraw()


class GestureControlApp(ctk.CTk):
    """Ultimate gesture control application with advanced features."""

    def __init__(self):
        super().__init__()

        # Window configuration
        self.title("Advanced Gesture Control System Pro")
        self.geometry("1400x900")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        # State management
        self.is_running = False
        self.control_enabled = False
        self.show_overlay = True
        self.show_landmarks = True
        self.show_control_zone = True
        self.mirror_mode = True

        # Components
        self.hand_tracker: Optional[HandTracker] = None
        self.gesture_recognizer = GestureRecognizer()
        self.action_controller = ActionController()
        self.floating_window: Optional[FloatingPreviewWindow] = None

        # Threading
        self.processing_thread: Optional[threading.Thread] = None
        self.frame_queue = queue.Queue(maxsize=2)

        # Metrics
        self.fps = 0.0
        self.gesture_count = 0
        self.session_start_time = None
        self.gesture_history = []

        # Control zone (virtual screen mapping)
        self.control_zone_margin = 0.15  # 15% margin

        # Settings
        self.settings_file = "gesture_control_settings.json"
        self.load_settings()

        # Build UI
        self.create_ui()

        logger.info("Application initialized")

    def load_settings(self):
        """Load saved settings."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    self.control_zone_margin = settings.get('control_zone_margin', 0.15)
                    self.mirror_mode = settings.get('mirror_mode', True)
        except Exception as e:
            logger.error(f"Error loading settings: {e}")

    def save_settings(self):
        """Save current settings."""
        try:
            settings = {
                'control_zone_margin': self.control_zone_margin,
                'mirror_mode': self.mirror_mode
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f)
        except Exception as e:
            logger.error(f"Error saving settings: {e}")

    def create_ui(self):
        """Create the enhanced user interface."""
        # Configure grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # Left sidebar
        self.create_left_sidebar()

        # Main content area
        self.create_main_content()

        # Right sidebar
        self.create_right_sidebar()

        # Bottom control panel
        self.create_bottom_panel()

    def create_left_sidebar(self):
        """Create left sidebar with controls."""
        sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=3, sticky="nsew", padx=0, pady=0)
        sidebar.grid_propagate(False)

        # Logo/Title
        title_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        title_frame.pack(fill="x", padx=20, pady=20)

        ctk.CTkLabel(
            title_frame,
            text="🖐️ Gesture\nControl Pro",
            font=ctk.CTkFont(size=24, weight="bold"),
            justify="center"
        ).pack()

        # Status Section
        status_section = ctk.CTkFrame(sidebar)
        status_section.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(
            status_section,
            text="System Status",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        self.status_label = ctk.CTkLabel(status_section, text="● Stopped", text_color="gray")
        self.status_label.pack(pady=2)

        self.control_status_label = ctk.CTkLabel(status_section, text="🎮 Control: OFF", text_color="gray")
        self.control_status_label.pack(pady=2)

        # Quick Actions
        actions_frame = ctk.CTkFrame(sidebar)
        actions_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(
            actions_frame,
            text="Quick Actions",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        self.start_button = ctk.CTkButton(
            actions_frame,
            text="▶ Start Tracking",
            command=self.start_tracking,
            height=35,
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.start_button.pack(pady=5, padx=10, fill="x")

        self.control_button = ctk.CTkButton(
            actions_frame,
            text="🎮 Enable Control",
            command=self.toggle_control,
            height=35,
            state="disabled",
            font=ctk.CTkFont(size=13, weight="bold")
        )
        self.control_button.pack(pady=5, padx=10, fill="x")

        self.floating_button = ctk.CTkButton(
            actions_frame,
            text="🪟 Floating Window",
            command=self.toggle_floating_window,
            height=35,
            state="disabled"
        )
        self.floating_button.pack(pady=5, padx=10, fill="x")

        self.stop_button = ctk.CTkButton(
            actions_frame,
            text="⏹ Stop",
            command=self.stop_tracking,
            height=35,
            state="disabled",
            fg_color="darkred",
            hover_color="red"
        )
        self.stop_button.pack(pady=5, padx=10, fill="x")

        # Display Options
        display_frame = ctk.CTkFrame(sidebar)
        display_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(
            display_frame,
            text="Display Options",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        self.overlay_switch = ctk.CTkSwitch(
            display_frame,
            text="Show Gesture Info",
            command=self.toggle_overlay
        )
        self.overlay_switch.select()
        self.overlay_switch.pack(pady=3, padx=10, anchor="w")

        self.landmarks_switch = ctk.CTkSwitch(
            display_frame,
            text="Show Hand Landmarks",
            command=self.toggle_landmarks
        )
        self.landmarks_switch.select()
        self.landmarks_switch.pack(pady=3, padx=10, anchor="w")

        self.zone_switch = ctk.CTkSwitch(
            display_frame,
            text="Show Control Zone",
            command=self.toggle_control_zone
        )
        self.zone_switch.select()
        self.zone_switch.pack(pady=3, padx=10, anchor="w")

        self.mirror_switch = ctk.CTkSwitch(
            display_frame,
            text="Mirror Mode",
            command=self.toggle_mirror
        )
        if self.mirror_mode:
            self.mirror_switch.select()
        self.mirror_switch.pack(pady=3, padx=10, anchor="w")

        # Advanced Settings
        settings_frame = ctk.CTkFrame(sidebar)
        settings_frame.pack(fill="both", expand=True, padx=15, pady=10)

        ctk.CTkLabel(
            settings_frame,
            text="Advanced Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        ).pack(pady=5)

        # Mouse smoothing
        ctk.CTkLabel(settings_frame, text="Mouse Smoothing", font=ctk.CTkFont(size=11)).pack(pady=(10,2))
        self.smoothing_slider = ctk.CTkSlider(
            settings_frame, from_=0, to=0.8, number_of_steps=8,
            command=self.on_smoothing_change
        )
        self.smoothing_slider.set(0.3)
        self.smoothing_slider.pack(pady=2, padx=10, fill="x")
        self.smoothing_label = ctk.CTkLabel(settings_frame, text="30%", font=ctk.CTkFont(size=10))
        self.smoothing_label.pack(pady=2)

        # Control zone margin
        ctk.CTkLabel(settings_frame, text="Control Zone Size", font=ctk.CTkFont(size=11)).pack(pady=(10,2))
        self.zone_slider = ctk.CTkSlider(
            settings_frame, from_=0, to=0.3, number_of_steps=30,
            command=self.on_zone_change
        )
        self.zone_slider.set(self.control_zone_margin)
        self.zone_slider.pack(pady=2, padx=10, fill="x")
        self.zone_label = ctk.CTkLabel(settings_frame, text=f"{int((1-self.control_zone_margin*2)*100)}%",
                                       font=ctk.CTkFont(size=10))
        self.zone_label.pack(pady=2)

        # Gesture sensitivity
        ctk.CTkLabel(settings_frame, text="Gesture Sensitivity", font=ctk.CTkFont(size=11)).pack(pady=(10,2))
        self.sensitivity_slider = ctk.CTkSlider(
            settings_frame, from_=0.3, to=0.8, number_of_steps=10,
            command=self.on_sensitivity_change
        )
        self.sensitivity_slider.set(0.5)
        self.sensitivity_slider.pack(pady=2, padx=10, fill="x")
        self.sensitivity_label = ctk.CTkLabel(settings_frame, text="Medium", font=ctk.CTkFont(size=10))
        self.sensitivity_label.pack(pady=2)

    def create_main_content(self):
        """Create main content area with camera preview."""
        main_frame = ctk.CTkFrame(self, corner_radius=0)
        main_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=0, pady=0)
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(main_frame, height=60, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=10)
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header,
            text="Camera Preview",
            font=ctk.CTkFont(size=20, weight="bold")
        ).grid(row=0, column=0, sticky="w")

        # Session info
        self.session_label = ctk.CTkLabel(
            header,
            text="Session: Not Started",
            font=ctk.CTkFont(size=12)
        )
        self.session_label.grid(row=0, column=1, sticky="e")

        # Preview area
        self.preview_frame = ctk.CTkFrame(main_frame)
        self.preview_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=(0,20))

        self.preview_label = ctk.CTkLabel(
            self.preview_frame,
            text="📷 Camera Preview\n\nClick 'Start Tracking' to begin\n\nThe blue rectangle shows your control zone\nMove your hand within this area for best results",
            font=ctk.CTkFont(size=16),
            justify="center"
        )
        self.preview_label.pack(expand=True, fill="both", padx=10, pady=10)

    def create_right_sidebar(self):
        """Create right sidebar with statistics and gesture guide."""
        sidebar = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        sidebar.grid(row=0, column=2, rowspan=3, sticky="nsew", padx=0, pady=0)
        sidebar.grid_propagate() # False

        # Performance Metrics
        metrics_frame = ctk.CTkFrame(sidebar)
        metrics_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(
            metrics_frame,
            text="📊 Performance",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)

        metrics_grid = ctk.CTkFrame(metrics_frame, fg_color="transparent")
        metrics_grid.pack(fill="x", padx=10, pady=5)

        self.fps_label = ctk.CTkLabel(metrics_grid, text="FPS: 0", font=ctk.CTkFont(size=13))
        self.fps_label.pack(pady=2, anchor="w")

        self.gesture_label = ctk.CTkLabel(metrics_grid, text="Gesture: None", font=ctk.CTkFont(size=13))
        self.gesture_label.pack(pady=2, anchor="w")

        self.confidence_label = ctk.CTkLabel(metrics_grid, text="Confidence: 0%", font=ctk.CTkFont(size=13))
        self.confidence_label.pack(pady=2, anchor="w")

        self.action_count_label = ctk.CTkLabel(metrics_grid, text="Actions: 0", font=ctk.CTkFont(size=13))
        self.action_count_label.pack(pady=2, anchor="w")

        # Recent Gestures
        recent_frame = ctk.CTkFrame(sidebar)
        recent_frame.pack(fill="x", padx=15, pady=10)

        ctk.CTkLabel(
            recent_frame,
            text="🕐 Recent Actions",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)

        self.recent_text = ctk.CTkTextbox(recent_frame, height=120, font=ctk.CTkFont(size=11))
        self.recent_text.pack(pady=5, padx=10, fill="x")
        self.recent_text.configure(state="disabled")

        # Gesture Guide
        guide_frame = ctk.CTkFrame(sidebar)
        guide_frame.pack(fill="both", expand=True, padx=15, pady=10)

        ctk.CTkLabel(
            guide_frame,
            text="📖 Gesture Guide",
            font=ctk.CTkFont(size=16, weight="bold")
        ).pack(pady=5)

        gestures = [
            ("👆 Index Extended", "Move Cursor", "Point with index finger"),
            ("🤏 Thumb-Index Pinch", "Left Click", "Pinch thumb and index"),
            ("🤌 Thumb-Middle Pinch", "Right Click", "Pinch thumb and middle"),
            ("✌️ Two Fingers", "Scroll", "Move hand up/down"),
            ("✊ Closed Fist", "Minimize", "Close all fingers"),
            ("🖐️ Open Palm", "Maximize", "Extend all fingers"),
            ("🤙 Thumb-Pinky", "Volume", "Move hand up/down"),
        ]

        for emoji, action, desc in gestures:
            gesture_item = ctk.CTkFrame(guide_frame)
            gesture_item.pack(fill="x", padx=10, pady=5)

            ctk.CTkLabel(
                gesture_item,
                text=emoji,
                font=ctk.CTkFont(size=20)
            ).pack(side="left", padx=5)

            info_frame = ctk.CTkFrame(gesture_item, fg_color="transparent")
            info_frame.pack(side="left", fill="x", expand=True, padx=5)

            ctk.CTkLabel(
                info_frame,
                text=action,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w"
            ).pack(anchor="w")

            ctk.CTkLabel(
                info_frame,
                text=desc,
                font=ctk.CTkFont(size=10),
                text_color="gray",
                anchor="w"
            ).pack(anchor="w")

    def create_bottom_panel(self):
        """Create bottom status panel."""
        bottom_panel = ctk.CTkFrame(self, height=40, corner_radius=0)
        bottom_panel.grid(row=2, column=1, sticky="ew", padx=0, pady=0)

        ctk.CTkLabel(
            bottom_panel,
            text="Advanced Gesture Control System Pro v2.0 | Ready",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        ).pack(side="left", padx=20, pady=10)

        self.tips_label = ctk.CTkLabel(
            bottom_panel,
            text="💡 Tip: Start tracking and enable control to begin",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        self.tips_label.pack(side="right", padx=20, pady=10)

    # Event Handlers
    def on_smoothing_change(self, value):
        """Handle smoothing slider change."""
        self.smoothing_label.configure(text=f"{int(value*100)}%")
        if self.action_controller:
            self.action_controller.set_mouse_smoothing(value)

    def on_zone_change(self, value):
        """Handle control zone slider change."""
        self.control_zone_margin = value
        size_percent = int((1 - value * 2) * 100)
        self.zone_label.configure(text=f"{size_percent}%")
        self.save_settings()

    def on_sensitivity_change(self, value):
        """Handle sensitivity slider change."""
        if value < 0.45:
            label = "Low"
        elif value < 0.65:
            label = "Medium"
        else:
            label = "High"
        self.sensitivity_label.configure(text=label)

    def toggle_overlay(self):
        """Toggle gesture info overlay."""
        self.show_overlay = self.overlay_switch.get()

    def toggle_landmarks(self):
        """Toggle hand landmarks display."""
        self.show_landmarks = self.landmarks_switch.get()

    def toggle_control_zone(self):
        """Toggle control zone display."""
        self.show_control_zone = self.zone_switch.get()

    def toggle_mirror(self):
        """Toggle mirror mode."""
        self.mirror_mode = self.mirror_switch.get()
        self.save_settings()

    def toggle_floating_window(self):
        """Toggle floating preview window."""
        if self.floating_window is None or not self.floating_window.winfo_exists():
            self.floating_window = FloatingPreviewWindow(self)
            self.floating_window.deiconify()
            self.floating_button.configure(text="🪟 Close Floating")
        else:
            self.floating_window.destroy()
            self.floating_window = None
            self.floating_button.configure(text="🪟 Floating Window")

    def start_tracking(self):
        """Start hand tracking."""
        if self.is_running:
            return

        try:
            self.hand_tracker = HandTracker()
            if not self.hand_tracker.start():
                logger.error("Failed to start hand tracker")
                return

            self.is_running = True
            self.gesture_count = 0
            self.gesture_history = []
            self.session_start_time = time.time()

            # Update UI
            self.start_button.configure(state="disabled")
            self.control_button.configure(state="normal")
            self.floating_button.configure(state="normal")
            self.stop_button.configure(state="normal")
            self.status_label.configure(text="● Running", text_color="green")

            # Start processing
            self.processing_thread = threading.Thread(target=self.processing_loop, daemon=True)
            self.processing_thread.start()

            self.update_ui()
            self.update_session_time()

            logger.info("Tracking started")

        except Exception as e:
            logger.error(f"Error starting: {e}")
            self.stop_tracking()

    def stop_tracking(self):
        """Stop hand tracking."""
        if not self.is_running:
            return

        self.is_running = False
        self.control_enabled = False

        if self.hand_tracker:
            self.hand_tracker.stop()
            self.hand_tracker = None

        # Update UI
        self.start_button.configure(state="normal")
        self.control_button.configure(state="disabled", text="🎮 Enable Control")
        self.floating_button.configure(state="disabled")
        self.stop_button.configure(state="disabled")
        self.status_label.configure(text="● Stopped", text_color="gray")
        self.control_status_label.configure(text="🎮 Control: OFF", text_color="gray")

        self.preview_label.configure(
            image=None,
            text="📷 Camera Preview\n\nClick 'Start Tracking' to begin\n\nThe blue rectangle shows your control zone\nMove your hand within this area for best results"
        )

        logger.info("Tracking stopped")

    def toggle_control(self):
        """Toggle gesture control."""
        self.control_enabled = not self.control_enabled

        if self.control_enabled:
            self.control_button.configure(text="🎮 Disable Control")
            self.control_status_label.configure(text="🎮 Control: ON", text_color="green")
            self.tips_label.configure(text="💡 Control enabled - gestures will trigger actions")
        else:
            self.control_button.configure(text="🎮 Enable Control")
            self.control_status_label.configure(text="🎮 Control: OFF", text_color="gray")
            self.tips_label.configure(text="💡 Control disabled - safe testing mode")

    def draw_control_zone(self, frame):
        """Draw control zone rectangle on frame."""
        h, w = frame.shape[:2]
        margin_x = int(w * self.control_zone_margin)
        margin_y = int(h * self.control_zone_margin)

        # Draw semi-transparent overlay outside control zone
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, margin_y), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, h-margin_y), (w, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, 0), (margin_x, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (w-margin_x, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

        # Draw control zone border
        cv2.rectangle(frame, (margin_x, margin_y), (w-margin_x, h-margin_y), (0, 255, 255), 3)

        # Add corner markers
        corner_size = 20
        cv2.line(frame, (margin_x, margin_y), (margin_x+corner_size, margin_y), (0, 255, 0), 3)
        cv2.line(frame, (margin_x, margin_y), (margin_x, margin_y+corner_size), (0, 255, 0), 3)
        cv2.line(frame, (w-margin_x, margin_y), (w-margin_x-corner_size, margin_y), (0, 255, 0), 3)
        cv2.line(frame, (w-margin_x, margin_y), (w-margin_x, margin_y+corner_size), (0, 255, 0), 3)
        cv2.line(frame, (margin_x, h-margin_y), (margin_x+corner_size, h-margin_y), (0, 255, 0), 3)
        cv2.line(frame, (margin_x, h-margin_y), (margin_x, h-margin_y-corner_size), (0, 255, 0), 3)
        cv2.line(frame, (w-margin_x, h-margin_y), (w-margin_x-corner_size, h-margin_y), (0, 255, 0), 3)
        cv2.line(frame, (w-margin_x, h-margin_y), (w-margin_x, h-margin_y-corner_size), (0, 255, 0), 3)

        # Add label
        cv2.putText(frame, "CONTROL ZONE", (margin_x + 10, margin_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame

    def draw_cursor_indicator(self, frame, landmarks):
        """Draw cursor indicator."""
        if not landmarks:
            return frame

        h, w = frame.shape[:2]
        index_tip = landmarks[8]
        x, y = int(index_tip[0] * w), int(index_tip[1] * h)

        # Animated crosshair
        cv2.circle(frame, (x, y), 15, (0, 255, 255), 2)
        cv2.circle(frame, (x, y), 8, (255, 255, 0), -1)
        cv2.line(frame, (x-25, y), (x+25, y), (0, 255, 255), 2)
        cv2.line(frame, (x, y-25), (x, y+25), (0, 255, 255), 2)

        return frame

    def draw_gesture_info(self, frame, gesture_type, confidence):
        """Draw enhanced gesture information."""
        h, w = frame.shape[:2]

        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 50), (350, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Border
        cv2.rectangle(frame, (10, 50), (350, 180), (0, 255, 0), 2)

        # Gesture name
        gesture_name = gesture_type.value.replace('_', ' ').title()
        cv2.putText(frame, f"Gesture: {gesture_name}", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Confidence bar
        cv2.putText(frame, "Confidence:", (20, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        bar_width = int(280 * confidence)
        bar_color = (0, int(255 * confidence), int(255 * (1-confidence)))
        cv2.rectangle(frame, (20, 135), (20 + bar_width, 150), bar_color, -1)
        cv2.rectangle(frame, (20, 135), (300, 150), (255, 255, 255), 2)
        cv2.putText(frame, f"{int(confidence*100)}%", (310, 148),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Status indicator
        status_text = "ACTIVE" if self.control_enabled else "MONITORING"
        status_color = (0, 255, 0) if self.control_enabled else (255, 165, 0)
        cv2.putText(frame, status_text, (20, 170),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)

        return frame

    def map_to_screen_coordinates(self, hand_x, hand_y):
        """Map hand coordinates to screen with control zone."""
        # Normalize to control zone
        mapped_x = (hand_x - self.control_zone_margin) / (1 - 2 * self.control_zone_margin)
        mapped_y = (hand_y - self.control_zone_margin) / (1 - 2 * self.control_zone_margin)

        # Clamp to 0-1 range
        mapped_x = max(0, min(1, mapped_x))
        mapped_y = max(0, min(1, mapped_y))

        return mapped_x, mapped_y

    def add_recent_action(self, action_name):
        """Add action to recent actions list."""
        timestamp = time.strftime("%H:%M:%S")
        self.gesture_history.append(f"[{timestamp}] {action_name}")

        # Keep only last 10 actions
        if len(self.gesture_history) > 10:
            self.gesture_history.pop(0)

        # Update display
        self.recent_text.configure(state="normal")
        self.recent_text.delete("1.0", "end")
        self.recent_text.insert("1.0", "\n".join(self.gesture_history))
        self.recent_text.configure(state="disabled")

    def processing_loop(self):
        """Main processing loop with enhanced features."""
        logger.info("Processing loop started")
        last_index_pos = None
        scroll_start_y = None

        while self.is_running and self.hand_tracker:
            try:
                frame = self.hand_tracker.get_frame()
                if frame is None:
                    continue

                # Process frame
                processed_frame = self.hand_tracker.process_frame(frame)

                # Get hand info
                hand_info = self.hand_tracker.get_hand_info()

                # Draw control zone if enabled
                if self.show_control_zone:
                    processed_frame = self.draw_control_zone(processed_frame)

                if not hand_info:
                    self.fps = self.hand_tracker.get_average_fps()
                    try:
                        self.frame_queue.put_nowait({
                            'frame': processed_frame,
                            'fps': self.fps,
                            'gesture': GestureType.NONE,
                            'confidence': 0.0
                        })
                    except queue.Full:
                        pass
                    continue

                # Get landmarks
                first_hand_landmarks = hand_info[0][1]

                # Recognize gesture
                gesture_type, confidence = self.gesture_recognizer.recognize_gesture(first_hand_landmarks)

                # Draw overlays
                if self.show_overlay and gesture_type != GestureType.NONE:
                    processed_frame = self.draw_gesture_info(processed_frame, gesture_type, confidence)

                if gesture_type == GestureType.INDEX_EXTENDED:
                    processed_frame = self.draw_cursor_indicator(processed_frame, first_hand_landmarks)

                # Execute actions if control enabled
                if self.control_enabled and gesture_type != GestureType.NONE:
                    action_executed = self.execute_gesture_action(
                        gesture_type, first_hand_landmarks, last_index_pos, scroll_start_y
                    )

                    if action_executed:
                        last_index_pos = self.gesture_recognizer.get_cursor_position(first_hand_landmarks)
                        if gesture_type == GestureType.TWO_FINGER_SCROLL:
                            if scroll_start_y is None and last_index_pos:
                                scroll_start_y = last_index_pos[1]
                        else:
                            scroll_start_y = None
                else:
                    scroll_start_y = None

                self.fps = self.hand_tracker.get_average_fps()

                # Send to queue
                try:
                    self.frame_queue.put_nowait({
                        'frame': processed_frame,
                        'fps': self.fps,
                        'gesture': gesture_type,
                        'confidence': confidence
                    })
                except queue.Full:
                    pass

            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.01)

    def execute_gesture_action(self, gesture_type, landmarks, last_pos, scroll_start_y):
        """Execute system action based on gesture."""
        try:
            if gesture_type == GestureType.INDEX_EXTENDED:
                cursor_pos = self.gesture_recognizer.get_cursor_position(landmarks)
                if cursor_pos:
                    # Map to screen with control zone
                    mapped_x, mapped_y = self.map_to_screen_coordinates(cursor_pos[0], cursor_pos[1])
                    self.action_controller.move_mouse(mapped_x, mapped_y)
                return True

            elif gesture_type == GestureType.THUMB_INDEX_PINCH:
                if self.gesture_recognizer.can_activate_gesture(gesture_type):
                    self.action_controller.left_click()
                    self.gesture_recognizer.activate_gesture(gesture_type)
                    self.gesture_count += 1
                    self.add_recent_action("Left Click")
                    return True

            elif gesture_type == GestureType.THUMB_MIDDLE_PINCH:
                if self.gesture_recognizer.can_activate_gesture(gesture_type):
                    self.action_controller.right_click()
                    self.gesture_recognizer.activate_gesture(gesture_type)
                    self.gesture_count += 1
                    self.add_recent_action("Right Click")
                    return True

            elif gesture_type == GestureType.TWO_FINGER_SCROLL:
                cursor_pos = self.gesture_recognizer.get_cursor_position(landmarks)
                if cursor_pos and scroll_start_y is not None:
                    delta_y = cursor_pos[1] - scroll_start_y
                    if abs(delta_y) > 0.02:
                        scroll_amount = int(-delta_y * 150)
                        self.action_controller.scroll(scroll_amount)
                        return True

            elif gesture_type == GestureType.CLOSED_FIST:
                if self.gesture_recognizer.can_activate_gesture(gesture_type):
                    self.action_controller.minimize_window()
                    self.gesture_recognizer.activate_gesture(gesture_type)
                    self.gesture_count += 1
                    self.add_recent_action("Minimize Window")
                    return True

            elif gesture_type == GestureType.OPEN_PALM:
                if self.gesture_recognizer.can_activate_gesture(gesture_type):
                    self.action_controller.maximize_window()
                    self.gesture_recognizer.activate_gesture(gesture_type)
                    self.gesture_count += 1
                    self.add_recent_action("Maximize Window")
                    return True

            elif gesture_type == GestureType.THUMB_PINKY_EXTENDED:
                cursor_pos = self.gesture_recognizer.get_cursor_position(landmarks)
                if cursor_pos:
                    if cursor_pos[1] < 0.4:
                        self.action_controller.adjust_volume('up')
                        self.add_recent_action("Volume Up")
                    elif cursor_pos[1] > 0.6:
                        self.action_controller.adjust_volume('down')
                        self.add_recent_action("Volume Down")
                    return True

        except Exception as e:
            logger.error(f"Error executing action: {e}")

        return False

    def update_ui(self):
        """Update UI with latest data."""
        if not self.is_running:
            return

        try:
            frame_info = self.frame_queue.get_nowait()
            frame = frame_info['frame']
            fps = frame_info['fps']
            gesture = frame_info['gesture']
            confidence = frame_info['confidence']

            # Convert frame
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)

            # Update main preview
            preview_width = self.preview_frame.winfo_width()
            preview_height = self.preview_frame.winfo_height()

            if preview_width > 1 and preview_height > 1:
                img_tk = ctk.CTkImage(
                    light_image=img,
                    dark_image=img,
                    size=(preview_width, preview_height)
                )
                self.preview_label.configure(image=img_tk, text="")
                self.preview_label.image = img_tk

            # Update floating window if exists
            if self.floating_window and self.floating_window.winfo_exists():
                floating_img = ctk.CTkImage(
                    light_image=img,
                    dark_image=img,
                    size=(620, 460)
                )
                self.floating_window.preview_label.configure(image=floating_img)
                self.floating_window.preview_label.image = floating_img

            # Update metrics
            self.fps_label.configure(text=f"FPS: {fps:.1f}")

            gesture_name = gesture.value.replace('_', ' ').title() if gesture != GestureType.NONE else "None"
            self.gesture_label.configure(text=f"Gesture: {gesture_name}")

            # Color-coded confidence
            conf_color = "green" if confidence > 0.7 else "orange" if confidence > 0.5 else "red"
            self.confidence_label.configure(
                text=f"Confidence: {confidence*100:.0f}%",
                text_color=conf_color
            )

            self.action_count_label.configure(text=f"Actions: {self.gesture_count}")

        except queue.Empty:
            pass
        except Exception as e:
            logger.error(f"Error updating UI: {e}")

        self.after(30, self.update_ui)

    def update_session_time(self):
        """Update session duration."""
        if not self.is_running or self.session_start_time is None:
            return

        elapsed = int(time.time() - self.session_start_time)
        hours = elapsed // 3600
        minutes = (elapsed % 3600) // 60
        seconds = elapsed % 60

        time_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        self.session_label.configure(text=f"Session: {time_str}")

        self.after(1000, self.update_session_time)

    def on_closing(self):
        """Handle window closing."""
        logger.info("Application closing...")
        self.save_settings()
        self.stop_tracking()

        if self.floating_window and self.floating_window.winfo_exists():
            self.floating_window.destroy()

        self.destroy()


if __name__ == "__main__":
    logger.info("Starting Advanced Gesture Control System Pro")
    app = GestureControlApp()
    app.mainloop()