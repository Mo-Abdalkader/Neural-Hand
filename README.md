# 🖐️ Advanced Hand Tracking Control System

[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10.0-orange.svg)](https://mediapipe.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-red.svg)](https://opencv.org/)

Control your computer with hand gestures using AI-powered computer vision! This professional-grade application enables touchless interaction with your PC through intuitive hand movements.

[//]: # (![Demo]&#40;&#41;)

## ✨ Features

### 🎯 Core Functionality
- **Real-time Hand Tracking** - Powered by Google's MediaPipe with 21-point landmark detection
- **Gesture Recognition** - Intelligent classification of 10+ distinct hand gestures
- **Mouse Control** - Natural cursor movement with configurable smoothing
- **Click Actions** - Left/right click via intuitive pinch gestures
- **Scroll Control** - Two-finger scrolling with adjustable speed
- **Window Management** - Minimize, maximize, and close windows with gestures
- **Volume Control** - Adjust system volume with thumb-pinky gesture

### 🎨 Professional UI
- **Modern Dark Theme** - Built with CustomTkinter for a sleek appearance
- **Live Camera Preview** - Real-time visualization with overlay indicators
- **Floating Window Mode** - Always-on-top preview with adjustable opacity
- **Performance Metrics** - FPS counter, gesture confidence, and session statistics
- **Control Zone Visualization** - Visual guides for optimal hand positioning
- **Gesture Guide** - Built-in reference for all supported gestures

### ⚙️ Advanced Configuration
- **Mouse Smoothing** - Adjustable cursor responsiveness (0-80%)
- **Control Zone Sizing** - Customize active tracking area
- **Gesture Sensitivity** - Fine-tune detection thresholds
- **Mirror Mode** - Toggle camera flip for natural interaction
- **Cooldown System** - Prevent accidental repeated actions
- **Logging System** - Comprehensive debug and error logging

## 🎬 Quick Demo

### Supported Gestures

| Gesture | Action | Visual |
|---------|--------|--------|
| 👆 **Index Extended** | Move Cursor | Point with index finger |
| 🤏 **Thumb-Index Pinch** | Left Click | Pinch thumb and index together |
| 🤌 **Thumb-Middle Pinch** | Right Click | Pinch thumb and middle finger |
| ✌️ **Two Fingers** | Scroll | Extend index and middle, move up/down |
| ✊ **Closed Fist** | Minimize Window | Close all fingers into fist |
| 🖐️ **Open Palm** | Maximize Window | Extend all five fingers |
| 👎 **Thumb Down** | Close Window | Point thumb downward |
| 🤙 **Thumb-Pinky** | Volume Control | Extend thumb and pinky, move up/down |

## 🚀 Quick Start

### Prerequisites

```bash
# System Requirements
- Python 3.8 or higher
- Webcam (built-in or USB)
- Windows 10+ / macOS / Linux
- 4GB RAM (8GB recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Mo-Abdalkader/Neural-Hand.git
cd Neural-Hand
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Running the Application

```bash
python main.py
```

### First Time Setup

1. Click **"▶ Start Tracking"** to initialize the camera
2. Position your hand in the blue control zone
3. Click **"🎮 Enable Control"** to activate gesture recognition
4. Start performing gestures!

## 📖 Documentation

### Project Structure

```
hand-tracking-control/
│
├── main.py                        # Main application and UI
├── hand_tracker.py                # MediaPipe hand tracking module
├── gesture_recognizer.py          # Gesture classification engine
├── action_controller.py           # System action executor
├── requirements.txt               # Python dependencies
│
├── gesture_control.log            # Runtime logs (auto-generated)
└── gesture_control_settings.json  # User settings (auto-generated)
```

### Key Components

#### Hand Tracker (`hand_tracker.py`)
- Initializes MediaPipe Hands model
- Captures and processes video frames
- Extracts 21 hand landmarks in 3D space
- Provides FPS monitoring

#### Gesture Recognizer (`gesture_recognizer.py`)
- Analyzes landmark positions and distances
- Classifies gestures with confidence scores
- Implements temporal smoothing to reduce jitter
- Manages gesture activation cooldowns

#### Action Controller (`action_controller.py`)
- Executes system-level actions safely
- Provides mouse movement with smoothing
- Handles keyboard shortcuts for window management
- Implements safety mechanisms and rate limiting

#### Main Application (`main.py`)
- Professional GUI built with CustomTkinter
- Real-time video processing thread
- Frame queue for smooth UI updates
- Settings persistence and session management

## 🎛️ Configuration

### Adjusting Sensitivity

Edit `gesture_recognizer.py`:
```python
self.pinch_threshold = 0.05      # Pinch detection distance
self.hold_time = 0.3             # Gesture hold duration
self.cooldown_time = 0.5         # Action repeat delay
```

### Mouse Smoothing

Edit `action_controller.py`:
```python
self.mouse_smoothing = 0.3       # 0 = instant, 1 = very smooth
```

### Camera Settings

Edit `hand_tracker.py`:
```python
min_detection_confidence = 0.5   # Hand detection threshold
min_tracking_confidence = 0.5    # Tracking confidence
model_complexity = 1             # 0=Lite, 1=Full, 2=Heavy
```

## 🔧 Advanced Usage

### Custom Gestures

Extend the gesture recognition by adding new gesture types in `gesture_recognizer.py`:

```python
class GestureType(Enum):
    # Add your custom gesture
    CUSTOM_GESTURE = "custom_gesture"

def _detect_custom_gesture(self, landmarks: np.ndarray) -> float:
    # Implement your detection logic
    # Return confidence score (0-1)
    pass
```

### System Actions

Add custom actions in `action_controller.py`:

```python
def custom_action(self) -> bool:
    """Execute custom system action."""
    try:
        # Your action code here
        pyautogui.hotkey('win', 'd')  # Example: Show desktop
        return True
    except Exception as e:
        logger.error(f"Error: {e}")
        return False
```

## 📊 Performance

### Typical Performance Metrics
- **FPS**: 25-30 on modern hardware
- **Latency**: 30-50ms gesture-to-action
- **CPU Usage**: 15-25% (varies by camera resolution)
- **Memory**: 300-400MB
- **Accuracy**: 90-95% gesture recognition

### Optimization Tips
1. Lower camera resolution for better FPS
2. Ensure good lighting conditions
3. Close unnecessary background applications
4. Use model_complexity=0 for faster processing
5. Reduce mouse smoothing for lower latency

## 🐛 Troubleshooting

### Camera Not Found
```bash
# Check available cameras
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"

# Try different camera index in hand_tracker.py
camera_id = 1  # Instead of 0
```

### Low FPS
- Reduce camera resolution to 320x240
- Use model_complexity=0
- Ensure good lighting (reduces processing time)
- Close other camera-using applications

### Gestures Not Detected
- Ensure "Enable Control" is activated
- Improve lighting (avoid backlighting)
- Keep hand within the blue control zone
- Perform gestures more deliberately
- Check `gesture_control.log` for errors

### PyAutoGUI Not Working (Windows)
```bash
# Run as Administrator
# Right-click Python executable > Run as administrator
```

## 🛣️ Roadmap

### Version 2.0 (Coming Soon)
- [ ] **Standalone Executable** - Pre-built `.exe` for Windows (no Python required)
- [ ] **Multi-hand Support** - Track and control with both hands simultaneously
- [ ] **Custom Gesture Creator** - Visual interface to define new gestures
- [ ] **Gesture Profiles** - Save/load different gesture configurations
- [ ] **Keyboard Typing** - Virtual keyboard controlled by gestures
- [ ] **AI Gesture Learning** - Machine learning for personalized gesture recognition

### Version 3.0 (Future)
- [ ] **Cross-platform Desktop App** - Electron-based containerized application
- [ ] **Voice Command Integration** - Combine gestures with voice control
- [ ] **Accessibility Features** - Enhanced support for users with disabilities

### Enterprise Features (Planned)
- [ ] **Multi-monitor Support** - Seamless control across multiple screens
- [ ] **Presentation Mode** - Optimized gestures for presentations
- [ ] **Security Features** - Password protection and secure mode
- [ ] **Analytics Dashboard** - Usage statistics and gesture heatmaps

## 📦 Distribution Plans

### Windows Executable (Q4 2025)
I;m working on a standalone Windows `.exe` installer:
- ✅ One-click installation
- ✅ No Python or dependencies required
- ✅ Auto-update functionality
- ✅ System tray integration
- ✅ Professional installer wizard

### Docker Container (Q1 2026)
Containerized desktop application with full GUI:
- 🐳 Docker image with X11 forwarding
- 🐳 Hardware acceleration support
- 🐳 Easy deployment on any platform
- 🐳 Isolated environment

### Cross-platform Builds
- **Windows**: `.exe` installer + portable version
- **macOS**: `.dmg` installer + Homebrew formula
- **Linux**: `.AppImage`, `.deb`, `.rpm` packages

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/AmazingFeature`)
3. **Commit your changes** (`git commit -m 'Add some AmazingFeature'`)
4. **Push to the branch** (`git push origin feature/AmazingFeature`)
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to all functions
- Include unit tests for new features
- Update documentation as needed

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MediaPipe** - Google's ML framework for hand tracking
- **OpenCV** - Computer vision library
- **CustomTkinter** - Modern UI framework
- **PyAutoGUI** - Cross-platform GUI automation

## 📧 Contact

**Project Maintainer**: Your Name
- GitHub: [@Mo-Abdalkader](https://github.com/Mo-Abdalkader)
- Email: Mohameed.Abdalkadeer@gmail.com
- LinkedIn: [Mohamed Abdalkader](https://www.linkedin.com/in/mo-abdalkader/)

## ⭐ Star History

If you find this project useful, please consider giving it a star! It helps others discover the project.

[![Star History Chart](https://api.star-history.com/svg?repos=Mo-Abdalkader/Neural-Hand&type=Date)](https://star-history.com/#Mo-Abdalkader/Neural-Hand&Date)

## 💖 Support

If you like this project, you can support it by:
- ⭐ Starring the repository
- 🐛 Reporting bugs and issues
- 💡 Suggesting new features
- 📖 Improving documentation
- 🔀 Contributing code

---

**Built with ❤️ using Python, MediaPipe, and OpenCV**

*Control your computer with the wave of a hand!* 🖐️✨