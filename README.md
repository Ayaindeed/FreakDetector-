# 🎭 FreakDetector - Enhanced AI Gesture Recognition System

> **Originally forked from [Elijah-cyber7/FreakDetector](https://github.com/Elijah-cyber7/FreakDetector)**  
> Enhanced with multiple gesture detection and web interface capabilities

FreakDetector is an intelligent real-time gesture recognition system that uses computer vision to detect facial expressions and hand gestures, responding with entertaining media content. Built with MediaPipe and OpenCV, it now supports multiple detection modes including a modern web interface.

## What's New in This Fork

This enhanced version includes several major improvements over the original:

- **Eyebrow Raise Detection** - Detects when you raise your eyebrows significantly
- **Thumbs Up Detection** - Recognizes clear thumbs up hand gestures  
- **Streamlit Web App** - Modern browser-based interface with real-time camera feed
- **Multi-Gesture System** - All gestures work simultaneously with individual media responses
- **Better User Experience** - Enhanced UI, status indicators, and error handling
- **Improved Detection** - More reliable gesture recognition algorithms

## Gesture Detection Modes

| Gesture | Trigger | Media Response |
|---------|---------|----------------|
| **Tongue + Head Shake** | Stick out tongue while shaking head | `orca.mp4` |
| **Eyebrow Raise** | Raise eyebrows significantly | `eyebrow_raise.mp4` |
| **Thumbs Up** | Clear thumbs up hand gesture | `thumbsup.gif` |

## Quick Start

### Option 1: Desktop Application
```bash
# Clone the repository
git clone https://github.com/Ayaindeed/FreakDetector-.git
cd FreakDetector-

# Install dependencies
pip install -r requirements.txt

# Run the desktop version
python main.py
```

### Option 2: Web Application (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt

# Launch web interface
streamlit run streamlit_app.py
```

The web app will open in your browser with an intuitive interface featuring live camera feed, real-time detection status, and media responses.

## Dependencies

```
opencv-python>=4.5.0
mediapipe>=0.9.0
streamlit>=1.25.0
numpy>=1.21.0
Pillow>=8.3.0
```

## Project Structure

```
FreakDetector/
├── main.py                 # Desktop application
├── streamlit_app.py         # Web application  
├── requirements.txt         # Dependencies
├── USAGE.md                # Detailed usage guide
├── Assets/                 # Media files
│   ├── orca.mp4           # Tongue + shake response
│   ├── eyebrow_raise.mp4  # Eyebrow raise response
│   └── thumbsup.gif       # Thumbs up response
└── README.md              # This file
```

## Usage Tips

- **Lighting**: Ensure good lighting for accurate detection
- **Camera Position**: Face the camera directly for best results
- **Gesture Duration**: Hold gestures for 2-3 seconds to trigger
- **Cooldown**: Wait between gestures to avoid rapid triggering

## Customization

### Adding New Gestures
1. Add your media file to the `Assets/` folder
2. Update the `VIDEO_PATHS` dictionary in the code
3. Implement your detection function following the existing patterns
4. Add the gesture to the main detection loop

### Adjusting Sensitivity
Modify these parameters in the settings section:
- `SHAKE_THRESHOLD`: Head shake sensitivity
- `TONGUE_THRESHOLD`: Tongue detection sensitivity  
- `EYEBROW_THRESHOLD`: Eyebrow raise sensitivity
- `SUSTAIN_FRAMES`: Frames needed to trigger response


## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-gesture`)
3. Commit your changes (`git commit -m 'Add amazing gesture detection'`)
4. Push to the branch (`git push origin feature/amazing-gesture`)
5. Open a Pull Request

## License

This project maintains the same license as the original repository. See the original project for licensing details.

## Credits

- **Original Creator**: [Andrew Allen (ElijahCyber)](https://github.com/Elijah-cyber7)
- **Enhanced by**: [Ayaindeed](https://github.com/Ayaindeed)
- **Technologies**: MediaPipe (Google), OpenCV, Streamlit

## Troubleshooting

**Camera Issues**: Check camera permissions and ensure no other apps are using the camera  
**Performance**: Reduce camera resolution or adjust detection thresholds  
**Media Playback**: Verify media files exist in the Assets folder  

For detailed troubleshooting, see [USAGE.md](USAGE.md).

---

**Ready to detect some gestures?**  Launch the app and start making faces!