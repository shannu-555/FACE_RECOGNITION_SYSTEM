# Face Recognition System

A simple and effective face recognition system built with Python and OpenCV. Import face images from files and recognize them in real-time.

## Quick Start

### Option 1: Easy Start (Recommended)
```bash
python start.py
```

### Option 2: Direct Start
```bash
python image_based_face_recognition.py
```

## Features

- **Face Detection**: Real-time face detection using OpenCV
- **Image Import**: Import face images from existing photo files
- **Batch Processing**: Import multiple images from directories
- **Face Recognition**: Recognize known faces in real-time
- **Persistent Storage**: Save face encodings for future use
- **Easy Setup**: No complex dependencies required

## Supported Image Formats

- **JPG/JPEG** - Most common format
- **PNG** - High quality format
- **BMP** - Windows bitmap format
- **TIFF** - High quality format

## Usage

1. **Start the system**: Run `python start.py`
2. **Import face images**: Select option 2, then choose single image or directory import
3. **Start recognition**: Select option 1 to begin real-time recognition

## File Structure

```
face-recognition-system/
├── start.py                           # Easy startup script
├── image_based_face_recognition.py    # Main application
├── simple_face_recognition.py         # Camera-based alternative
├── install_simple.py                  # Installation script
├── requirements.txt                   # Dependencies
├── README.md                          # This file
└── known_faces/                       # Saved face images (auto-created)
```

## Tips for Best Results

- Use high-quality, well-lit face images
- Face should be clearly visible and front-facing
- Avoid sunglasses, hats, or face coverings
- Use descriptive filenames for batch import
- Import multiple angles of the same person

## Troubleshooting

- **Camera not working**: Make sure no other app is using the camera
- **No faces detected**: Check image quality and lighting
- **Poor recognition**: Try importing better quality images
- **Installation issues**: Run `python install_simple.py`

## Requirements

- Python 3.7 or higher
- Webcam
- Windows 10/11 (tested on Windows)

## License

This project is open source and available under the MIT License. 