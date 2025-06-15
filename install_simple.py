#!/usr/bin/env python3
"""
Simple installation script for Face Recognition System (OpenCV only version)
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages for simple face recognition system...")
    
    # Install basic requirements
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])
        print("✓ opencv-python installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install opencv-python")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy"])
        print("✓ numpy installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install numpy")
        return False
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        print("✓ Pillow installed successfully")
    except subprocess.CalledProcessError:
        print("✗ Failed to install Pillow")
        return False
    
    print("✓ All requirements installed successfully")
    return True

def check_installation():
    """Check if all required packages are installed"""
    required_packages = [
        'cv2',
        'numpy',
        'PIL'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
                print(f"✓ {package} is installed (version: {cv2.__version__})")
            elif package == 'numpy':
                import numpy
                print(f"✓ {package} is installed (version: {numpy.__version__})")
            elif package == 'PIL':
                from PIL import Image
                print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is missing")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def main():
    print("=== Simple Face Recognition System Setup ===")
    print("(OpenCV only version - no dlib required)")
    print()
    
    # Check if requirements are already installed
    if check_installation():
        print("\nAll required packages are already installed!")
        print("You can now run: python simple_face_recognition.py")
        return
    
    print("\nSome packages are missing. Installing...")
    
    # Install requirements
    if install_requirements():
        print("\n✓ Setup completed successfully!")
        print("You can now run: python simple_face_recognition.py")
        
        # Final check
        print("\nFinal verification:")
        if check_installation():
            print("\n🎉 Everything is ready! You can start using the face recognition system.")
        else:
            print("\n⚠️  Some packages may not be properly installed. Please try manual installation.")
    else:
        print("\n✗ Setup failed!")
        print("\nManual installation instructions:")
        print("1. Run: pip install opencv-python")
        print("2. Run: pip install numpy")
        print("3. Run: pip install Pillow")

if __name__ == "__main__":
    main() 