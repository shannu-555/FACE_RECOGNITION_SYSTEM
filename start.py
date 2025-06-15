#!/usr/bin/env python3
"""
Simple startup script for Face Recognition System
"""

import os
import sys

def check_dependencies():
    """Check if required packages are installed"""
    try:
        import cv2
        import numpy
        from PIL import Image
        return True
    except ImportError:
        return False

def main():
    print("=== Face Recognition System ===")
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Required packages not found!")
        print("Please run: python install_simple.py")
        return
    
    print("✅ Dependencies OK")
    print()
    
    while True:
        print("Choose your system:")
        print("1. Image-based (Import from photos) - RECOMMENDED")
        print("2. Camera-based (Capture through webcam)")
        print("3. Install dependencies")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            print("\nStarting Image-based Face Recognition...")
            os.system("python image_based_face_recognition.py")
        elif choice == '2':
            print("\nStarting Camera-based Face Recognition...")
            os.system("python simple_face_recognition.py")
        elif choice == '3':
            print("\nInstalling dependencies...")
            os.system("python install_simple.py")
        elif choice == '4':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 