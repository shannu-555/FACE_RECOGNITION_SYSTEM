import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import json
import glob

class ImageBasedFaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create directories if they don't exist
        self.faces_dir = "known_faces"
        self.encodings_file = "face_encodings_images.pkl"
        self.create_directories()
        
        # Load existing face encodings
        self.load_known_faces()
    
    def create_directories(self):
        """Create necessary directories"""
        if not os.path.exists(self.faces_dir):
            os.makedirs(self.faces_dir)
            print(f"Created directory: {self.faces_dir}")
    
    def load_known_faces(self):
        """Load existing face encodings from file"""
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data['encodings']
                    self.known_face_names = data['names']
                print(f"Loaded {len(self.known_face_names)} known faces")
            except Exception as e:
                print(f"Error loading face encodings: {e}")
    
    def save_known_faces(self):
        """Save face encodings to file"""
        try:
            data = {
                'encodings': self.known_face_encodings,
                'names': self.known_face_names
            }
            with open(self.encodings_file, 'wb') as f:
                pickle.dump(data, f)
            print("Face encodings saved successfully")
        except Exception as e:
            print(f"Error saving face encodings: {e}")
    
    def extract_face_features(self, face_image):
        """Extract simple features from face image using OpenCV"""
        # Resize to standard size
        face_image = cv2.resize(face_image, (100, 100))
        
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Apply histogram equalization
        gray = cv2.equalizeHist(gray)
        
        # Flatten the image to create a feature vector
        features = gray.flatten()
        
        # Normalize features
        features = features.astype(np.float32) / 255.0
        
        return features
    
    def compare_faces(self, face_encoding1, face_encoding2, threshold=0.8):
        """Compare two face encodings using cosine similarity"""
        # Calculate cosine similarity
        dot_product = np.dot(face_encoding1, face_encoding2)
        norm1 = np.linalg.norm(face_encoding1)
        norm2 = np.linalg.norm(face_encoding2)
        
        if norm1 == 0 or norm2 == 0:
            return False
        
        similarity = dot_product / (norm1 * norm2)
        return similarity > threshold
    
    def detect_faces_in_image(self, image_path):
        """Detect faces in an image file"""
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return []
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        return faces, image
    
    def import_face_from_image(self, image_path, name):
        """Import a face from an image file"""
        print(f"Importing face from: {image_path}")
        
        # Detect faces in the image
        faces, image = self.detect_faces_in_image(image_path)
        
        if len(faces) == 0:
            print("No faces detected in the image!")
            return False
        
        if len(faces) > 1:
            print(f"Multiple faces detected ({len(faces)}). Using the first one.")
        
        # Extract the first face
        x, y, w, h = faces[0]
        face_image = image[y:y+h, x:x+w]
        
        # Extract features
        face_encoding = self.extract_face_features(face_image)
        
        # Check if face already exists
        if len(self.known_face_encodings) > 0:
            for existing_encoding in self.known_face_encodings:
                if self.compare_faces(face_encoding, existing_encoding):
                    print("Face already registered!")
                    return False
        
        # Add new face
        self.known_face_encodings.append(face_encoding)
        self.known_face_names.append(name)
        
        # Save face image
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        face_image_path = os.path.join(self.faces_dir, f"{name}_{timestamp}.jpg")
        cv2.imwrite(face_image_path, face_image)
        
        # Save encodings
        self.save_known_faces()
        
        print(f"Face imported successfully for: {name}")
        return True
    
    def import_faces_from_directory(self, directory_path):
        """Import faces from all images in a directory"""
        print(f"Scanning directory: {directory_path}")
        
        # Supported image formats
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(directory_path, ext)))
            image_files.extend(glob.glob(os.path.join(directory_path, ext.upper())))
        
        if not image_files:
            print("No image files found in the directory!")
            return
        
        print(f"Found {len(image_files)} image files")
        
        for image_path in image_files:
            # Extract name from filename (remove extension)
            filename = os.path.basename(image_path)
            name = os.path.splitext(filename)[0]
            
            print(f"\nProcessing: {filename}")
            self.import_face_from_image(image_path, name)
    
    def import_single_image(self):
        """Import a single image file"""
        print("\nSupported formats: jpg, jpeg, png, bmp, tiff")
        image_path = input("Enter the path to the image file: ").strip()
        
        if not os.path.exists(image_path):
            print("File not found!")
            return
        
        name = input("Enter the name for this person: ").strip()
        if not name:
            print("Name cannot be empty!")
            return
        
        self.import_face_from_image(image_path, name)
    
    def import_from_directory(self):
        """Import faces from a directory"""
        directory_path = input("Enter the path to the directory containing images: ").strip()
        
        if not os.path.exists(directory_path):
            print("Directory not found!")
            return
        
        if not os.path.isdir(directory_path):
            print("Path is not a directory!")
            return
        
        self.import_faces_from_directory(directory_path)
    
    def preview_image_faces(self, image_path):
        """Preview faces detected in an image"""
        faces, image = self.detect_faces_in_image(image_path)
        
        if len(faces) == 0:
            print("No faces detected in the image!")
            return
        
        # Draw rectangles around detected faces
        for i, (x, y, w, h) in enumerate(faces):
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, f"Face {i+1}", (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display the image
        cv2.imshow('Detected Faces', image)
        print(f"Detected {len(faces)} face(s). Press any key to close.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def run_recognition(self):
        """Run the main face recognition loop"""
        if len(self.known_face_encodings) == 0:
            print("No faces registered. Please import some face images first.")
            return
        
        print("Starting face recognition...")
        print("Press 'q' to quit, 'i' to import new images")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_image = frame[y:y+h, x:x+w]
                
                # Extract features
                face_encoding = self.extract_face_features(face_image)
                
                # Compare with known faces
                name = "Unknown"
                
                for i, known_encoding in enumerate(self.known_face_encodings):
                    if self.compare_faces(face_encoding, known_encoding, threshold=0.7):
                        name = self.known_face_names[i]
                        break
                
                # Draw rectangle and label
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw label
                cv2.rectangle(frame, (x, y - 35), (x + w, y), color, cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (x + 6, y - 6), font, 0.6, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('i'):
                self.show_import_menu()
        
        cap.release()
        cv2.destroyAllWindows()
    
    def show_import_menu(self):
        """Show menu for importing images"""
        print("\n=== Import Images Menu ===")
        print("1. Import single image")
        print("2. Import from directory")
        print("3. Preview faces in image")
        print("4. Back to main menu")
        
        choice = input("Enter your choice (1-4): ").strip()
        
        if choice == '1':
            self.import_single_image()
        elif choice == '2':
            self.import_from_directory()
        elif choice == '3':
            image_path = input("Enter the path to the image file: ").strip()
            if os.path.exists(image_path):
                self.preview_image_faces(image_path)
            else:
                print("File not found!")
        elif choice == '4':
            pass
        else:
            print("Invalid choice!")
    
    def list_registered_faces(self):
        """List all registered faces"""
        if len(self.known_face_names) == 0:
            print("No faces registered yet.")
        else:
            print("Registered faces:")
            for i, name in enumerate(self.known_face_names, 1):
                print(f"{i}. {name}")
    
    def delete_face(self, name):
        """Delete a registered face"""
        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            self.known_face_names.pop(index)
            self.known_face_encodings.pop(index)
            self.save_known_faces()
            print(f"Deleted face: {name}")
        else:
            print(f"Face '{name}' not found")

def main():
    print("=== Image-Based Face Recognition System ===")
    print("(Import face images from files)")
    system = ImageBasedFaceRecognitionSystem()
    
    while True:
        print("\nOptions:")
        print("1. Start face recognition")
        print("2. Import face images")
        print("3. List registered faces")
        print("4. Delete face")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            system.run_recognition()
        elif choice == '2':
            system.show_import_menu()
        elif choice == '3':
            system.list_registered_faces()
        elif choice == '4':
            system.list_registered_faces()
            name = input("Enter name to delete: ").strip()
            if name:
                system.delete_face(name)
        elif choice == '5':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 