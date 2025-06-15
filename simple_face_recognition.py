import os
import cv2
import numpy as np
import pickle
from datetime import datetime
import json

class SimpleFaceRecognitionSystem:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # Create directories if they don't exist
        self.faces_dir = "known_faces"
        self.encodings_file = "face_encodings_simple.pkl"
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
    
    def register_new_face(self, name):
        """Register a new face by capturing and encoding it"""
        print(f"Registering new face for: {name}")
        print("Press 'c' to capture the face, 'q' to quit")
        
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
            
            # Draw rectangles around detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "Press 'c' to capture", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Register New Face', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and len(faces) > 0:
                # Capture the first detected face
                x, y, w, h = faces[0]
                face_image = frame[y:y+h, x:x+w]
                
                # Extract features
                face_encoding = self.extract_face_features(face_image)
                
                # Check if face already exists
                if len(self.known_face_encodings) > 0:
                    for existing_encoding in self.known_face_encodings:
                        if self.compare_faces(face_encoding, existing_encoding):
                            print("Face already registered!")
                            break
                    else:
                        # Add new face
                        self.known_face_encodings.append(face_encoding)
                        self.known_face_names.append(name)
                        
                        # Save face image
                        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                        face_image_path = os.path.join(self.faces_dir, f"{name}_{timestamp}.jpg")
                        cv2.imwrite(face_image_path, face_image)
                        
                        # Save encodings
                        self.save_known_faces()
                        
                        print(f"Face registered successfully for: {name}")
                else:
                    # First face, add it
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
                    
                    # Save face image
                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                    face_image_path = os.path.join(self.faces_dir, f"{name}_{timestamp}.jpg")
                    cv2.imwrite(face_image_path, face_image)
                    
                    # Save encodings
                    self.save_known_faces()
                    
                    print(f"Face registered successfully for: {name}")
                break
            elif key == ord('q'):
                print("Registration cancelled")
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def run_recognition(self):
        """Run the main face recognition loop"""
        if len(self.known_face_encodings) == 0:
            print("No faces registered. Please register some faces first.")
            return
        
        print("Starting face recognition...")
        print("Press 'q' to quit, 'r' to register new face")
        
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
                best_match = 0
                
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
            elif key == ord('r'):
                name = input("Enter name for new face: ")
                if name.strip():
                    self.register_new_face(name.strip())
        
        cap.release()
        cv2.destroyAllWindows()
    
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
    print("=== Simple Face Recognition System ===")
    print("(Uses OpenCV only - no dlib required)")
    system = SimpleFaceRecognitionSystem()
    
    while True:
        print("\nOptions:")
        print("1. Start face recognition")
        print("2. Register new face")
        print("3. List registered faces")
        print("4. Delete face")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ").strip()
        
        if choice == '1':
            system.run_recognition()
        elif choice == '2':
            name = input("Enter name for the new face: ").strip()
            if name:
                system.register_new_face(name)
            else:
                print("Name cannot be empty!")
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