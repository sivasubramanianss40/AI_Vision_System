import streamlit as st
import cv2
import os
import face_recognition
from PIL import Image
import numpy as np
import torch
import timm
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN
from ultralytics import YOLO
import time
from datetime import datetime
import shutil
import mediapipe as mp

# Configure Streamlit page
st.set_page_config(
    page_title="AI Vision System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin: 1rem 0;
        padding: 0.5rem 0;
        border-bottom: 2px solid #a23b72;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
    }
    .status-info {
        color: #17a2b8;
        font-weight: bold;
    }
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: none;
        background: linear-gradient(45deg, #1e3d59, #2e86ab);
        color: white;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    base_dir = "face_data"
    model_path = os.path.join(base_dir, "swin_fewshot_model.pt")
    st.session_state.model_trained = os.path.exists(model_path)
if 'face_count' not in st.session_state:
    base_dir = "face_data"
    train_dir = os.path.join(base_dir, "train")
    face_count = 0
    if os.path.exists(train_dir):
        for person_dir in os.listdir(train_dir):
            person_path = os.path.join(train_dir, person_dir)
            if os.path.isdir(person_path):
                face_count += len([f for f in os.listdir(person_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    st.session_state.face_count = face_count
if 'person_name' not in st.session_state:
    st.session_state.person_name = ""

def create_directories():
    """Create necessary directories"""
    base_dir = "face_data"
    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    return base_dir, train_dir

def capture_faces_from_webcam(person_name, num_images=20):
    """Capture faces from webcam"""
    base_dir, train_dir = create_directories()
    person_dir = os.path.join(train_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    count = 0
    captured_images = []
    
    # Create placeholders for dynamic updates
    status_placeholder = st.empty()
    image_placeholder = st.empty()
    progress_bar = st.progress(0)
    
    try:
        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to read from webcam")
                break
            
            frame = cv2.flip(frame, 1)  # Mirror effect
            
            # Resize frame for faster face detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small = small_frame[:, :, ::-1]
            
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small)
            
            # Draw rectangle around faces
            display_frame = frame.copy()
            for top, right, bottom, left in face_locations:
                # Scale back up face locations
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Face {count+1}/{num_images}", 
                            (left, top-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)
            
            # Show current frame
            rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(rgb_display, channels="RGB", use_container_width=True)
            
            # Capture face if detected
            if face_locations:
                for top, right, bottom, left in face_locations:
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Add padding
                    padding = 20
                    top = max(0, top - padding)
                    left = max(0, left - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    right = min(frame.shape[1], right + padding)
                    
                    face_img = frame[top:bottom, left:right]
                    
                    if face_img.size > 0:
                        filename = os.path.join(person_dir, f"{person_name}_{count:03d}.jpg")
                        cv2.imwrite(filename, face_img)
                        captured_images.append(face_img)
                        count += 1
                        
                        # Update progress
                        progress = count / num_images
                        progress_bar.progress(progress)
                        status_placeholder.markdown(f'<p class="status-info">✅ Captured {count}/{num_images} images</p>', 
                                                  unsafe_allow_html=True)
                        
                        time.sleep(0.2)  # Small delay between captures
                        break
            
            if count >= num_images:
                break
                
    except Exception as e:
        st.error(f"Error during capture: {str(e)}")
    finally:
        cap.release()
    
    if count > 0:
        st.session_state.face_count += count
        status_placeholder.markdown(f'<p class="status-success">🎉 Successfully captured {count} face images!</p>', 
                                  unsafe_allow_html=True)
        return True
    else:
        status_placeholder.markdown(f'<p class="status-error">❌ No faces captured</p>', 
                                  unsafe_allow_html=True)
        return False

def train_face_model():
    """Train the face recognition model"""
    try:
        base_dir, train_dir = create_directories()
        
        # Check if training data exists
        if not os.path.exists(train_dir) or not os.listdir(train_dir):
            st.error("No training data found. Please capture faces first.")
            return False
        
        with st.spinner("Training face recognition model..."):
            # Transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5]*3, [0.5]*3)
            ])
            
            # Dataset
            dataset = ImageFolder(train_dir, transform=transform)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
            
            # Swin Transformer encoder
            model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
            model.head = torch.nn.Identity()
            model.eval()
            
            # Extract embeddings
            prototypes = {}
            progress_bar = st.progress(0)
            total_samples = len(loader)
            
            with torch.no_grad():
                for i, (img, label) in enumerate(loader):
                    name = idx_to_class[label.item()]
                    emb = model(img).squeeze(0)
                    if name not in prototypes:
                        prototypes[name] = []
                    prototypes[name].append(emb)
                    
                    progress_bar.progress((i + 1) / total_samples)
            
            # Mean embeddings = class prototypes
            for name in prototypes:
                prototypes[name] = torch.stack(prototypes[name]).mean(dim=0)
            
            # Save model + prototypes
            model_path = os.path.join(base_dir, "swin_fewshot_model.pt")
            torch.save({
                'model_state_dict': model.state_dict(),
                'prototypes': prototypes
            }, model_path)
            
            st.session_state.model_trained = True
            st.success("✅ Face recognition model trained successfully!")
            return True
            
    except Exception as e:
        st.error(f"Training failed: {str(e)}")
        return False

def reset_model():
    """Reset the trained model and session state"""
    try:
        base_dir = "face_data"
        model_path = os.path.join(base_dir, "swin_fewshot_model.pt")
        
        # Remove model file if it exists
        if os.path.exists(model_path):
            os.remove(model_path)
        
        # Remove face_data directory
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        
        # Reset session state
        st.session_state.model_trained = False
        st.session_state.face_count = 0
        st.session_state.person_name = ""
        
        st.success("✅ Model and data reset successfully!")
    except Exception as e:
        st.error(f"Reset failed: {str(e)}")

class RealTimeRecognition:
    def __init__(self):
        self.model_path = "face_data/swin_fewshot_model.pt"
        self.model = None
        self.prototypes = None
        self.yolo_objects = None
        self.mtcnn = None
        self.hands = None
        self.mp_draw = None
        self.transform = None
        self.setup_models()
    
    def setup_models(self):
        """Initialize all models"""
        try:
            # Face recognition model
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location="cpu")
                self.model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
                self.model.head = torch.nn.Identity()
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self.prototypes = checkpoint["prototypes"]
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5]*3, [0.5]*3)
                ])
            
            # YOLO for object detection
            try:
                self.yolo_objects = YOLO("ultralytics/yolov10n.pt")
                if torch.cuda.is_available():
                    self.yolo_objects.to('cuda')  # Enable GPU for faster inference
            except Exception as e:
                st.error(f"Failed to load YOLOv10n model for objects: {str(e)}")
                self.yolo_objects = None
            
            # MediaPipe for hand detection
            try:
                mp_hands = mp.solutions.hands
                self.mp_draw = mp.solutions.drawing_utils
                self.hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7
                )
            except Exception as e:
                st.error(f"Failed to load MediaPipe Hands: {str(e)}")
                self.hands = None
            
            # MTCNN for face detection
            self.mtcnn = MTCNN(keep_all=True, device="cpu", min_face_size=40)
            
        except Exception as e:
            st.error(f"Model setup failed: {str(e)}")
    
    def identify_face(self, face_img_pil):
        """Identify face using trained model"""
        if self.model is None or self.prototypes is None:
            return "Unknown"
        
        try:
            face_tensor = self.transform(face_img_pil).unsqueeze(0)
            with torch.no_grad():
                emb = self.model(face_tensor).squeeze(0)
            dists = {name: torch.norm(emb - proto).item() for name, proto in self.prototypes.items()}
            return min(dists, key=dists.get)
        except:
            return "Unknown"
    
    def process_frame(self, frame):
        """Process single frame for recognition"""
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_names, object_names, hand_labels = [], [], []
        
        # Face Detection & Recognition
        if self.mtcnn:
            boxes, _ = self.mtcnn.detect(frame)
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2 = min(frame.shape[1], x2)
                    y2 = min(frame.shape[0], y2)
                    
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size == 0:
                        continue
                    
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    name = self.identify_face(face_pil)
                    face_names.append(name)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        # Object Detection
        if self.yolo_objects:
            results = self.yolo_objects(frame, imgsz=416, verbose=False, half=torch.cuda.is_available())
            if results and len(results[0].boxes) > 0:
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                object_names = list({self.yolo_objects.names[cid] for cid in class_ids if self.yolo_objects.names[cid].lower() != "person"})
        
        # Hand Detection
        if self.hands:
            hand_result = self.hands.process(rgb)
            if hand_result.multi_hand_landmarks:
                for hand_landmarks, hand_info in zip(hand_result.multi_hand_landmarks, 
                                                   hand_result.multi_handedness):
                    label = hand_info.classification[0].label
                    hand_labels.append(f"{label} Hand")
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, 
                                              mp.solutions.hands.HAND_CONNECTIONS)
        
        # Generate smart label
        label_text = self.generate_label(face_names, object_names, hand_labels)
        
        # Display label
        cv2.rectangle(frame, (5, 5), (frame.shape[1] - 5, 35), (0, 0, 0), -1)
        cv2.putText(frame, label_text, (10, 25), 
                   cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)
        
        return frame, label_text
    
    def generate_label(self, face_names, object_names, hand_labels):
        """Generate smart label text"""
        label_text = ""
        clean_objects = [obj for obj in object_names if obj.lower() != "person"]
        
        if face_names:
            label_text = " & ".join(face_names)
            if clean_objects:
                label_text += f" with {', '.join(clean_objects[:3])}"  # Limit to 3 objects
        elif clean_objects:
            label_text = f"{', '.join(clean_objects[:3])}"
        
        if hand_labels:
            if label_text:
                label_text += f" showing {', '.join(hand_labels)}"
            else:
                label_text = f"{', '.join(hand_labels)}"
        
        if not label_text:
            label_text = "Scanning..."
        
        return label_text
    

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ AI Vision System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        st.markdown("---")
        
        # Mode selection
        page = st.selectbox("Select Mode", 
                           ["📸 Face Capture", "🧠 Train Model", "🔍 Live Recognition"])
        
        st.markdown("---")
        st.markdown("### ℹ️ System Information")
        if st.session_state.face_count > 0:
            st.success(f"✅ {st.session_state.face_count} faces captured")
        else:
            st.info("📸 No faces captured yet")
            
        if st.session_state.model_trained:
            st.success("✅ Model trained")
        else:
            st.warning("⚠️ Model not trained")
        
        st.markdown("---")
        reset_btn = st.button("🔴 Reset Model", type="secondary")
        if reset_btn:
            reset_model()
    
    # Main content based on selected page
    if page == "📸 Face Capture":
        st.markdown('<h2 class="section-header">📸 Face Capture</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Settings")
            person_name = st.text_input("Person Name", value="", placeholder="Enter person's name")
            num_images = st.slider("Number of Images", 10, 50, 20)
            
            capture_btn = st.button("🎥 Start Capture", type="primary")
            
            if capture_btn and person_name:
                st.session_state.person_name = person_name
                with col2:
                    success = capture_faces_from_webcam(person_name, num_images)
            elif capture_btn and not person_name:
                st.error("Please enter a person's name")
        
        with col2:
            st.markdown("### Live Preview")
            if not capture_btn:
                st.info("👆 Click 'Start Capture' to begin")
    
    elif page == "🧠 Train Model":
        st.markdown('<h2 class="section-header">🧠 Model Training</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("### Training Status")
            
            # Check training data
            base_dir, train_dir = create_directories()
            if os.path.exists(train_dir):
                subdirs = [d for d in os.listdir(train_dir) 
                          if os.path.isdir(os.path.join(train_dir, d))]
                if subdirs:
                    st.success(f"Found {len(subdirs)} person(s): {', '.join(subdirs)}")
                    for person in subdirs:
                        person_path = os.path.join(train_dir, person)
                        image_count = len([f for f in os.listdir(person_path) 
                                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                        st.info(f"📁 {person}: {image_count} images")
                else:
                    st.warning("No training data found")
            
            train_btn = st.button("🚀 Train Model", type="primary")
            
            if train_btn:
                success = train_face_model()
        
        with col2:
            st.markdown("### Training Info")
            st.info("""
            **Training Process:**
            1. Loads captured face images
            2. Uses Swin Transformer for feature extraction
            3. Creates face embeddings
            4. Saves trained model for recognition
            
            **Requirements:**
            - At least 10 images per person
            - Good lighting conditions
            - Clear face visibility
            """)
    
    elif page == "🔍 Live Recognition":
        st.markdown('<h2 class="section-header">🔍 Live Recognition</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first before starting live recognition")
            return
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.markdown("### Controls")
            start_recognition = st.button("🎥 Start Recognition", type="primary")
            stop_recognition = st.button("⏹ Stop", type="secondary")
            
            st.markdown("### Detection Info")
            detection_info = st.empty()
        
        with col1:
            st.markdown("### Live Feed")
            video_placeholder = st.empty()
            
            if start_recognition:
                recognizer = RealTimeRecognition()
                cap = cv2.VideoCapture(0)
                
                # Set camera properties for better performance
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 416)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 416)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                frame_count = 0
                
                try:
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from camera")
                            break
                        
                        # Process every 4th frame for better performance
                        if frame_count % 4 == 0:
                            processed_frame, label_text = recognizer.process_frame(frame)
                            
                            # Convert to RGB for Streamlit
                            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                            
                            # Update detection info
                            with detection_info.container():
                                st.markdown(f"**Current Detection:** {label_text}")
                                st.markdown(f"**Frame:** {frame_count}")
                                st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
                        
                        frame_count += 1
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.033)  # ~30 FPS
                        
                except Exception as e:
                    st.error(f"Recognition error: {str(e)}")
                finally:
                    cap.release()

if __name__ == "__main__":
    main()