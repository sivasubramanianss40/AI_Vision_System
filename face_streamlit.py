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
import mediapipe as mp
import tempfile
import time
from datetime import datetime
import threading
import queue
import shutil

# Configure Streamlit page
st.set_page_config(
    page_title="AI Vision System",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI with minimal labels
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1e3d59;
        text-align: center;
        margin-bottom: 1.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.3rem;
        color: #2e86ab;
        margin: 0.8rem 0;
        padding: 0.3rem 0;
        border-bottom: 2px solid #a23b72;
    }
    .status-success {
        color: #28a745;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .status-error {
        color: #dc3545;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .status-info {
        color: #17a2b8;
        font-weight: bold;
        font-size: 0.9rem;
    }
    .detection-label {
        background: rgba(30, 61, 89, 0.9);
        color: white;
        padding: 0.15rem 0.4rem;
        border-radius: 8px;
        font-size: 0.65rem;
        font-weight: bold;
        display: inline-block;
        margin: 0.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3);
        backdrop-filter: blur(5px);
    }
    .minimal-label {
        font-size: 0.6rem !important;
        padding: 0.1rem 0.3rem !important;
        border-radius: 4px !important;
    }
    .compact-info {
        font-size: 0.7rem;
        color: #666;
        margin: 0.05rem 0;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        border: none;
        background: linear-gradient(45deg, #1e3d59, #2e86ab);
        color: white;
        font-weight: bold;
        font-size: 0.9rem;
        padding: 0.4rem 0.8rem;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.2);
    }
    .video-container {
        border: 2px solid #2e86ab;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .reset-button > button {
        background: linear-gradient(45deg, #dc3545, #c82333) !important;
        color: white !important;
    }
    .reset-button > button:hover {
        background: linear-gradient(45deg, #c82333, #bd2130) !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'face_count' not in st.session_state:
    st.session_state.face_count = 0
if 'person_name' not in st.session_state:
    st.session_state.person_name = ""
if 'recognition_active' not in st.session_state:
    st.session_state.recognition_active = False

def create_directories():
    """Create necessary directories"""
    base_dir = "face_data"
    train_dir = os.path.join(base_dir, "train")
    os.makedirs(train_dir, exist_ok=True)
    return base_dir, train_dir

def reset_model_data():
    """Reset all model data and training data"""
    try:
        base_dir = "face_data"
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir)
        
        # Reset session state
        st.session_state.model_trained = False
        st.session_state.face_count = 0
        st.session_state.person_name = ""
        st.session_state.recognition_active = False
        
        return True
    except Exception as e:
        st.error(f"Failed to reset model data: {str(e)}")
        return False

def capture_faces_from_webcam(person_name, num_images=20):
    """Optimized face capture with better performance"""
    base_dir, train_dir = create_directories()
    person_dir = os.path.join(train_dir, person_name)
    os.makedirs(person_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(0)
    # Optimize camera settings for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Reduced resolution
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 20)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    count = 0
    frame_skip = 0
    
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
            
            # Skip frames for better performance
            frame_skip += 1
            if frame_skip % 3 != 0:  # Process every 3rd frame
                continue
            
            # Use smaller frame for face detection
            small_frame = cv2.resize(frame, (160, 120))
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces with faster model
            face_locations = face_recognition.face_locations(rgb_small, model="hog", number_of_times_to_upsample=1)
            
            # Draw rectangle around faces
            display_frame = frame.copy()
            for top, right, bottom, left in face_locations:
                # Scale back up face locations
                scale_x = frame.shape[1] / small_frame.shape[1]
                scale_y = frame.shape[0] / small_frame.shape[0]
                
                top = int(top * scale_y)
                right = int(right * scale_x)
                bottom = int(bottom * scale_y)
                left = int(left * scale_x)
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{count+1}/{num_images}", 
                           (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show current frame with container width
            rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            image_placeholder.image(rgb_display, channels="RGB", use_container_width=True)
            
            # Capture face if detected
            if face_locations:
                for top, right, bottom, left in face_locations:
                    scale_x = frame.shape[1] / small_frame.shape[1]
                    scale_y = frame.shape[0] / small_frame.shape[0]
                    
                    top = int(top * scale_y)
                    right = int(right * scale_x)
                    bottom = int(bottom * scale_y)
                    left = int(left * scale_x)
                    
                    # Add padding
                    padding = 10
                    top = max(0, top - padding)
                    left = max(0, left - padding)
                    bottom = min(frame.shape[0], bottom + padding)
                    right = min(frame.shape[1], right + padding)
                    
                    face_img = frame[top:bottom, left:right]
                    
                    if face_img.size > 0:
                        filename = os.path.join(person_dir, f"{person_name}_{count:03d}.jpg")
                        cv2.imwrite(filename, face_img)
                        count += 1
                        
                        # Update progress
                        progress = count / num_images
                        progress_bar.progress(progress)
                        status_placeholder.markdown(f'<p class="status-info">✅ Captured {count}/{num_images}</p>', 
                                                  unsafe_allow_html=True)
                        
                        time.sleep(0.05)  # Minimal delay
                        break
            
            if count >= num_images:
                break
                
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        cap.release()
    
    if count > 0:
        st.session_state.face_count = count
        status_placeholder.markdown(f'<p class="status-success">🎉 Captured {count} images!</p>', 
                                  unsafe_allow_html=True)
        return True
    else:
        status_placeholder.markdown('<p class="status-error">❌ No faces captured</p>', 
                                  unsafe_allow_html=True)
        return False

def train_face_model():
    """Optimized model training"""
    try:
        base_dir, train_dir = create_directories()
        
        # Check if training data exists
        if not os.path.exists(train_dir) or not os.listdir(train_dir):
            st.error("No training data found. Please capture faces first.")
            return False
        
        with st.spinner("Training face recognition model..."):
            # Optimized transform
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet stats
            ])
            
            # Dataset
            dataset = ImageFolder(train_dir, transform=transform)
            loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)  # Reduced batch size
            idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
            
            # Lighter model for better performance
            model = timm.create_model("efficientnet_b0", pretrained=True)
            model.classifier = torch.nn.Identity()
            model.eval()
            
            # Extract embeddings
            prototypes = {}
            progress_bar = st.progress(0)
            total_batches = len(loader)
            
            with torch.no_grad():
                for i, (imgs, labels) in enumerate(loader):
                    embs = model(imgs)
                    for emb, label in zip(embs, labels):
                        name = idx_to_class[label.item()]
                        if name not in prototypes:
                            prototypes[name] = []
                        prototypes[name].append(emb)
                    
                    progress_bar.progress((i + 1) / total_batches)
            
            # Mean embeddings = class prototypes
            for name in prototypes:
                prototypes[name] = torch.stack(prototypes[name]).mean(dim=0)
            
            # Save model + prototypes
            model_path = os.path.join(base_dir, "efficient_facemodel.pt")
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

class RealTimeRecognition:
    def __init__(self):
        self.model_path = "face_data/efficient_facemodel.pt"
        self.model = None
        self.prototypes = None
        self.yolo = None
        self.mtcnn = None
        self.hands = None
        self.mp_draw = None
        self.transform = None
        self.frame_count = 0
        self.last_detection_time = 0
        self.detection_cache = {}
        self.setup_models()
    
    def setup_models(self):
        """Initialize models with performance optimizations"""
        try:
            # Face recognition model
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location="cpu")
                self.model = timm.create_model("efficientnet_b0", pretrained=False)
                self.model.classifier = torch.nn.Identity()
                self.model.load_state_dict(checkpoint["model_state_dict"])
                self.model.eval()
                self.prototypes = checkpoint["prototypes"]
                
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
            
            # Lighter YOLO model
            self.yolo = YOLO("yolov8n.pt")
            
            # Optimized MTCNN
            self.mtcnn = MTCNN(keep_all=True, device="cpu", min_face_size=30, 
                              thresholds=[0.6, 0.7, 0.7])
            
            # Optimized MediaPipe
            mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.hands = mp_hands.Hands(
                static_image_mode=False, 
                max_num_hands=1,  # Reduced for performance
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6
            )
            
        except Exception as e:
            st.error(f"Model setup failed: {str(e)}")
    
    def identify_face(self, face_img_pil):
        """Fast face identification with caching"""
        if self.model is None or self.prototypes is None:
            return "Unknown"
        
        try:
            # Simple cache based on image hash
            img_hash = hash(face_img_pil.tobytes())
            if img_hash in self.detection_cache:
                return self.detection_cache[img_hash]
            
            face_tensor = self.transform(face_img_pil).unsqueeze(0)
            with torch.no_grad():
                emb = self.model(face_tensor).squeeze(0)
            
            dists = {name: torch.norm(emb - proto).item() for name, proto in self.prototypes.items()}
            best_match = min(dists, key=dists.get)
            
            # Adjusted threshold for better performance
            result = best_match if dists[best_match] < 1.0 else "Unknown"
            
            # Cache result
            self.detection_cache[img_hash] = result
            
            # Limit cache size
            if len(self.detection_cache) > 50:
                self.detection_cache.clear()
            
            return result
        except:
            return "Unknown"
    
    def process_frame(self, frame):
        """Highly optimized frame processing"""
        self.frame_count += 1
        frame = cv2.flip(frame, 1)
        
        # Process at lower resolution for speed
        process_frame = cv2.resize(frame, (320, 240))
        rgb = cv2.cvtColor(process_frame, cv2.COLOR_BGR2RGB)
        
        face_names, object_names, hand_labels = [], [], []
        current_time = time.time()
        
        # Face Detection & Recognition (every 6 frames for better performance)
        if self.mtcnn and self.frame_count % 6 == 0:
            boxes, _ = self.mtcnn.detect(process_frame)
            if boxes is not None and len(boxes) > 0:
                # Only process first face for speed
                box = boxes[0]
                x1, y1, x2, y2 = map(int, box)
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(process_frame.shape[1], x2)
                y2 = min(process_frame.shape[0], y2)
                
                # Scale to original frame
                scale_x = frame.shape[1] / process_frame.shape[1]
                scale_y = frame.shape[0] / process_frame.shape[0]
                
                orig_x1 = int(x1 * scale_x)
                orig_y1 = int(y1 * scale_y)
                orig_x2 = int(x2 * scale_x)
                orig_y2 = int(y2 * scale_y)
                
                face_crop = process_frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
                    name = self.identify_face(face_pil)
                    face_names.append(name)
                    
                    # Draw minimal rectangle
                    cv2.rectangle(frame, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 1)
                    # Minimal text
                    cv2.putText(frame, name[:8], (orig_x1, orig_y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Object Detection (every 10 frames)
        if self.yolo and self.frame_count % 10 == 0:
            results = self.yolo(process_frame, verbose=False)
            if results and len(results[0].boxes) > 0:
                class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                # Only get top 2 objects
                unique_objects = list(set([self.yolo.names[cid] for cid in class_ids[:2] 
                                         if cid < len(self.yolo.names) and self.yolo.names[cid] != "person"]))
                object_names = unique_objects[:1]  # Only show 1 object
        
        # Hand Detection (every 8 frames)
        if self.hands and self.frame_count % 8 == 0:
            hand_result = self.hands.process(rgb)
            if hand_result.multi_hand_landmarks:
                hand_labels.append("👋")  # Simple indicator
        
        # Generate ultra-compact label
        label_text = self.generate_minimal_label(face_names, object_names, hand_labels)
        
        # Minimal overlay
        if label_text and label_text != "...":
            label_width = min(len(label_text) * 8 + 10, frame.shape[1] - 10)
            cv2.rectangle(frame, (5, 5), (label_width, 20), (0, 0, 0), -1)
            cv2.putText(frame, label_text, (8, 16), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame, label_text
    
    def generate_minimal_label(self, face_names, object_names, hand_labels):
        """Generate ultra-compact label"""
        if face_names and face_names[0] != "Unknown":
            name = face_names[0][:6]  # Limit name length
            if object_names:
                return f"{name}+{object_names[0][:4]}"
            elif hand_labels:
                return f"{name}👋"
            else:
                return name
        elif object_names:
            return object_names[0][:6]
        elif hand_labels:
            return "👋"
        else:
            return "..."

def main():
    # Header
    st.markdown('<h1 class="main-header">👁️ AI Vision System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        st.markdown("---")
        
        # Navigation
        page = st.selectbox("Select Mode", 
                           ["📸 Capture Faces", "🧠 Train Model", "🔍 Live Recognition"])
        
        st.markdown("---")
        st.markdown("### ℹ️ System Status")
        if st.session_state.face_count > 0:
            st.success(f"✅ {st.session_state.face_count} faces captured")
        else:
            st.info("📸 No faces captured yet")
            
        if st.session_state.model_trained:
            st.success("✅ Model trained")
        else:
            st.warning("⚠️ Model not trained")
        
        st.markdown("---")
        st.markdown("### 🔄 Reset Options")
        
        # Reset button with custom styling
        reset_col1, reset_col2 = st.columns([1, 1])
        with reset_col1:
            if st.button("🗑️ Reset All", key="reset_button", help="Delete all face data and reset model"):
                if st.session_state.get('confirm_reset', False):
                    if reset_model_data():
                        st.success("✅ All data reset!")
                        st.rerun()
                else:
                    st.session_state.confirm_reset = True
                    st.warning("Click again to confirm reset")
        
        with reset_col2:
            if st.session_state.get('confirm_reset', False):
                if st.button("❌ Cancel"):
                    st.session_state.confirm_reset = False
                    st.rerun()
    
    # Main content based on selected page
    if page == "📸 Capture Faces":
        st.markdown('<h2 class="section-header">📸 Face Capture</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Settings")
            person_name = st.text_input("Person Name", value="", placeholder="Enter person's name")
            num_images = st.slider("Number of Images", 10, 30, 15)  # Reduced max for speed
            
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
            **Optimized Training:**
            • Uses EfficientNet for faster processing
            • Reduced batch size for better performance
            • Optimized for real-time recognition
            
            **Requirements:**
            • At least 10 images per person
            • Good lighting conditions
            • Clear face visibility
            """)
    
    elif page == "🔍 Live Recognition":
        st.markdown('<h2 class="section-header">🔍 Live Recognition</h2>', unsafe_allow_html=True)
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first before starting live recognition")
            return
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### Controls")
            
            start_btn = st.button("🎥 Start", type="primary")
            stop_btn = st.button("⏹️ Stop", type="secondary")
            
            if stop_btn:
                st.session_state.recognition_active = False
            
            if start_btn:
                st.session_state.recognition_active = True
            
            st.markdown("### Performance")
            fps_display = st.empty()
            detection_display = st.empty()
        
        with col1:
            st.markdown("### Live Feed")
            video_placeholder = st.empty()
            
            if st.session_state.recognition_active:
                recognizer = RealTimeRecognition()
                cap = cv2.VideoCapture(0)
                
                # Optimized camera settings
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
                cap.set(cv2.CAP_PROP_FPS, 25)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                frame_count = 0
                start_time = time.time()
                
                try:
                    while st.session_state.recognition_active:
                        ret, frame = cap.read()
                        if not ret:
                            st.error("Failed to read from camera")
                            break
                        
                        # Process frame
                        processed_frame, label_text = recognizer.process_frame(frame)
                        
                        # Convert to RGB for Streamlit (fixed deprecated parameter)
                        rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                        
                        # Update performance info every 30 frames
                        frame_count += 1
                        if frame_count % 30 == 0:
                            elapsed = time.time() - start_time
                            fps = 30 / elapsed if elapsed > 0 else 0
                            
                            fps_display.markdown(f"**FPS:** {fps:.1f}")
                            detection_display.markdown(f'<div class="detection-label minimal-label">{label_text}</div>', 
                                                     unsafe_allow_html=True)
                            
                            start_time = time.time()
                        
                        # Minimal delay for smooth playback
                        time.sleep(0.02)  # ~50 FPS max
                        
                except Exception as e:
                    st.error(f"Recognition error: {str(e)}")
                finally:
                    cap.release()
                    st.session_state.recognition_active = False

if __name__ == "__main__":
    main()