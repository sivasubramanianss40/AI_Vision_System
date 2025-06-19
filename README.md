# 🧠 Siva's Simple Vision AI System

A real-time AI system combining **face recognition**, **object detection**, and **hand gesture detection** using YOLOv10, Swin Transformer (few-shot learning), and MediaPipe.

---

## ✅ Features

- Face recognition (few-shot Swin Transformer)  
- Object detection (YOLOv10n)  
- Right/Left hand classification (MediaPipe)  
- Dynamic label logic (e.g., `"Siva"`, `"Siva with bottle"`, `"Siva showing Right Hand"`)

---

## 📁 Model Paths to Check

Ensure these paths are correct in your code:

model_path = "./Documents/face/swin_fewshot_model.pt"
yolo = YOLO("yolov10n.pt")


---

## ▶️ Run with Streamlit


streamlit run face_streamlit.py


---

## 💡 Example Outputs

- `Siva` → when only face is detected  
- `Siva with bottle` → with object detection  
- `Siva showing Right Hand` → with hand gesture  
- `No faces, hands, or objects detected` → when nothing found

---

## 📷 Notes

- Ensure your webcam is accessible  
- The app mirrors the camera like a selfie  
- Hands are labeled `Right Hand` or `Left Hand` with landmark connections

---

## 👤 Author

**Sivasubramanian M**  
ML Engineer
