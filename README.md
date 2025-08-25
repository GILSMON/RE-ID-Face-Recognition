````markdown
## Face Recognition and Tracking Project  

This repository contains a suite of scripts for **face recognition and tracking** using the **DeepFace** and **DeepSORT** algorithms.  
It provides both **basic functionality** and **performance evaluation** of the implemented approaches.  

---

## 🚀 Getting Started  

### ✅ Prerequisites  
- Python **3.x**  
- pip (Python package manager)  

---

### 🔧 Installation  

Clone the repository:  
```bash
git clone https://github.com/GILSMON/RE-ID-Face-Recognition
cd RE-ID-Face-Recognition
````

Create and activate a virtual environment:

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> 💡 If you don’t have a `requirements.txt`, you can generate one from your current environment:

```bash
pip freeze > requirements.txt
```

---

## 📂 Project Structure

```
your-repository/
│
├── face_rec_deepface_performance_track.py      # DeepFace recognition + performance logging
├── face_rec_deepface.py                        # Lightweight DeepFace recognition (no logging)
├── face_rec_deepSORT_performance_track.py      # DeepFace + DeepSORT tracking + performance logging
├── face_rec_deepSORT_tracking.py               # DeepFace + DeepSORT tracking (general use)
├── requirements.txt                            # Project dependencies
└── README.md                                   # Project documentation
```

---

## ▶️ Usage

Run the scripts as needed:

### 1. DeepFace Recognition (with performance tracking)

```bash
python face_rec_deepface_performance_track.py
```

### 2. DeepFace Recognition (lightweight version)

```bash
python face_rec_deepface.py
```

### 3. DeepFace + DeepSORT (with performance tracking)

```bash
python face_rec_deepSORT_performance_track.py
```

### 4. DeepFace + DeepSORT (tracking only, no logging)

```bash
python face_rec_deepSORT_tracking.py
```

---

## 📊 Features

* ✅ Face detection & recognition with **DeepFace**
* ✅ Face tracking across frames with **DeepSORT**
* ✅ Performance evaluation (FPS, CPU, Memory usage)
* ✅ Modular scripts for experimentation and lightweight usage

---

## 📝 Notes

* Use the `*_performance_track.py` scripts when you need detailed performance metrics.
* Use the lighter versions (`face_rec_deepface.py` / `face_rec_deepSORT_tracking.py`) for regular tracking tasks.
* Works best with video inputs where consistent identity tracking is required.

---

 
---

