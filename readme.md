#  AI Object Detection — Street Objects Detector

Detect and recognize objects in street videos and in real time using **YOLOv8 AI**!

## Author
**Asma Mansour**
Software Engineer — Data, AI & Automation

##  Features

*  **Detect 80+ Objects** — Cars, people, traffic lights, trucks, bicycles, and more!
*  **Video Detection** — Process pre-recorded `.mp4` videos frame by frame
*  **Real-Time Detection** — Detect objects directly from a webcam
*  **Confidence Scores** — Display the confidence score for each detection
*  **FPS Monitoring** — Display the real-time processing speed
*  **Modern Design** — Clean, minimal boxes with subtle corner accents
*  **Save Output** — Automatically save the processed webcam video
*  **Screenshot** — Capture any frame with the `S` key
*  **Fast & Efficient** — YOLOv8n model for fast object detection

##  Project Evolution

The project was developed in two steps.

### 1. Detection from a Video

The first version processes a pre-recorded video frame by frame.

```text
MP4 Video
    ↓
OpenCV
    ↓
YOLOv8
    ↓
Object Detection
    ↓
Annotated Video
```

### 2. Real-Time Detection

The project was then adapted to work directly with a webcam.

```text
Webcam
    ↓
OpenCV
    ↓
YOLOv8
    ↓
Real-Time Detection
    ↓
Objects + Confidence + FPS
```

The webcam version also allows the processed video to be automatically saved and screenshots to be captured during the detection.

##  What It Detects

YOLOv8 can detect more than 80 object classes, including:

| Category           | Examples                                |
| ------------------ | --------------------------------------- |
| 🚗 Vehicles        | Car, Truck, Bus, Bicycle, Motorcycle    |
| 🚶 People          | Person                                  |
| 🚦 Infrastructure  | Traffic Light, Stop Sign, Parking Meter |
| 🏙️ Street Objects | Bench, Fire Hydrant, Potted Plant       |
| 🐶 Animals         | Cat, Dog, Horse, Sheep                  |
| ✈️ Others          | Airplane, Boat, Train, and more         |

##  Technologies

* **Python 3.9+** — Programming language
* **YOLOv8** — Object detection model
* **Ultralytics** — YOLO implementation
* **OpenCV** — Video processing and visualization
* **NumPy** — Numerical operations

##  Model 
This project uses YOLOv8n, a lightweight object detection model pre-trained on the COCO dataset.
##  Project Structure

```text
ai-object-detection/
│
├── object_detection.py      # Detection on a pre-recorded video
├── camera_detection.py      # Real-time detection with webcam
├── yolov8n.pt               # YOLOv8 nano model
├── requirements.txt         # Python dependencies
├── README.md
└── .gitignore
```

##  Installation

### 1. Clone the repository

```bash
git clone https://github.com/mansour-asma22/ai-object-detection.git
cd ai-object-detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

##  Usage

###  Video Detection

Place your `.mp4` video in the project folder and configure the filename in `object_detection.py`.
```bash
python object_detection.py
```
### Real-Time Webcam Detection
Run:
```bash
python camera_detection.py
```

The webcam will open automatically and YOLOv8 will start detecting objects in real time.

###  Controls

| Key | Action            |
| --- | ----------------- |
| `Q` | Quit              |
| `S` | Save a screenshot |

The processed webcam video is automatically saved as:

```text
camera_output.mp4
```

##  What I Worked On

This project allowed me to work on:

* Using a pretrained YOLOv8 model for object detection
* Processing video frames with OpenCV
* Extracting object classes, coordinates and confidence scores
* Creating custom bounding boxes and labels
* Adapting video processing to a live webcam stream
* Measuring real-time processing performance with FPS
* Saving processed video and screenshots
* Handling camera and application shutdown correctly



