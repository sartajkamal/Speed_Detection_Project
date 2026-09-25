# Speed Detection & License Plate Extraction

An AI-based computer vision system developed to process road traffic footage, detect and track vehicles, estimate their speed, identify speeding violations, and extract license plate information.

The system converts raw traffic footage into structured violation records containing vehicle speed, license plate information, timestamp, camera/location details, and supporting evidence.

## 📌 Project Overview

Manual monitoring of traffic footage is time-consuming and can make it difficult to identify speeding vehicles and accurately record their license plates.

This project automates the process by analyzing road footage and identifying vehicles that exceed a configurable speed limit. For detected violations, the system extracts the license plate, applies image enhancement and OCR, and stores supporting evidence for review.

## ✨ Key Features

* 🚗 Vehicle detection and tracking
* ⚡ Vehicle speed estimation
* 🚨 Speed violation detection using a configurable speed limit
* 🔎 License plate extraction
* 📝 License plate recognition using EasyOCR
* 🖼️ License plate image enhancement
* 📊 Violation record generation
* 📁 Supporting evidence storage
* ⚙️ Configurable camera, location, speed limit, and OCR settings
* 🔍 Violation filtering and review support

## 🔄 System Workflow

```text
Road Traffic Video
        ↓
Vehicle Detection & Tracking
        ↓
Speed Estimation
        ↓
Speed Limit Comparison
        ↓
Speeding Vehicle Detected
        ↓
License Plate Extraction
        ↓
Image Enhancement
        ↓
OCR / Plate Recognition
        ↓
Violation Record
        ↓
Supporting Evidence
```

## 🛠️ Technologies Used

* **Python** — Core development
* **YOLOv8** — Vehicle detection
* **OpenCV** — Video processing and computer vision
* **EasyOCR** — License plate text recognition
* **NumPy** — Numerical and image processing
* **Pandas** — Data handling
* **Tkinter** — User interface
* **JSON / CSV** — Configuration and violation records
* **Celery** — Background processing/integration components

## 📁 Project Structure

```text
Speed_Detection_Project/
│
├── main.py
├── admin_config.json
├── yolov8n.pt
├── test_adapter.py
│
├── videos/
│   └── input traffic videos
│
├── models/
│
├── violations_evidence/
│   ├── crops/
│   ├── enhanced/
│   └── frames/
│
└── workers/
    ├── __init__.py
    ├── celery_app.py
    ├── media.py
    ├── pipeline_adapter.py
    └── tasks/
        └── process_footage.py
```

## ⚙️ Configuration

The system supports configurable settings such as:

* Speed limit
* Camera ID
* Camera location
* OCR confidence threshold
* Detection parameters
* Image processing settings

Example:

```json
{
    "speed_limit": 50.0,
    "camera_id": "CAM-01",
    "location": "Main Road Camera",
    "min_ocr_confidence": 0.20
}
```

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
cd Speed_Detection_Project
```

Create and activate a virtual environment:

```bash
python -m venv venv
```

On Windows:

```powershell
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## ▶️ Running the Project

Place the input traffic videos inside the `videos/` directory and run:

```bash
python main.py
```

The system will process the footage and perform vehicle detection, speed estimation, violation detection, license plate extraction, and OCR processing.

## 📊 Output

For detected speeding violations, the system can generate information including:

* License plate number
* Vehicle speed
* Speed limit
* Timestamp
* Camera ID
* Location
* OCR confidence
* Supporting evidence

Evidence is stored in:

```text
violations_evidence/
├── crops/
├── enhanced/
└── frames/
```

## 🔎 License Plate Processing

The system applies image-processing techniques to improve license plate readability before OCR.

Processing may include:

* Image upscaling
* CLAHE enhancement
* Sharpening
* OTSU thresholding
* Adaptive thresholding

EasyOCR is then used to recognize the characters from the processed plate image.

## 🧪 Testing

The project includes an adapter test for verifying integration with the processing pipeline.

```bash
python test_adapter.py
```

## ⚠️ Limitations

System performance depends on the quality of the input footage. Factors such as low resolution, motion blur, poor lighting, glare, camera angle, weather conditions, and partially visible plates can affect detection and OCR accuracy.

The system generates evidence and violation records for review. It does not issue fines, identify vehicle owners, or perform legal actions.

## 🎯 Project Goals

The main goals of the project are to:

1. Automate vehicle detection from traffic footage.
2. Estimate vehicle speed.
3. Identify vehicles exceeding the configured speed limit.
4. Extract and recognize license plates.
5. Generate evidence-backed violation records.
6. Reduce the need for continuous manual video monitoring.

## 🚀 Future Improvements

Possible future improvements include:

* Real-time traffic monitoring
* Live violation alerts
* Improved license plate recognition
* Better performance in low-light and difficult weather conditions
* Multi-lane vehicle tracking
* Live monitoring dashboard
* Bulk violation export
* Audit trail for review actions
* Plate anonymization for privacy-sensitive use cases

## 👩‍💻 About the Project

This project was developed as part of my **Machine Learning Internship at EmpowerBits**.

**Author:** Sartaj Kamal
**Role:** Machine Learning Intern
**Organization:** EmpowerBits
**Field:** Machine Learning / Computer Vision

## 📄 Documentation

The project requirements and functionality are documented in the **Speed Detection & License Plate Extraction Product Requirements Document (PRD)**.

---

```
Speed Detection & License Plate Extraction
Developed during Machine Learning Internship at EmpowerBits
```
