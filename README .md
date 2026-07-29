# 🦺 AI PPE Compliance System

An AI-powered computer vision tool that automatically detects Personal Protective Equipment (PPE) compliance on construction and industrial sites. Upload a photo, and the system identifies each worker, checks for required safety gear, and flags violations in real time.

![Status](https://img.shields.io/badge/status-active-success)
![Python](https://img.shields.io/badge/python-3.x-blue)
![Model](https://img.shields.io/badge/model-YOLOv8%20Nano-orange)
![Framework](https://img.shields.io/badge/UI-Streamlit-red)

🔗 **[Live Demo](https://ai-ppe-compliance-system-xk7nrd9yjtrkq9kxptn9tz.streamlit.app/)** &nbsp;|&nbsp; 💻 **[GitHub Repository](https://github.com/fizanoor-tech/AI-PPE-)**

---

## 📸 Screenshots

### Home Page
Upload an image directly from the browser — no setup required on the user's end.

![Home Page](<img width="1268" height="743" alt="Screenshot 2026-07-29 184232" src="https://github.com/user-attachments/assets/3c99724d-159d-4974-b315-a4ef41988cec" />
)

### Uploaded Image
The system accepts a worksite photo and prepares it for analysis.

![Uploaded Image](<img width="1077" height="833" alt="Screenshot 2026-07-29 184320" src="https://github.com/user-attachments/assets/26f7c2b6-3136-4628-b736-1662e9d83a1e" />
)

### Detection Result
Each worker is detected and boxed individually — **green** for compliant, **red** for a violation — with the specific missing item labeled directly on the image.

![Detection Result](<img width="751" height="733" alt="Screenshot 2026-07-29 184405" src="https://github.com/user-attachments/assets/cca9f6cd-b1f3-4c20-a419-fa9fca7a693d" />
)

### Compliance Results
A structured, per-worker breakdown of compliance status, including exactly which PPE item is missing, plus a downloadable annotated image.

![Compliance Results](<img width="1073" height="747" alt="Screenshot 2026-07-29 184428" src="https://github.com/user-attachments/assets/46e1d644-1092-4487-89b6-d82ccac234fd" />
)

---

## 🚀 Overview

Manual PPE compliance checks on job sites are slow, inconsistent, and easy to miss. The **AI PPE Compliance System** solves this by using a trained object detection model to automatically:

- Detect every worker present in an image
- Identify the PPE items each worker is or isn't wearing (e.g. safety vest)
- Flag violations with clear visual annotations
- Log results in a structured, exportable format for audits and reporting

The tool is deployed as an interactive **Streamlit** web app, so it can be used by site supervisors with zero technical setup.

---

## ✨ Features

- 🖼️ **Simple upload interface** — drag and drop a JPG/PNG image
- 🧠 **YOLOv8-based detection** — fast, accurate worker and PPE identification
- 🟩🟥 **Color-coded bounding boxes** — instantly see who's compliant and who isn't
- 📋 **Per-worker compliance breakdown** — exact missing items listed for each person
- 📊 **CSV logging** — every detection is logged with timestamp, worker ID, and status for record-keeping
- 📁 **Automatic sorting** — images are routed into `alerts_output/` or `compliant_output/` based on detected violations
- ⬇️ **Downloadable annotated images** — export results for reports or documentation

---

## 🛠️ Tech Stack

| Component        | Technology              |
|-------------------|--------------------------|
| Detection Model   | YOLOv8 Nano (Ultralytics) |
| Image Handling    | OpenCV, PIL/Pillow        |
| Data Logging      | Python `csv` module       |
| Web Interface     | Streamlit                 |
| Language          | Python                    |

---

## 📊 Model Performance

Validated on **143 images / 945 labeled instances**.

| Class     | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
|-----------|-----------|-----------|--------|--------|-----------|
| **Overall** | 945     | 0.860     | 0.778  | 0.833  | 0.446     |
| Person    | 239       | 0.828     | 0.883  | 0.888  | 0.505     |
| Vest      | 171       | 0.836     | 0.801  | 0.849  | 0.519     |
| Helmet    | 201       | 0.871     | 0.811  | 0.845  | 0.447     |
| Boots     | 151       | 0.842     | 0.707  | 0.799  | 0.452     |
| Goggles   | 47        | 0.919     | 0.745  | 0.809  | 0.371     |
| Gloves    | 136       | 0.864     | 0.721  | 0.811  | 0.381     |

⚡ **Inference speed:** ~2.0 ms/image (~500 FPS capability) — fast enough for real-time deployment.

---

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/ai-ppe-compliance-system.git
cd ai-ppe-compliance-system

# Create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## ▶️ Usage

Run the Streamlit app locally:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`), upload a worksite image, and click **Analyze Image** to view compliance results.

---

## 📂 System Outputs

For every processed image, the system generates:

- **Annotated image** with color-coded bounding boxes (green = compliant, red = violation) and per-worker missing-item labels
- **Structured CSV log entry** per worker — timestamp, filename, worker ID, status, missing PPE items, total workers in image, and violation count
- **Auto-sorted image files** placed into `alerts_output/` or `compliant_output/`, based on the computed violation count derived directly from detection data (not inferred from rendered image text)

---

## 📁 Project Structure

```
ai-ppe-compliance-system/
├── app.py                          # Streamlit application entry point
├── compliance.py                   # Compliance-checking logic
├── predict.py                      # Inference / prediction script
├── train.py                        # Model training script
├── best.pt                         # Trained YOLOv8 model weights
├── images/
│   ├── home_page.png                # README screenshot - upload page
│   ├── uploaded_image.png           # README screenshot - uploaded image
│   ├── detection_result.png         # README screenshot - detection boxes
│   ├── compliance_results.png       # README screenshot - compliance breakdown
│   ├── demo_screenshot.png          # App demo screenshot
│   ├── confusion_matrix.png         # Model evaluation - confusion matrix
│   └── result.png                   # Model evaluation - results
├── source_images/
│   ├── alert_images/               # Sample images with violations (5)
│   └── compliance_images/          # Sample fully compliant images
├── failure_mode_analysis.pdf       # Detailed failure mode analysis report
├── requirements.txt
└── README.md
```

---

## ⚠️ Known Limitations & Roadmap

Based on internal model evaluation, the following failure modes have been identified and prioritized:

| Failure Mode              | Domain                    | Status                          | Priority   |
|----------------------------|----------------------------|----------------------------------|------------|
| Frame-edge clipping        | Image boundary handling    | ✅ Fixed                        | Resolved   |
| EXIF rotation (no metadata)| Input data availability    | 🟡 Partial                      | Documented, scoped for v2 |
| Crouching / unusual posture| Spatial heuristic logic    | 📝 Documented                   | High (v2)  |
| Text label clipping        | Rendering / UI             | 🔍 Identified                   | Low        |

> These edge cases affect the observed real-world compliance-rate figure in isolated scenarios but do **not** affect the core mAP@50 validation metric, which is computed on a separate, clean validation split.

---

## 🗺️ Roadmap

- [ ] Improve posture-invariant detection (crouching/bending workers)
- [ ] Add EXIF-based auto image rotation
- [ ] Expand PPE classes (helmet, gloves, boots, goggles)
- [ ] Batch image / video stream processing
- [ ] Dashboard analytics for historical compliance trends

---

## 📄 License

This project is licensed under the MIT License — feel free to use, modify, and distribute with attribution.

---

## 🙋 Contact

For questions, feedback, or collaboration inquiries, feel free to open an issue or reach out directly.
