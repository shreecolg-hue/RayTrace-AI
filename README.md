# RayTrace AI — Dual-Model Solar Module Defect Triage & Laser Recovery Optimizer

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-ResNet50%20(96.89%25)-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Web%20Studio-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4--Point%20Crop-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-emerald.svg)](#license)

**RayTrace AI** is an automated computer-vision and photonics-routing system built to replace unprofitable, one-size-fits-all solar PV recycling lines with **condition-aware dynamic triage**. 

Conventional mechanical and thermal solar recycling plants in India spend **₹883–₹1,079 per module** crushing end-of-life panels into low-grade glass cullet at a net unit loss (`-₹189` to `-₹273/module`). RayTrace AI inspects incoming solar modules, classifies defect severity with **96.89% validation accuracy**, computes optimal CNC laser trajectories via Dynamic Programming, and forecasts net material recovery yield in Indian Rupees (`₹`).

---

## Key Features & Three-Tier Triage Workflow

Rather than subjecting every panel to destructive 500°C pyrolysis and mechanical shredding, RayTrace AI routes each module through one of three precision recovery pathways:

| Triage Path | Detected Condition | Laser Gantry Action | Material Recovery Outcome | Unit Economics (`₹ / Module`) |
| :--- | :--- | :--- | :--- | :--- |
| **Path A** | **Clean / Good Condition** | **Full Matrix Laser Cut** (`45.0 W`) — Perimeter + cell-string separation | Retains **100% intact tempered glass**, whole silicon wafers, Ag/Cu busbars, and Al frame | **+₹1,170 net profit** (`₹1,550` yield vs. `₹380` OPEX) |
| **Path B** | **Electrically Faulty** | **Perimeter-Only Laser Cut** (`45.0 W`) — Outer glass delamination only | Saves the **intact front glass sheet** (`₹520`) while isolating damaged semiconductors | **+₹760 net profit** (`₹1,070` yield vs. `₹310` OPEX) |
| **Path C** | **Physical Damage / Hybrid Defect** | **Laser Bypassed** (`0.0 W`) — Routed straight to EVA heating & crushing | Prevents wasted laser cycles on shattered glass; recovers cullet & metals cleanly | **-₹240 net** (`₹540` yield vs. `₹780` OPEX — saves `₹103/panel` vs. static) |

> **Hierarchical Defect Precedence Rule (Hybrid Faults):**  
> If two types of defects—both **electrical and physical**—are present on a solar panel simultaneously, the system automatically proceeds with the **Physical Damage pathway (Path C)** because the front tempered glass cannot be retrieved intact.

---

## System Architecture

RayTrace AI operates as a four-stage inference and optimization pipeline:

```text
[Raw Module Image]
       │
       ▼
┌──────────────────────────────────────────────────────────┐
│ Stage 0: OpenCV 4-Point Perspective Auto-Cropper         │
│ • Canny edge detection + morphological closing           │
│ • Strips background clutter & rectifies tilted panels    │
└───────────────────────────┬──────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────┐
│ Model 1: PyTorch ResNet50 Defect Triage Classifier       │
│ • 96.89% validation accuracy across 3 defect classes     │
│ • Enforces Physical > Electrical precedence rule         │
└───────────────────────────┬──────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌───────────────────────────┐ ┌────────────────────────────┐
│ Stage 2: DP Laser Planner │ │ Model 2: Yield Regressor   │
│ • Full matrix (Path A)    │ │ • RandomForestRegressor    │
│ • Glass perimeter (Path B)│ │ • Inputs: defect ratio,    │
│ • 0W Bypass (Path C)      │ │   cut length, segments, W  │
└─────────────┬─────────────┘ └─────────────┬──────────────┘
              │                             │
              └─────────────┬───────────────┘
                            ▼
        [FastAPI Web Studio & CNC JSON Waypoints]
```

---

## Repository Structure

```text
RayTrace-AI/
├── app.py                        # FastAPI backend & Photonics Web Dashboard UI
├── predict_and_cut.py            # Auto-cropping, ResNet50 inference & 3-tier routing logic
├── glass_cut_optimizer.py        # Dynamic Programming laser trajectory generator
├── train_recovery_model.py       # RandomForestRegressor recovery yield predictor
├── best_solar_classifier.pth     # Trained PyTorch ResNet50 model weights
├── recovery_yield_model.joblib   # Trained Random Forest yield model artifact
├── index.html                    # Interactive Portfolio & INR Unit Economics Showcase
└── temp_uploads/                 # Runtime directory for uploaded panel images
```

---

## Quickstart & Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/RayTrace-AI.git
cd RayTrace-AI
```

### 2. Install Dependencies
Ensure you are using Python 3.9+ (with PyTorch installed for your CPU or CUDA GPU):
```bash
python -m pip install fastapi uvicorn python-multipart torch torchvision pillow opencv-python numpy scikit-learn joblib
```

### 3. Launch the FastAPI Inspection Studio
```bash
python -m uvicorn app:app --reload
```
Open your browser and navigate to **`http://127.0.0.1:8000`** to upload a solar panel image, view the auto-cropped inspection overlay, inspect the computed CNC laser waypoints `(X, Y)`, and see the predicted recovery yield.

### 4. View the Interactive Portfolio & INR Simulator
Open `index.html` directly in any modern browser to explore:
- **Interactive Static vs. Dynamic Recycling Simulator** (`100` to `10,000` modules)
- **Live SVG Laser Trajectory Visualizer** across Path A, Path B, and Path C
- **Indian PV Unit Economics Breakdown (`₹ / Module`)** benchmarked against CEEW / CPCB baselines

---

## Unit Economics Benchmark (`₹ / 20 kg Module`)

Grounded in Indian solar PV recycling benchmarks (CEEW / CPCB baselines for standard 60/72-cell modules):

- **Conventional Mechanical Line:** `₹883` OPEX vs. `₹610` cullet/scrap yield $\rightarrow$ **`-₹273 / module` loss**
- **Conventional Chemical / Pyrolysis Line:** `₹1,079` OPEX vs. `₹890` material yield $\rightarrow$ **`-₹189 / module` loss**
- **RayTrace AI (Path A — Clean Module):** `₹380` OPEX vs. `₹1,550` intact glass + wafers + Ag/Cu/Al $\rightarrow$ **`+₹1,170 / module` net profit**
- **RayTrace AI (Path B — Electric Faulty):** `₹310` OPEX vs. `₹1,070` intact glass + frame/metals $\rightarrow$ **`+₹760 / module` net profit**

---

## Project Team

| Name | USN |
| :--- | :--- |
| **Shree Hashwanth** | `1DS25CS249` |
| **Sanjana M Kurandawad** | `1DS25CS234` |
| **Tanushree S** | `1DS25CS294` |

---

## License

This project is open-source and available under the [MIT License](LICENSE).
