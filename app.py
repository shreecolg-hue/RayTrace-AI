import os
import shutil
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from predict_and_cut import load_trained_model, analyze_solar_panel

app = FastAPI(title="RayTrace AI - Solar Panel Inspection & Laser Cut Optimizer")

# Load ML model once at startup
model, class_names, device = load_trained_model("best_solar_classifier.pth")

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RayTrace AI | Photonics & Solar Recovery Studio</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-base: #060b14;
            --bg-surface: rgba(15, 23, 42, 0.78);
            --bg-elevated: rgba(30, 41, 59, 0.65);
            --border-subtle: rgba(148, 163, 184, 0.14);
            --border-accent: rgba(56, 189, 248, 0.4);
            --accent-cyan: #38bdf8;
            --accent-blue: #0284c7;
            --accent-emerald: #10b981;
            --accent-amber: #f59e0b;
            --accent-rose: #f43f5e;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
        }

        * { box-sizing: border-box; }

        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-base);
            background-image:
                radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.12), transparent 35%),
                radial-gradient(circle at 85% 90%, rgba(16, 185, 129, 0.08), transparent 35%),
                linear-gradient(rgba(148, 163, 184, 0.03) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148, 163, 184, 0.03) 1px, transparent 1px);
            background-size: 100% 100%, 100% 100%, 36px 36px, 36px 36px;
            color: var(--text-primary);
            margin: 0;
            padding: 32px 20px 64px;
            min-height: 100vh;
        }

        .container { max-width: 1220px; margin: 0 auto; }

        /* Top Navigation Header */
        .topbar {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            padding-bottom: 24px;
            border-bottom: 1px solid var(--border-subtle);
            margin-bottom: 28px;
        }
        .brand { display: flex; align-items: center; gap: 14px; }
        .brand-icon {
            width: 44px; height: 44px;
            border-radius: 12px;
            background: linear-gradient(135deg, #0284c7, #38bdf8);
            display: grid; place-items: center;
            box-shadow: 0 0 24px rgba(56, 189, 248, 0.35);
        }
        .brand h1 {
            margin: 0; font-size: 24px; font-weight: 700; letter-spacing: -0.02em;
            background: linear-gradient(to right, #ffffff, #7dd3fc);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .brand p { margin: 2px 0 0; font-size: 13px; color: var(--text-secondary); }

        .model-pills { display: flex; gap: 10px; flex-wrap: wrap; }
        .model-pill {
            display: flex; align-items: center; gap: 8px;
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            padding: 7px 13px; border-radius: 999px;
            font-size: 12px; color: var(--text-secondary);
            backdrop-filter: blur(12px);
        }
        .dot-live {
            width: 8px; height: 8px; border-radius: 50%;
            background: var(--accent-emerald);
            box-shadow: 0 0 8px var(--accent-emerald);
        }

        /* Drag & Drop Upload Zone */
        .upload-card {
            background: var(--bg-surface);
            border: 1.5px dashed var(--border-accent);
            border-radius: 16px;
            padding: 34px 24px;
            text-align: center;
            backdrop-filter: blur(16px);
            transition: all 0.25s ease;
            cursor: pointer;
            position: relative;
        }
        .upload-card:hover, .upload-card.dragover {
            border-color: var(--accent-cyan);
            background: rgba(15, 23, 42, 0.92);
            box-shadow: 0 12px 40px rgba(2, 132, 199, 0.16);
        }
        .upload-icon {
            width: 52px; height: 52px; margin: 0 auto 12px;
            border-radius: 14px; background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.25);
            display: grid; place-items: center; color: var(--accent-cyan);
        }
        .upload-title { font-size: 16px; font-weight: 600; margin: 0 0 6px; }
        .upload-hint { font-size: 13px; color: var(--text-secondary); margin: 0 0 18px; }
        #imageInput { display: none; }

        .file-preview-chip {
            display: none;
            align-items: center;
            justify-content: center;
            gap: 12px;
            margin: 0 auto 18px;
            background: rgba(30, 41, 59, 0.9);
            border: 1px solid var(--border-subtle);
            padding: 8px 14px;
            border-radius: 10px;
            width: fit-content;
        }
        .file-preview-chip img {
            width: 36px; height: 36px; object-fit: cover; border-radius: 6px;
        }
        .file-preview-chip span { font-size: 13px; color: var(--text-primary); font-weight: 500; }

        .btn-primary {
            background: linear-gradient(135deg, #0284c7, #0ea5e9);
            color: white; border: none;
            padding: 13px 28px; border-radius: 10px;
            font-size: 15px; font-weight: 600; cursor: pointer;
            box-shadow: 0 4px 20px rgba(14, 165, 233, 0.35);
            transition: transform 0.15s ease, box-shadow 0.2s ease;
            display: inline-flex; align-items: center; gap: 8px;
        }
        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 26px rgba(14, 165, 233, 0.5);
        }
        .btn-primary:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }

        /* Animated Laser Scan Progress Bar */
        .scanner-bar {
            display: none;
            margin-top: 20px;
            max-width: 480px;
            margin-left: auto; margin-right: auto;
        }
        .scanner-track {
            height: 4px; width: 100%; background: rgba(148, 163, 184, 0.15);
            border-radius: 999px; overflow: hidden;
        }
        .scanner-fill {
            height: 100%; width: 45%;
            background: linear-gradient(90deg, transparent, #38bdf8, #10b981, transparent);
            animation: scanMove 1.2s infinite ease-in-out;
        }
        @keyframes scanMove {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(250%); }
        }
        .scanner-text { font-size: 12.5px; color: var(--accent-cyan); margin-top: 8px; }

        /* Results Dashboard */
        .dashboard { display: none; margin-top: 30px; animation: fadeUp 0.4s ease; }
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        /* 4-Card KPI Grid */
        .kpi-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 22px;
        }
        .kpi-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 18px 20px;
            backdrop-filter: blur(12px);
        }
        .kpi-label {
            font-size: 11.5px; text-transform: uppercase; letter-spacing: 0.06em;
            color: var(--text-secondary); font-weight: 600; margin-bottom: 10px;
        }
        .kpi-value { font-size: 26px; font-weight: 700; letter-spacing: -0.02em; }
        .kpi-sub { font-size: 12.5px; color: var(--text-secondary); margin-top: 6px; }

        .badge {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px; border-radius: 999px;
            font-weight: 700; text-transform: uppercase; font-size: 13px; letter-spacing: 0.03em;
        }
        .badge.good { background: rgba(16, 185, 129, 0.16); color: #6ee7b7; border: 1px solid rgba(16, 185, 129, 0.35); }
        .badge.electric { background: rgba(245, 158, 11, 0.16); color: #fcd34d; border: 1px solid rgba(245, 158, 11, 0.35); }
        .badge.physical { background: rgba(244, 63, 94, 0.16); color: #fda4af; border: 1px solid rgba(244, 63, 94, 0.35); }

        .conf-bar-bg {
            height: 6px; background: rgba(148, 163, 184, 0.15);
            border-radius: 999px; margin-top: 10px; overflow: hidden;
        }
        .conf-bar-fill { height: 100%; background: var(--accent-cyan); border-radius: 999px; transition: width 0.5s ease; }

        /* 3-Tier Triage Routing Bar */
        .triage-strip {
            display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px;
            margin-bottom: 20px;
        }
        .triage-node {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 12px;
            padding: 12px 16px;
            opacity: 0.45;
            transition: all 0.25s ease;
        }
        .triage-node.active-good { opacity: 1; border-color: var(--accent-emerald); background: rgba(16, 185, 129, 0.08); }
        .triage-node.active-electric { opacity: 1; border-color: var(--accent-amber); background: rgba(245, 158, 11, 0.08); }
        .triage-node.active-physical { opacity: 1; border-color: var(--accent-rose); background: rgba(244, 63, 94, 0.08); }
        .triage-node strong { display: block; font-size: 13px; margin-bottom: 3px; }
        .triage-node span { font-size: 12px; color: var(--text-secondary); }

        /* Recovery Protocol Card */
        .protocol-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-left: 4px solid var(--accent-cyan);
            border-radius: 14px;
            padding: 20px 24px;
            margin-bottom: 22px;
        }
        .protocol-card h3 { margin: 0 0 12px; font-size: 16px; color: var(--text-primary); }
        .step-list { list-style: none; padding: 0; margin: 0; display: grid; gap: 10px; }
        .step-list li {
            display: flex; align-items: flex-start; gap: 10px;
            font-size: 14px; color: #e2e8f0; line-height: 1.5;
            background: rgba(15, 23, 42, 0.55);
            padding: 10px 14px; border-radius: 8px;
        }

        /* Side-by-Side Inspection Viewport */
        .viewport-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 22px;
        }
        @media (max-width: 820px) {
            .viewport-grid, .triage-strip { grid-template-columns: 1fr; }
        }
        .viewport-card {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            overflow: hidden;
        }
        .viewport-header {
            padding: 13px 18px;
            border-bottom: 1px solid var(--border-subtle);
            display: flex; justify-content: space-between; align-items: center;
            background: rgba(15, 23, 42, 0.6);
        }
        .viewport-header h4 { margin: 0; font-size: 14px; font-weight: 600; }
        .viewport-tag {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px; color: var(--accent-cyan);
            background: rgba(56, 189, 248, 0.1);
            padding: 3px 8px; border-radius: 6px;
        }
        .viewport-body {
            padding: 16px;
            display: flex; align-items: center; justify-content: center;
            min-height: 300px;
            background: #040810;
        }
        .viewport-body img {
            max-width: 100%; max-height: 440px;
            border-radius: 8px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.5);
        }

        /* CNC / Laser Waypoint Drawer */
        .coords-panel {
            background: var(--bg-surface);
            border: 1px solid var(--border-subtle);
            border-radius: 14px;
            padding: 16px 20px;
        }
        .coords-header {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;
        }
        .coords-header span { font-size: 13px; font-weight: 600; color: var(--text-secondary); }
        .btn-copy {
            background: var(--bg-elevated); color: var(--text-primary);
            border: 1px solid var(--border-subtle);
            padding: 6px 12px; border-radius: 7px; font-size: 12px; cursor: pointer;
        }
        .btn-copy:hover { border-color: var(--accent-cyan); color: var(--accent-cyan); }
        .coords-box {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12.5px; color: #7dd3fc;
            background: #040810; padding: 12px 14px;
            border-radius: 8px; border: 1px solid rgba(56, 189, 248, 0.15);
            max-height: 110px; overflow-y: auto; word-break: break-all;
        }
    </style>
</head>
<body>
<div class="container">
    <!-- Top Header -->
    <header class="topbar">
        <div class="brand">
            <div class="brand-icon">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#ffffff" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="3" width="18" height="18" rx="2"/>
                    <line x1="3" y1="9" x2="21" y2="9"/>
                    <line x1="3" y1="15" x2="21" y2="15"/>
                    <line x1="9" y1="3" x2="9" y2="21"/>
                    <line x1="15" y1="3" x2="15" y2="21"/>
                </svg>
            </div>
            <div>
                <h1>RayTrace AI</h1>
                <p>Dual-Model Solar Defect Triage &amp; Precision Laser Recovery Planner</p>
            </div>
        </div>
        <div class="model-pills">
            <div class="model-pill"><span class="dot-live"></span> Model 1: ResNet50 Vision</div>
            <div class="model-pill"><span class="dot-live"></span> Model 2: Random Forest Yield</div>
        </div>
    </header>

    <!-- Upload Zone -->
    <div class="upload-card" id="dropZone" onclick="document.getElementById('imageInput').click()">
        <input type="file" id="imageInput" accept="image/*" onchange="handleFileSelect(event)" />
        <div class="upload-icon">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
            </svg>
        </div>
        <p class="upload-title">Drop a solar module image here or click to browse</p>
        <p class="upload-hint">Supports electroluminescence (EL) or RGB panel inspection photos (JPG, PNG)</p>

        <div class="file-preview-chip" id="previewChip" onclick="event.stopPropagation()">
            <img id="thumbImg" src="" alt="Preview" />
            <span id="fileNameText">panel.jpg</span>
        </div>

        <button class="btn-primary" id="analyzeBtn" onclick="event.stopPropagation(); analyzeImage();">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>
            </svg>
            Analyze Panel &amp; Optimize Recovery
        </button>

        <div class="scanner-bar" id="loading">
            <div class="scanner-track"><div class="scanner-fill"></div></div>
            <div class="scanner-text">Auto-cropping module, running ResNet50 triage &amp; optimizing laser trajectory...</div>
        </div>
    </div>

    <!-- Results Dashboard -->
    <main class="dashboard" id="resultCard">
        <!-- 4 KPI Cards -->
        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-label">Model 1 &bull; Panel Condition</div>
                <span id="classBadge" class="badge good">GOOD CONDITION</span>
                <div class="conf-bar-bg"><div id="confBar" class="conf-bar-fill" style="width: 0%"></div></div>
                <div class="kpi-sub" id="confidenceText">Confidence: --%</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-label">Model 2 &bull; Predicted Recovery Yield</div>
                <div class="kpi-value" id="yieldValue" style="color: #4ade80;">--%</div>
                <div class="kpi-sub" id="defectSub">Estimated Defect Ratio: --%</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-label">Laser Trajectory Path</div>
                <div class="kpi-value" id="cutLengthValue" style="color: #38bdf8;">-- mm</div>
                <div class="kpi-sub" id="segmentsSub">0 Cut Segments</div>
            </div>

            <div class="kpi-card">
                <div class="kpi-label">Laser Beam Power</div>
                <div class="kpi-value" id="laserPowerValue">-- W</div>
                <div class="kpi-sub" id="routingSub">Active Photonics Mode</div>
            </div>
        </div>

        <!-- 3-Stage Triage Pipeline Indicator -->
        <div class="triage-strip">
            <div class="triage-node" id="nodeGood">
                <strong>Path A: Full Retention</strong>
                <span>Clean module &rarr; Full laser cut (Glass + Silicon + Metals)</span>
            </div>
            <div class="triage-node" id="nodeElectric">
                <strong>Path B: Selective Glass Retention</strong>
                <span>Electric fault &rarr; Perimeter glass cut only; damaged cells excluded</span>
            </div>
            <div class="triage-node" id="nodePhysical">
                <strong>Path C: Thermal &amp; Crushing</strong>
                <span>Physical damage &rarr; Skip laser; heat EVA &amp; crush glass cullet</span>
            </div>
        </div>

        <!-- Actionable Recovery Protocol -->
        <div class="protocol-card" id="strategyBox">
            <h3 id="strategyTitle">Recovery Protocol</h3>
            <ul class="step-list" id="stepsList"></ul>
        </div>

        <!-- Side-by-Side Image Viewports -->
        <div class="viewport-grid">
            <div class="viewport-card">
                <div class="viewport-header">
                    <h4>Input Solar Module</h4>
                    <span class="viewport-tag">RAW / CROPPED INPUT</span>
                </div>
                <div class="viewport-body">
                    <img id="origImg" src="" alt="Uploaded Solar Panel" />
                </div>
            </div>

            <div class="viewport-card">
                <div class="viewport-header">
                    <h4 id="outputImgTitle">Optimized Laser Cut Trajectory</h4>
                    <span class="viewport-tag" id="outputTag">DP TRAJECTORY</span>
                </div>
                <div class="viewport-body">
                    <img id="cutImg" src="" alt="Laser Trajectory Output" />
                </div>
            </div>
        </div>

        <!-- CNC Laser Waypoints Drawer -->
        <div class="coords-panel">
            <div class="coords-header">
                <span>CNC Laser Controller Waypoints (X, Y Coordinates)</span>
                <button class="btn-copy" id="copyBtn" onclick="copyWaypoints()">Copy JSON</button>
            </div>
            <div class="coords-box" id="coordsBox">[]</div>
        </div>
    </main>
</div>

<script>
let latestWaypointsJson = "[]";

// Drag & Drop Handlers
const dropZone = document.getElementById('dropZone');
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
});
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        const input = document.getElementById('imageInput');
        input.files = e.dataTransfer.files;
        handleFileSelect({ target: input });
    }
});

function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;
    document.getElementById('fileNameText').textContent = file.name;
    document.getElementById('thumbImg').src = URL.createObjectURL(file);
    document.getElementById('previewChip').style.display = 'inline-flex';
}

async function analyzeImage() {
    const fileInput = document.getElementById('imageInput');
    if (!fileInput.files.length) {
        alert("Please select or drop a solar panel image first.");
        return;
    }

    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    document.getElementById('loading').style.display = 'block';

    try {
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        const res = await fetch("/api/analyze", { method: "POST", body: formData });
        const data = await res.json();

        // Reset triage nodes
        document.getElementById('nodeGood').className = 'triage-node';
        document.getElementById('nodeElectric').className = 'triage-node';
        document.getElementById('nodePhysical').className = 'triage-node';

        const badge = document.getElementById('classBadge');
        const strategyBox = document.getElementById('strategyBox');
        badge.textContent = data.classification;

        if (data.classification === "good condition") {
            badge.className = "badge good";
            strategyBox.style.borderLeftColor = "#10b981";
            document.getElementById('nodeGood').classList.add('active-good');
        } else if (data.classification === "electric faulty") {
            badge.className = "badge electric";
            strategyBox.style.borderLeftColor = "#f59e0b";
            document.getElementById('nodeElectric').classList.add('active-electric');
        } else {
            badge.className = "badge physical";
            strategyBox.style.borderLeftColor = "#f43f5e";
            document.getElementById('nodePhysical').classList.add('active-physical');
        }

        // Populate KPI cards
        document.getElementById('confidenceText').textContent = `Model Confidence: ${data.confidence_percent}%`;
        document.getElementById('confBar').style.width = `${Math.min(100, data.confidence_percent)}%`;
        document.getElementById('yieldValue').textContent = `${data.predicted_recovery_yield_pct}%`;

        const summary = data.cut_summary;
        document.getElementById('defectSub').textContent = `Estimated Defect Area: ${(summary.defect_area_ratio * 100).toFixed(1)}%`;
        document.getElementById('cutLengthValue').textContent = `${summary.total_cut_length_mm} mm`;
        document.getElementById('segmentsSub').textContent = `${summary.num_cut_segments} Laser Segments`;
        document.getElementById('laserPowerValue').textContent = `${summary.laser_power_w} W`;
        document.getElementById('routingSub').textContent = summary.laser_power_w > 0 ? "Laser Gantry Engaged" : "Laser Bypassed";

        // Populate Protocol Steps
        document.getElementById('strategyTitle').textContent = data.recovery_strategy;
        const stepsList = document.getElementById('stepsList');
        stepsList.innerHTML = "";
        data.process_steps.forEach(step => {
            const li = document.createElement("li");
            li.textContent = step;
            stepsList.appendChild(li);
        });

        // Populate Images & Waypoints
        const coordsBox = document.getElementById('coordsBox');
        latestWaypointsJson = JSON.stringify(data.laser_trajectory_points);

        if (data.laser_trajectory_points && data.laser_trajectory_points.length > 0) {
            coordsBox.textContent = latestWaypointsJson;
            document.getElementById('outputImgTitle').textContent =
                data.classification === "good condition"
                    ? "Full Laser Cut Trajectory (Glass + Silicon + Metals)"
                    : "Perimeter Glass-Only Cut (Damaged Cells Excluded)";
            document.getElementById('outputTag').textContent =
                data.classification === "good condition" ? "FULL RETENTION CUT" : "GLASS-ONLY PERIMETER";
        } else {
            coordsBox.textContent = "[]  // Laser bypassed: module routed to thermal heating & mechanical glass crushing";
            document.getElementById('outputImgTitle').textContent = "Thermal & Crushing Triage Overlay";
            document.getElementById('outputTag').textContent = "THERMAL + CRUSHING";
        }

        document.getElementById('origImg').src = data.original_image_b64;
        document.getElementById('cutImg').src = data.cut_image_b64;
        document.getElementById('resultCard').style.display = 'block';
        document.getElementById('resultCard').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } finally {
        btn.disabled = false;
        document.getElementById('loading').style.display = 'none';
    }
}

function copyWaypoints() {
    navigator.clipboard.writeText(latestWaypointsJson);
    const btn = document.getElementById('copyBtn');
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = "Copy JSON"; }, 1800);
}
</script>
</body>
</html>
"""

def to_base64(path):
    with open(path, "rb") as f:
        return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")

@app.get("/", response_class=HTMLResponse)
def home():
    return HTML_PAGE

@app.post("/api/analyze")
async def analyze_endpoint(file: UploadFile = File(...)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    result = analyze_solar_panel(temp_path, model, class_names, device)

    return {
        "classification": result["classification"],
        "confidence_percent": result["confidence_percent"],
        "recovery_strategy": result["recovery_strategy"],
        "process_steps": result["process_steps"],
        "predicted_recovery_yield_pct": result["predicted_recovery_yield_pct"],
        "cut_summary": result["cut_summary"],
        "laser_trajectory_points": result["laser_trajectory_points"],
        "original_image_b64": to_base64(temp_path),
        "cut_image_b64": to_base64(result["output_image"])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)