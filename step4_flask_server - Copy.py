# ============================================================
# STEP 4 — Flask Server (ESP32 talks to this)
# ============================================================
# What this script does:
#   Starts a small web server on your PC at port 5000.
#   The ESP32 sends PPG features as JSON over WiFi.
#   The server runs the ML model and sends back the
#   glucose prediction to the ESP32.
#
#          ESP32  ──WiFi──►  This server  ──►  OLED
#
# How to run:
#   python step4_flask_server.py
#
# Keep this running while using the ESP32.
# Press Ctrl+C to stop the server.
# ============================================================

import joblib
import json
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# ── Load model on startup ─────────────────────────────────────
print("=" * 50)
print("  STEP 4 — Glucose Prediction Flask Server")
print("=" * 50)

print("\n  Loading ML model...")
model  = joblib.load('models/best_model.pkl')
scaler = joblib.load('models/scaler.pkl')

with open('models/features.json') as f:
    FEATURES = json.load(f)

print(f"  Model  : {type(model).__name__}")
print(f"  Features ({len(FEATURES)}): {FEATURES}")

# ── Helper: glucose category ──────────────────────────────────
def categorise(g):
    if g < 70:  return 'Low'
    if g < 100: return 'Normal'
    if g < 126: return 'Pre-Diabetic'
    if g < 200: return 'High'
    return 'Very High'

# ── Route 1: Health check ─────────────────────────────────────
# ESP32 or browser can call this to verify server is running
# URL: http://YOUR_PC_IP:5000/health
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'running',
        'model':   type(model).__name__,
        'message': 'Glucose server is ready'
    })

# ── Route 2: Predict glucose ──────────────────────────────────
# ESP32 sends JSON with feature values → gets glucose back
# URL: http://YOUR_PC_IP:5000/predict  (POST)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from ESP32
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data received'}), 400

        # Log what ESP32 sent
        print(f"\n  [REQUEST] From {request.remote_addr}")
        print(f"  Features received:")
        for feat in FEATURES:
            val = data.get(feat, 'MISSING')
            print(f"    {feat:<20} : {val}")

        # Check all features are present
        missing = [f for f in FEATURES if f not in data]
        if missing:
            return jsonify({'error': f'Missing features: {missing}'}), 400

        # Build feature array in correct order
        x = np.array([[data[f] for f in FEATURES]])

        # Scale features (same way as training)
        x_scaled = scaler.transform(x)

        # Run ML prediction
        glucose = float(model.predict(x_scaled)[0])

        # Clamp to physiological range
        glucose = max(40.0, min(450.0, glucose))

        # Calculate confidence from signal quality
        sq         = float(data.get('signal_quality', 70))
        confidence = round(min(92.0, sq * 0.88), 1)

        # Build response
        result = {
            'glucose_mgdl': round(glucose, 1),
            'category':     categorise(glucose),
            'confidence':   confidence,
            'status':       'ok'
        }

        # Log result
        print(f"\n  [RESULT]  {result['glucose_mgdl']} mg/dL")
        print(f"            Category   : {result['category']}")
        print(f"            Confidence : {result['confidence']}%")

        return jsonify(result)

    except Exception as e:
        print(f"\n  [ERROR] {str(e)}")
        return jsonify({'error': str(e)}), 500

# ── Route 3: Manual test from browser ────────────────────────
# Open this in browser to test without ESP32
# URL: http://YOUR_PC_IP:5000/test
@app.route('/test', methods=['GET'])
def test():
    # Simulate a typical Normal glucose reading
    sample = {
        'ir_mean':         120000,
        'ir_ac':           2400,
        'red_mean':        85000,
        'red_ac':          1550,
        'ratio':           0.87,
        'dc_ratio':        0.71,
        'perfusion_index': 2.0,
        'normalized_ir':   3.1,
        'heart_rate':      74,
        'signal_quality':  85,
    }

    x        = np.array([[sample[f] for f in FEATURES]])
    x_scaled = scaler.transform(x)
    glucose  = float(model.predict(x_scaled)[0])
    glucose  = max(40.0, min(450.0, glucose))

    result = {
        'test_input':     sample,
        'glucose_mgdl':   round(glucose, 1),
        'category':       categorise(glucose),
        'confidence':     85.0,
        'note':           'This is a test with dummy values'
    }

    print(f"\n  [TEST]  Browser test hit → {glucose:.1f} mg/dL")
    return jsonify(result)

# ── Start server ──────────────────────────────────────────────
if __name__ == '__main__':
    import socket

    # Get local IP address to show user
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
    except:
        local_ip = "127.0.0.1"

    print(f"\n  {'─' * 48}")
    print(f"  Server starting...")
    print(f"  {'─' * 48}")
    print(f"  Your PC IP address : {local_ip}")
    print(f"  {'─' * 48}")
    print(f"  Endpoints:")
    print(f"    Health check : http://{local_ip}:5000/health")
    print(f"    Browser test : http://{local_ip}:5000/test")
    print(f"    ESP32 sends  : http://{local_ip}:5000/predict  (POST)")
    print(f"  {'─' * 48}")
    print(f"\n  ⚠️  Copy this URL for your ESP32 firmware:")
    print(f"  http://{local_ip}:5000/predict")
    print(f"\n  Press Ctrl+C to stop the server.")
    print(f"  {'─' * 48}\n")

    # Run on all network interfaces so ESP32 can reach it
    app.run(host='0.0.0.0', port=5000, debug=False)
    