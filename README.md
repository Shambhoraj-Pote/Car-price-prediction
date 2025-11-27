## Car Price AI

Real-time, image-based car price estimation with a FastAPI backend (ResNet18 embeddings + lightweight regressor head) and a professional single-page UI that supports webcam capture, image uploads, history, and sharing.

### Features
- Real-time webcam capture and one-click "Capture & Predict"
- Drag-and-drop/file upload support
- Clean UI with Tailwind, dark-mode friendly
- Client-side adjustment sliders (mileage, year, condition) to contextualize estimates
- Prediction history and quick sharing

### Quick Start
1. Install Python 3.10+ and Node is not required (UI is static).
2. Create a virtual environment (recommended).
3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run the server:

```bash
python run.py
```

5. Open the app at:

```
http://localhost:8000/
```

If your browser blocks camera access, allow it when prompted.

### How it works (demo)
- The backend uses a pretrained ResNet18 (ImageNet) to extract visual embeddings from the input image.
- A small neural regressor head converts embeddings into a price estimate, clamped to a reasonable range.
- Confidence is a heuristic derived from embedding statistics (demo only).

### Make it production-grade
- Replace the demo head with a trained regressor using a labeled dataset of (image, price) pairs.
- Consider multi-task learning: predict make/model/year/trim first, then feed those as features to price regression.
- Add EXIF/metadata parsing, multi-angle aggregation, background/occlusion checks.
- Integrate real market data for calibration and confidence intervals.

### Train on an OLX-style dataset (INR)
Prepare a CSV with columns: `image_path,price_inr` where `image_path` is relative to an images root folder, and `price_inr` is the target price in Indian Rupees. Example:

```csv
image_path,price_inr
sedans/honda_city_2019_front.jpg,725000
sedans/honda_city_2019_side.jpg,720000
```

Run training (requires PyTorch + torchvision):

```bash
python scripts/train_olx.py --csv data/olx.csv --images data/images --out models --epochs 8 --lr 3e-4 --batch-size 32
```

This freezes ResNet18, trains the small price head, and saves `models/olx_price_head.pt`. The backend will auto-load it on startup. When present, predictions are already in INR and the UI displays `₹`.

If no trained head is found, the app falls back to an embedding-based demo (converted to INR using a fixed rate) or a torchless statistics model.

### Training Hook (outline)
Create a script that:
1. Loads a dataset of car images with target prices.
2. Uses the same preprocessing as `backend/utils.py`.
3. Freezes the backbone (`feature_extractor`), trains the `PriceHead` on embeddings.
4. Saves trained weights and loads them in `CarPriceModel` at startup.

### Windows notes
- If `torch` install is slow on Windows, consider using a prebuilt wheel via PyTorch website guidance.

### Disclaimer
This repository provides a working demo pipeline and UI. The demo predictions are not accurate for real-world pricing without proper training data and calibration.


