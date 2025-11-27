from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Any
import io
from PIL import Image, UnidentifiedImageError

from .model import CarPriceModel
from .utils import image_to_tensor


app = FastAPI(title="AI Car Price Prediction", version="0.1.0")

app.add_middleware(
	CORSMiddleware,
	allow_origins=["*"],
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Serve static UI
app.mount("/static", StaticFiles(directory="static"), name="static")

model = CarPriceModel()


@app.get("/health")
def health() -> Dict[str, str]:
	return {"status": "ok"}


@app.post("/predict")
async def predict(
	image: UploadFile = File(...),
	mileage_km: float = Form(...),
	year: int = Form(...),
	condition: int = Form(...),
	engine_cc: int = Form(...),
	seats: int = Form(...),
	brand: str = Form(...),
) -> Dict[str, Any]:
	try:
		raw = await image.read()
		img = Image.open(io.BytesIO(raw)).convert("RGB")
	except UnidentifiedImageError:
		raise HTTPException(status_code=400, detail="Invalid image file")
	except Exception as exc:
		raise HTTPException(status_code=400, detail=f"Failed to read image: {exc}")

	# Basic validation
	if mileage_km < 0:
		raise HTTPException(status_code=400, detail="Mileage must be >= 0")
	if year < 1990 or year > 2026:
		raise HTTPException(status_code=400, detail="Year must be between 1990 and 2026")
	if condition < 1 or condition > 5:
		raise HTTPException(status_code=400, detail="Condition must be between 1 and 5")
	if engine_cc < 600 or engine_cc > 8000:
		raise HTTPException(status_code=400, detail="Engine size (cc) must be between 600 and 8000")
	if seats < 2 or seats > 10:
		raise HTTPException(status_code=400, detail="Seats must be between 2 and 10")
	if not brand or not brand.strip():
		raise HTTPException(status_code=400, detail="Brand is required")

	tensor = image_to_tensor(img)
	meta = {
		"mileage_km": mileage_km,
		"year": year,
		"condition": condition,
		"engine_cc": engine_cc,
		"seats": seats,
		"brand": brand.strip(),
	}
	result = model.predict_with_metadata(tensor, meta)
	return result


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
	with open("static/index.html", "r", encoding="utf-8") as f:
		return HTMLResponse(f.read())


