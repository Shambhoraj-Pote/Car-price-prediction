from typing import Any, Dict
import math
import numpy as np
import torch
import torch.nn as nn
import os
try:
	import torchvision.models as models  # type: ignore
	_HAS_TORCHVISION = True
except Exception:
	models = None  # type: ignore
	_HAS_TORCHVISION = False


class PriceHead(nn.Module):
	def __init__(self, in_features: int) -> None:
		super().__init__()
		self.regressor = nn.Sequential(
			nn.Linear(in_features, 256),
			nn.ReLU(inplace=True),
			nn.Dropout(0.2),
			nn.Linear(256, 64),
			nn.ReLU(inplace=True),
			nn.Linear(64, 1),
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.regressor(x)


class CarPriceModel:
	def __init__(self, device: str | None = None) -> None:
		self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
		self.torchvision_enabled = _HAS_TORCHVISION
		self.currency = "INR"
		self.usd_to_inr = 83.0  # fallback conversion if only USD head is available
		self.brand_weights = {
			# Example brand priors (can be refined with data)
			"maruti": 0.95,
			"hyundai": 1.0,
			"tata": 0.95,
			"mahindra": 1.0,
			"honda": 1.05,
			"toyota": 1.12,
			"skoda": 1.08,
			"volkswagen": 1.06,
			"bmw": 1.35,
			"mercedes": 1.4,
			"audi": 1.32,
			"kia": 1.03,
			"mg": 1.02,
			"nissan": 0.98,
			"renault": 0.96,
			"jeep": 1.12,
			"land rover": 1.45,
		}
		if self.torchvision_enabled:
			# Try to load pretrained weights; if offline, fall back to random init
			try:
				base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # type: ignore
			except Exception:
				base = models.resnet18(weights=None)  # type: ignore
			self.feature_extractor = nn.Sequential(*list(base.children())[:-1]).to(self.device).eval()
			for p in self.feature_extractor.parameters():
				p.requires_grad = False
			self.head = PriceHead(in_features=512).to(self.device).eval()
			# Initialize head with small weights to produce stable outputs
			for m in self.head.modules():
				if isinstance(m, nn.Linear):
					nn.init.xavier_uniform_(m.weight, gain=0.5)
					if m.bias is not None:
						nn.init.zeros_(m.bias)
		else:
			# Torchless fallback: no feature extractor; compute simple image stats instead
			self.feature_extractor = None
			self.head = None
		# Try to load trained OLX head (predicts INR directly). If present, disable USD->INR scaling.
		olx_head_path = os.path.join("models", "olx_price_head.pt")
		if self.torchvision_enabled and os.path.isfile(olx_head_path):
			state = torch.load(olx_head_path, map_location=self.device)
			try:
				self.head.load_state_dict(state)
				self.usd_to_inr = 1.0  # head outputs INR directly
				self.currency = "INR"
			except Exception:
				# If incompatible, keep default scaling
				pass

	@torch.inference_mode()
	def predict_from_tensor(self, tensor: torch.Tensor) -> Dict[str, Any]:
		if self.torchvision_enabled and self.feature_extractor is not None and self.head is not None:
			tensor = tensor.to(self.device, non_blocking=True)
			features = self.feature_extractor(tensor).flatten(1)
			raw_price = self.head(features).squeeze(1)
			price_usd = torch.clamp(raw_price, min=2000.0, max=150000.0)
			price_inr = price_usd * self.usd_to_inr
			price_value = float(price_inr.item())
			feat_std = float(features.std(dim=1).item())
			confidence = max(0.35, min(0.95, 0.5 + feat_std))
			return {
				"predicted_price_inr": round(price_value, 0),
				"confidence": round(confidence, 3),
				"attributes": {
					"embedding_std": round(feat_std, 4),
				},
				"model": {
					"backbone": "resnet18",
					"device": self.device,
					"fallback": False,
					"currency": self.currency,
				},
			}
		# Fallback path using simple image statistics
		arr = tensor.detach().cpu().numpy()
		# undo normalization roughly to 0..1 if likely ImageNet norm (best effort)
		mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
		std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)
		approx = arr * std + mean
		approx = np.clip(approx, 0.0, 1.0)
		# compute simple features
		brightness = float(approx.mean())
		contrast = float(approx.std())
		edge_signal = float(np.mean(np.abs(np.diff(approx, axis=-1))))  # crude edges on width
		# heuristic price
		base_usd = 25000.0 + 60000.0 * edge_signal + 20000.0 * (contrast - 0.2) + 5000.0 * (brightness - 0.5)
		base_inr = base_usd * self.usd_to_inr
		price_value = max(2000.0 * self.usd_to_inr, min(150000.0 * self.usd_to_inr, base_inr))
		# heuristic confidence
		confidence = max(0.35, min(0.75, 0.4 + 0.5 * contrast))
		return {
			"predicted_price_inr": round(float(price_value), 0),
			"confidence": round(float(confidence), 3),
			"attributes": {
				"brightness": round(brightness, 4),
				"contrast": round(contrast, 4),
				"edge_signal": round(edge_signal, 4),
			},
			"model": {
				"backbone": "fallback-stats",
				"device": "cpu",
				"fallback": True,
				"currency": self.currency,
			},
		}

	@torch.inference_mode()
	def predict_with_metadata(self, tensor: torch.Tensor, meta: Dict[str, Any]) -> Dict[str, Any]:
		base = self.predict_from_tensor(tensor)
		price = float(base.get("predicted_price_inr", 0.0))
		conf = float(base.get("confidence", 0.5))

		# Extract metadata with defaults
		mileage = meta.get("mileage_km") or 0.0
		year = meta.get("year") or 0
		condition = meta.get("condition") or 3
		engine_cc = meta.get("engine_cc") or 0
		seats = meta.get("seats") or 5
		brand = (meta.get("brand") or "").lower()

		# Apply multiplicative adjustments
		# Mileage: up to -40% at 200k km
		if mileage and mileage > 0:
			price *= max(0.6, 1.0 - (float(mileage) / 200000.0))
		# Year depreciation: ~3% per year
		if year and year > 0:
			age = max(0, 2025 - int(year))
			price *= max(0.5, 1.0 - 0.03 * age)
		# Condition [1..5]: center 3, +/- 8% per step
		if condition:
			price *= 1.0 + (int(condition) - 3) * 0.08
		# Engine size: relative to 1200cc baseline, +6% per +400cc up to +30%
		if engine_cc and engine_cc > 0:
			engine_factor = min(1.30, 1.0 + max(-0.2, (int(engine_cc) - 1200) / 400 * 0.06))
			price *= engine_factor
		# Seats: each seat above/below 5 adjusts by +/-3% up to +/-12%
		if seats and int(seats) > 0:
			seat_delta = max(-4, min(4, int(seats) - 5))
			price *= 1.0 + seat_delta * 0.03
		# Brand prior
		if brand:
			for key, w in self.brand_weights.items():
				if key in brand:
					price *= w
					break

		# Update confidence slightly when more metadata present
		meta_fields = sum(1 for v in [mileage, year, condition, engine_cc, seats, brand] if v)
		conf = min(0.98, conf + 0.01 * meta_fields)

		base["predicted_price_inr"] = round(price, 0)
		base["confidence"] = round(conf, 3)
		base.setdefault("attributes", {})
		base["attributes"].update({
			"mileage_km": mileage,
			"year": year,
			"condition": condition,
			"engine_cc": engine_cc,
			"seats": seats,
			"brand": brand,
		})
		return base


