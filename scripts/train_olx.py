import os
import argparse
import csv
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

try:
	import torchvision.models as models  # type: ignore
	import torchvision.transforms as T  # type: ignore
	_HAS_TORCHVISION = True
except Exception:
	_HAS_TORCHVISION = False

from backend.model import PriceHead


class OLXDataset(Dataset):
	def __init__(self, csv_path: str, images_root: str, transform) -> None:
		self.samples = []
		with open(csv_path, "r", encoding="utf-8") as f:
			reader = csv.DictReader(f)
			for row in reader:
				img_rel = row.get("image_path") or row.get("image") or row.get("path")
				price_str = row.get("price_inr") or row.get("price") or row.get("amount")
				if not img_rel or not price_str:
					continue
				try:
					price = float(price_str)
				except Exception:
					continue
				self.samples.append((os.path.join(images_root, img_rel), price))
		self.transform = transform

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
		path, price = self.samples[idx]
		img = Image.open(path).convert("RGB")
		x = self.transform(img)
		y = torch.tensor([price], dtype=torch.float32)
		return x, y


def get_dataloaders(csv_path: str, images_root: str, batch_size: int = 16):
	if not _HAS_TORCHVISION:
		raise RuntimeError("torchvision is required for training. Please install torch and torchvision.")
	transform = T.Compose([
		T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
		T.CenterCrop(224),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
	ds = OLXDataset(csv_path, images_root, transform)
	n = len(ds)
	n_train = int(0.9 * n)
	n_val = n - n_train
	train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
	return (
		DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True),
		DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True),
	)


def train(csv_path: str, images_root: str, out_dir: str = "models", epochs: int = 5, lr: float = 1e-3, batch_size: int = 16, device: str | None = None):
	if not _HAS_TORCHVISION:
		raise RuntimeError("torchvision is required for training.")
	device = device or ("cuda" if torch.cuda.is_available() else "cpu")
	base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # type: ignore
	feature_extractor = nn.Sequential(*list(base.children())[:-1]).to(device).eval()
	for p in feature_extractor.parameters():
		p.requires_grad = False
	head = PriceHead(in_features=512).to(device)
	criterion = nn.SmoothL1Loss()
	optimizer = torch.optim.AdamW(head.parameters(), lr=lr)

	train_loader, val_loader = get_dataloaders(csv_path, images_root, batch_size=batch_size)

	best_val = float("inf")
	os.makedirs(out_dir, exist_ok=True)

	for epoch in range(1, epochs + 1):
		head.train()
		running = 0.0
		for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [train]"):
			x = x.to(device); y = y.to(device)
			with torch.no_grad():
				feat = feature_extractor(x).flatten(1)
			pred = head(feat)
			loss = criterion(pred, y)
			optimizer.zero_grad(set_to_none=True)
			loss.backward()
			optimizer.step()
			running += loss.item() * x.size(0)
		train_loss = running / len(train_loader.dataset)

		# validation
		head.eval()
		val_running = 0.0
		with torch.no_grad():
			for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [val]"):
				x = x.to(device); y = y.to(device)
				feat = feature_extractor(x).flatten(1)
				pred = head(feat)
				loss = criterion(pred, y)
				val_running += loss.item() * x.size(0)
		val_loss = val_running / len(val_loader.dataset)
		print(f"epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f}")

		# checkpoint best
		if val_loss < best_val:
			best_val = val_loss
			torch.save(head.state_dict(), os.path.join(out_dir, "olx_price_head.pt"))
			print("Saved best head to", os.path.join(out_dir, "olx_price_head.pt"))


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train price head on OLX dataset (INR).")
	parser.add_argument("--csv", required=True, help="CSV file with columns: image_path,price_inr")
	parser.add_argument("--images", required=True, help="Root folder containing images referenced by image_path")
	parser.add_argument("--out", default="models", help="Output directory for weights")
	parser.add_argument("--epochs", type=int, default=5)
	parser.add_argument("--lr", type=float, default=1e-3)
	parser.add_argument("--batch-size", type=int, default=16)
	args = parser.parse_args()
	train(args.csv, args.images, out_dir=args.out, epochs=args.epochs, lr=args.lr, batch_size=args.batch_size)


