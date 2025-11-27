from typing import Tuple
from PIL import Image
import numpy as np
import torch
try:
	import torchvision.transforms as T  # type: ignore
	_HAS_TORCHVISION = True
except Exception:
	T = None  # type: ignore
	_HAS_TORCHVISION = False


if _HAS_TORCHVISION:
	_TRANSFORM = T.Compose([
		T.Resize(256, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
		T.CenterCrop(224),
		T.ToTensor(),
		T.Normalize(mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225]),
	])
else:
	# Minimal manual preprocessing mirroring ImageNet stats
	_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def image_to_tensor(img: Image.Image) -> torch.Tensor:
	if _HAS_TORCHVISION:
		t = _TRANSFORM(img)
		if t.ndim == 3:
			t = t.unsqueeze(0)
		return t
	# Manual path: resize shortest side to 256, center crop 224, normalize
	img = img.convert("RGB")
	w, h = img.size
	scale = 256 / min(w, h)
	nw, nh = int(round(w * scale)), int(round(h * scale))
	img = img.resize((nw, nh), Image.BICUBIC)
	left = (nw - 224) // 2
	top = (nh - 224) // 2
	img = img.crop((left, top, left + 224, top + 224))
	arr = np.asarray(img).astype(np.float32) / 255.0
	arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
	# HWC -> CHW
	arr = np.transpose(arr, (0, 1, 2))
	# ensure channel-first
	arr = np.transpose(arr, (2, 0, 1))
	tensor = torch.from_numpy(arr).unsqueeze(0)
	return tensor


