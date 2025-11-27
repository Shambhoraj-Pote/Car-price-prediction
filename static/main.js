const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("canvas");
const snapBtn = $("snap");
const fileInput = $("file");
const resultEl = $("result");
const confEl = $("confidence");
const attrsEl = $("attributes");
const historyEl = $("history");
const saveBtn = $("save");
const shareBtn = $("share");
const mileageEl = $("mileage");
const yearEl = $("year");
const conditionEl = $("condition");
const engineEl = $("engine_cc");
const seatsEl = $("seats");
const brandEl = $("brand");

const API_BASE = "";
const history = [];
let lastBlob = null;
let lastImgUrl = null;
const snapshotImg = $("snapshot");
const resetBtn = $("resetImage");

async function initCamera() {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment" }, audio: false });
		video.srcObject = stream;
	} catch (e) {
		console.warn("Camera not available:", e);
	}
}

function captureFrameBlob() {
	const w = video.videoWidth || 1280;
	const h = video.videoHeight || 720;
	canvas.width = w;
	canvas.height = h;
	const ctx = canvas.getContext("2d");
	ctx.drawImage(video, 0, 0, w, h);
	return new Promise((resolve) => {
		canvas.toBlob((blob) => resolve(blob), "image/jpeg", 0.9);
	});
}

function adjustPrice(basePrice) {
	// Adjustments are now applied server-side; leave pass-through for UI recalculation.
	return basePrice;
}

async function predictFromBlob(blob) {
	const form = new FormData();
	form.append("image", blob, "frame.jpg");
	if (mileageEl?.value) form.append("mileage_km", mileageEl.value);
	if (yearEl?.value) form.append("year", yearEl.value);
	if (conditionEl?.value) form.append("condition", conditionEl.value);
	if (engineEl?.value) form.append("engine_cc", engineEl.value);
	if (seatsEl?.value) form.append("seats", seatsEl.value);
	if (brandEl?.value) form.append("brand", brandEl.value);
	const res = await fetch(`${API_BASE}/predict`, { method: "POST", body: form });
	if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
	const json = await res.json();
	return json;
}

function renderPrediction(pred, imgUrl) {
	const raw = pred.predicted_price_inr ?? 0;
	const adjusted = adjustPrice(raw);
	resultEl.textContent = `₹${Math.round(adjusted).toLocaleString("en-IN")}`;
	confEl.textContent = `Confidence: ${(pred.confidence * 100).toFixed(1)}%`;
	const attr = pred.attributes || {};
	const details = attr.embedding_std != null
		? `Embedding std: ${attr.embedding_std}`
		: `Brightness: ${attr.brightness ?? "—"} • Contrast: ${attr.contrast ?? "—"}`;
	attrsEl.textContent = details;

	const item = {
		id: Date.now(),
		price: adjusted,
		confidence: pred.confidence,
		imgUrl,
	};
	history.unshift(item);
	updateHistory();
}

function updateHistory() {
	historyEl.innerHTML = "";
	for (const item of history) {
		const li = document.createElement("li");
		li.className = "flex items-center gap-3";
		li.innerHTML = `
			<img src="${item.imgUrl}" class="w-16 h-16 object-cover rounded-lg border border-slate-200 dark:border-slate-800" />
			<div class="flex-1">
				<div class="font-medium">₹${Math.round(item.price).toLocaleString("en-IN")}</div>
				<div class="text-xs text-slate-500">Conf ${(item.confidence * 100).toFixed(0)}%</div>
			</div>
			<button class="text-xs px-2 py-1 rounded border border-slate-200 dark:border-slate-800 hover:bg-slate-50 dark:hover:bg-slate-900">Compare</button>
		`;
		historyEl.appendChild(li);
	}
}

function requireFields() {
	if (!mileageEl.value || !yearEl.value || !engineEl.value || !seatsEl.value || !brandEl.value) {
		alert("Please fill Mileage, Year, Engine CC, Seats, and Brand before predicting.");
		return false;
	}
	return true;
}

snapBtn.addEventListener("click", async () => {
	try {
		if (!requireFields()) return;
		const blob = await captureFrameBlob();
		lastBlob = blob;
		lastImgUrl = URL.createObjectURL(blob);
		// Freeze video and show snapshot
		try { video.pause?.(); } catch {}
		snapshotImg.src = lastImgUrl;
		snapshotImg.classList.remove("hidden");
		snapBtn.disabled = true;
		fileInput.disabled = true;
		resetBtn.classList.remove("hidden");
		const pred = await predictFromBlob(lastBlob);
		renderPrediction(pred, lastImgUrl);
	} catch (e) {
		alert(e.message || "Prediction failed");
	}
});

fileInput.addEventListener("change", async (e) => {
	const file = e.target.files?.[0];
	if (!file) return;
	try {
		if (!requireFields()) { e.target.value = ""; return; }
		lastBlob = file;
		lastImgUrl = URL.createObjectURL(file);
		// Freeze video and show snapshot
		try { video.pause?.(); } catch {}
		snapshotImg.src = lastImgUrl;
		snapshotImg.classList.remove("hidden");
		snapBtn.disabled = true;
		fileInput.disabled = true;
		resetBtn.classList.remove("hidden");
		const pred = await predictFromBlob(lastBlob);
		renderPrediction(pred, lastImgUrl);
	} catch (err) {
		alert(err.message || "Upload failed");
	}
});

resetBtn.addEventListener("click", () => {
	// Allow selecting a new image
	lastBlob = null;
	if (lastImgUrl) URL.revokeObjectURL(lastImgUrl);
	lastImgUrl = null;
	snapshotImg.classList.add("hidden");
	snapshotImg.src = "";
	snapBtn.disabled = false;
	fileInput.disabled = false;
	resetBtn.classList.add("hidden");
	try { video.play?.(); } catch {}
});

// Debounced re-predict on attribute changes
let repredictTimer = null;
function scheduleRepredict() {
	if (!lastBlob) return;
	clearTimeout(repredictTimer);
	repredictTimer = setTimeout(async () => {
		try {
			const pred = await predictFromBlob(lastBlob);
			renderPrediction(pred, lastImgUrl);
		} catch (e) {
			console.warn(e);
		}
	}, 300);
}

[mileageEl, yearEl, conditionEl, engineEl, seatsEl, brandEl].forEach((el) => {
	el?.addEventListener("input", scheduleRepredict);
	el?.addEventListener("change", scheduleRepredict);
});

saveBtn.addEventListener("click", () => {
	if (!history[0]) return;
	const blob = new Blob([JSON.stringify(history[0], null, 2)], { type: "application/json" });
	const a = document.createElement("a");
	a.href = URL.createObjectURL(blob);
	a.download = `carvision-${history[0].id}.json`;
	a.click();
});

shareBtn.addEventListener("click", async () => {
	if (!history[0]) return;
	try {
		const text = `CarVision AI estimate: ₹${Math.round(history[0].price).toLocaleString("en-IN")} (conf ${(history[0].confidence * 100).toFixed(0)}%)`;
		if (navigator.share) {
			await navigator.share({ title: "CarVision AI", text, url: location.href });
		} else {
			await navigator.clipboard.writeText(text);
			alert("Copied to clipboard!");
		}
	} catch (e) {
		console.warn(e);
	}
});

initCamera();


