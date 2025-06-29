from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.services.shape_detector import detect_shapes_from_bytes
from app.services.clip_classifier import classify_with_clip
import base64
from io import BytesIO
import cv2

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect/", response_class=HTMLResponse)
async def detect_shape(
    request: Request,
    image: UploadFile = File(...),
    keyword: str = Form(...),
    threshold: float = Form(50.0)
):
    threshold = max(threshold, 30.0)  # enforce minimum threshold
    image_bytes = await image.read()

    shape_stats, annotated_image = detect_shapes_from_bytes(image_bytes)
    shape_match = keyword.lower() in [s.lower() for s in shape_stats]
    object_match, similarity_score = classify_with_clip(image_bytes, keyword, threshold)

    _, buffer = cv2.imencode('.png', annotated_image)
    encoded_image = base64.b64encode(buffer).decode('utf-8')

    result = {
        "shape_related": shape_match,
        "object_related": object_match,
        "similarity_score": similarity_score,
        "detected_shapes": shape_stats,
        "preview": encoded_image
    }

    return templates.TemplateResponse("index.html", {"request": request, "result": result})
