from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image
import numpy as np
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
async def detect_custom(
        request: Request,
        image: UploadFile = File(...),
        keyword: str = Form(...),
        threshold: float = Form(50.0),
        use_yolo: str = Form(None),  # Add this line
        use_clip: str = Form(None),
        use_shapes: str = Form(None)
):
    threshold = max(threshold, 30.0)
    image_bytes = await image.read()

    # Initialize result variables
    shape_stats = {}
    shape_match = False
    object_match = False
    similarity_score = 0
    detected_objects = []
    yolo_match = False

    # Step 1: Try YOLO if enabled
    if use_yolo:
        from app.services.yolo_detector import detect_objects
        yolo_match, detected_objects, annotated_image = detect_objects(image_bytes, keyword)

        if yolo_match:
            # If YOLO detects the object, we're done
            object_match = True

    # Only proceed with other methods if YOLO didn't find a match
    if not yolo_match:
        # Step 2: Try CLIP if enabled
        if use_clip:
            from app.services.clip_classifier import classify_with_clip
            object_match, similarity_score = classify_with_clip(image_bytes, keyword, threshold)
            if not object_match:
                # Step 3: Try shape detection if enabled and previous methods failed
                if use_shapes:
                    from app.services.shape_detector import detect_shapes_from_bytes
                    shape_stats, annotated_image = detect_shapes_from_bytes(image_bytes)
                    shape_match = keyword.lower() in [s.lower() for s in shape_stats]
                    object_match = shape_match

    # Prepare final response
    _, buffer = cv2.imencode('.png', annotated_image)
    encoded_image = base64.b64encode(buffer).decode("utf-8")

    result = {
        "object_related": object_match,
        "similarity_score": similarity_score if use_clip else None,
        "shape_related": shape_match if use_shapes else None,
        "detected_shapes": shape_stats if use_shapes else {},
        "detected_objects": detected_objects if use_yolo else [],
        "yolo_match": yolo_match if use_yolo else False,
        "preview": encoded_image
    }

    return templates.TemplateResponse("index.html", {"request": request, "result": result})