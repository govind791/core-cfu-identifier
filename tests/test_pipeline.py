import io
import pytest
import numpy as np
from PIL import Image

from app.worker.pipeline import ImageProcessor


@pytest.fixture
def processor():
    return ImageProcessor()


@pytest.fixture
def sample_image_bytes():
    img = Image.new("RGB", (800, 800), color=(255, 255, 255))
    
    pixels = img.load()
    np.random.seed(42)
    for _ in range(20):
        cx = np.random.randint(100, 700)
        cy = np.random.randint(100, 700)
        for dx in range(-5, 6):
            for dy in range(-5, 6):
                if dx*dx + dy*dy <= 25:
                    x, y = cx + dx, cy + dy
                    if 0 <= x < 800 and 0 <= y < 800:
                        pixels[x, y] = (200, 180, 150)
    
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.fixture
def blank_image_bytes():
    img = Image.new("RGB", (800, 800), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


def test_processor_initialization(processor):
    assert processor.min_resolution == 640
    assert processor.focus_threshold == 0.5
    assert processor.glare_threshold == 0.5


def test_load_image(processor, sample_image_bytes):
    image = processor._load_image(sample_image_bytes)
    assert image is not None
    assert image.shape[0] == 800
    assert image.shape[1] == 800


def test_process_blank_image(processor, blank_image_bytes):
    result = processor.process(blank_image_bytes, "TFA_90MM")
    
    assert "cfu_count_total" in result
    assert "detections" in result
    assert "quality" in result
    assert "plate_found" in result
    assert "focus_score" in result
    assert "glare_score" in result


def test_process_returns_valid_structure(processor, sample_image_bytes):
    result = processor.process(sample_image_bytes, "TFA_90MM")
    
    assert isinstance(result["cfu_count_total"], (int, type(None)))
    assert isinstance(result["detections"], list)
    assert isinstance(result["plate_found"], bool)
    assert 0.0 <= result["focus_score"] <= 1.0
    assert 0.0 <= result["glare_score"] <= 1.0
    assert isinstance(result["needs_review"], bool)
    assert isinstance(result["reason_codes"], list)


def test_calculate_focus_score(processor, sample_image_bytes):
    image = processor._load_image(sample_image_bytes)
    score = processor._calculate_focus_score(image)
    assert 0.0 <= score <= 1.0


def test_calculate_glare_score(processor, sample_image_bytes):
    image = processor._load_image(sample_image_bytes)
    score = processor._calculate_glare_score(image)
    assert 0.0 <= score <= 1.0


def test_invalid_result(processor):
    result = processor._invalid_result("Test reason")
    
    assert result["cfu_count_total"] is None
    assert result["detections"] == []
    assert result["plate_found"] is False
    assert result["needs_review"] is True
    assert "INVALID_IMAGE" in result["reason_codes"]


def test_small_image_rejected(processor):
    img = Image.new("RGB", (320, 320), color=(255, 255, 255))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    small_bytes = buffer.getvalue()
    
    result = processor.process(small_bytes, "TFA_90MM")
    
    assert result["cfu_count_total"] is None
    assert result["needs_review"] is True


def test_annotated_image_generation(processor, sample_image_bytes):
    detections = [
        {"x": 0.25, "y": 0.25, "radius_px": 10, "score": 0.9},
        {"x": 0.75, "y": 0.75, "radius_px": 8, "score": 0.85},
    ]
    
    annotated = processor.generate_annotated_image(
        sample_image_bytes,
        detections,
        cfu_count=2,
        plate_roi=None,
    )
    
    assert isinstance(annotated, bytes)
    assert len(annotated) > 0
    
    img = Image.open(io.BytesIO(annotated))
    assert img.format == "PNG"
