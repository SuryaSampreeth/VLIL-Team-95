
# ─── Configuration Defaults ──────────────────────────────────────
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8000
MAX_UPLOAD_SIZE_MB = 50
ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".dcm"]
REQUEST_TIMEOUT_SECONDS = 120
BATCH_SIZE = 8
NUM_WORKERS = 4

def validate_image_dimensions(width, height):
    MIN_DIM, MAX_DIM = 224, 4096
    return MIN_DIM <= width <= MAX_DIM and MIN_DIM <= height <= MAX_DIM

def sanitize_filename(name):
    import re
    return re.sub(r'[^\w\-.]', '_', name)

DISEASE_LABELS = [
    "Atelectasis", "Cardiomegaly", "Consolidation",
    "Edema", "Pleural Effusion", "No Finding"
]
SEVERITY_LEVELS = {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}
CONFIDENCE_THRESHOLD = 0.65
