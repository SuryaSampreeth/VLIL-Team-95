"""
ReXTrust backend on Modal (https://modal.com): the FastAPI app from interface/backend, served over HTTPS.

    modal volume put rextrust-models "Models" /          # once: upload weights (~3 GB) from the project folder
    modal secret create rextrust GOOGLE_API_KEY=...      # once: RadCheck key
    modal deploy interface/deploy/modal/modal_app.py     # prints the https://...modal.run URL

Code is baked into the image; weights live in a Volume mounted at /models and linked to <project>/Models,
which is where pipeline_core.py and config.yaml expect them. The container sleeps after SCALEDOWN seconds idle.
"""
import os
from pathlib import Path

import modal

# Project folder (pipeline_core.py, radcheck.py, Models/, interface/). Only meaningful on the deploying machine;
# inside the container this file sits at /root/modal_app.py and the local paths below are never read.
ROOT = Path(__file__).resolve().parents[3] if modal.is_local() else Path("/")
REMOTE = "/root/rextrust"
FRONTEND_ORIGIN = "https://rextrust.vercel.app"
SCALEDOWN = 600  # seconds of idle before the container (and its loaded models) shuts down

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install_from_requirements(str(ROOT / "interface/backend/requirements.txt"))
    .env({"REXTRUST_CORS": FRONTEND_ORIGIN})
    .add_local_file(ROOT / "pipeline_core.py", f"{REMOTE}/pipeline_core.py")
    .add_local_file(ROOT / "radcheck.py", f"{REMOTE}/radcheck.py")
    .add_local_dir(ROOT / "interface/backend", f"{REMOTE}/interface/backend", ignore=["**/__pycache__"])
    .add_local_dir(ROOT / "interface/assets", f"{REMOTE}/interface/assets")
)

models = modal.Volume.from_name("rextrust-models", create_if_missing=True)
app = modal.App("rextrust", image=image)


@app.function(
    volumes={"/models": models},
    secrets=[modal.Secret.from_name("rextrust")],
    cpu=4,
    memory=8192,
    timeout=600,
    scaledown_window=SCALEDOWN,
    max_containers=1,  # one warm copy of the models; also caps spend
)
@modal.concurrent(max_inputs=8)  # /health polls keep answering while a stage runs
@modal.asgi_app()
def web():
    import sys

    link = Path(REMOTE) / "Models"
    if not link.exists():
        link.symlink_to("/models")
    os.chdir(f"{REMOTE}/interface/backend")
    sys.path.insert(0, f"{REMOTE}/interface/backend")
    sys.path.insert(0, REMOTE)
    from app.main import app as fastapi_app

    return fastapi_app
