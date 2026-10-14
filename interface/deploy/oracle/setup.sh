#!/usr/bin/env bash
# One-time setup of the ReXTrust backend on an Oracle Cloud Ubuntu 22.04 VM (Ampere A1 or x86).
#
#   bash setup.sh <frontend-origin>        e.g.  bash setup.sh https://rextrust.vercel.app
#
# Expects the project uploaded to ~/rextrust (pipeline_core.py, radcheck.py, .env, Models/, interface/backend, interface/assets).
# Result: uvicorn on 127.0.0.1:8000 as a systemd service, behind Caddy with HTTPS at https://<public-ip-dashed>.sslip.io
set -euo pipefail

ORIGIN="${1:-https://rextrust.vercel.app}"
APP="$HOME/rextrust"
VENV="$APP/.venv"
PUBLIC_IP="$(curl -fsS https://api.ipify.org)"
DOMAIN="${PUBLIC_IP//./-}.sslip.io"

echo "==> Packages"
sudo apt-get update -y
sudo apt-get install -y python3-venv python3-dev git libgl1 libglib2.0-0 debian-keyring debian-archive-keyring apt-transport-https curl

echo "==> Python environment"
python3 -m venv "$VENV"
"$VENV/bin/pip" install -U pip wheel
# CPU-only torch: the default PyPI wheel on aarch64 is CPU; on x86 use the CPU index to skip ~2 GB of CUDA libs
if [ "$(uname -m)" = "x86_64" ]; then
  "$VENV/bin/pip" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
fi
"$VENV/bin/pip" install -r "$APP/interface/backend/requirements.txt"
# MediVLM's inference path only needs torch, torchvision, transformers and pyyaml (all covered above)

echo "==> Backend service"
sudo tee /etc/systemd/system/rextrust.service >/dev/null <<EOF
[Unit]
Description=ReXTrust FastAPI backend
After=network-online.target

[Service]
User=$USER
WorkingDirectory=$APP/interface/backend
Environment=REXTRUST_CORS=$ORIGIN
ExecStart=$VENV/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable --now rextrust

echo "==> Caddy (HTTPS reverse proxy)"
if ! command -v caddy >/dev/null; then
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/gpg.key | sudo gpg --dearmor --yes -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
  curl -1sLf https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
  sudo apt-get update -y && sudo apt-get install -y caddy
fi
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
$DOMAIN {
	reverse_proxy 127.0.0.1:8000 {
		transport http {
			read_timeout 10m
		}
	}
}
EOF

echo "==> Firewall: Oracle's Ubuntu image rejects everything but SSH in iptables"
for p in 80 443; do
  sudo iptables -C INPUT -p tcp --dport $p -j ACCEPT 2>/dev/null || sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport $p -j ACCEPT
done
sudo netfilter-persistent save || true
sudo systemctl restart caddy

echo
echo "Backend URL: https://$DOMAIN"
echo "Check:       curl https://$DOMAIN/health"
