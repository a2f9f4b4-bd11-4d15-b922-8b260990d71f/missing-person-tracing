#!/usr/bin/env bash
# Install MediaMTX and ffmpeg on the Pi 4 and stream the USB webcam at boot.
set -euo pipefail
VERSION="${MEDIAMTX_VERSION:-v1.9.3}"
ARCH=arm64v8

sudo apt-get update
sudo apt-get install -y ffmpeg v4l-utils curl

echo "cameras seen by the Pi:"
v4l2-ctl --list-devices || true

tmp="$(mktemp -d)"
curl -fsSL -o "$tmp/m.tar.gz" \
  "https://github.com/bluenviron/mediamtx/releases/download/${VERSION}/mediamtx_${VERSION}_linux_${ARCH}.tar.gz"
tar -xzf "$tmp/m.tar.gz" -C "$tmp"
sudo install -m 755 "$tmp/mediamtx" /usr/local/bin/mediamtx
sudo install -m 644 "$(dirname "$0")/mediamtx.yml" /usr/local/etc/mediamtx.yml

sudo tee /etc/systemd/system/mediamtx.service >/dev/null <<'UNIT'
[Unit]
Description=MediaMTX camera node
After=network-online.target
[Service]
ExecStart=/usr/local/bin/mediamtx /usr/local/etc/mediamtx.yml
Restart=always
RestartSec=2
[Install]
WantedBy=multi-user.target
UNIT

sudo systemctl daemon-reload
sudo systemctl enable --now mediamtx
sleep 3
systemctl is-active mediamtx
echo "stream should be at rtsp://$(hostname -I | awk '{print $1}'):8554/cam1"
