#!/bin/bash

set -u

echo ""
echo "========================================"
echo "       SECourses STARTUP SCRIPT"
echo "========================================"
date
echo ""

cd /workspace || exit 1

export HF_HOME="/workspace"

echo "===== [1/6] Downloading required files ====="

wget -O /workspace/Clear_Triton_Cache.py \
"https://raw.githubusercontent.com/walden966/test/main/Clear_Triton_Cache.py" || exit 1

wget -O /workspace/Models_Downloader.py \
"https://raw.githubusercontent.com/walden966/test/main/Models_Downloader.py" || exit 1

wget -O /workspace/RunPod_Install_SECourses_UpscalerPro.sh \
"https://raw.githubusercontent.com/walden966/test/main/RunPod_Install_SECourses_UpscalerPro.sh" || exit 1

wget -O /workspace/requirements.txt \
"https://raw.githubusercontent.com/walden966/test/main/requirements.txt" || exit 1

echo ""
echo "===== ALL FILES DOWNLOADED ====="
echo ""

chmod +x /workspace/RunPod_Install_SECourses_UpscalerPro.sh

echo "===== [2/6] Running SECourses installer ====="
echo ""

bash /workspace/RunPod_Install_SECourses_UpscalerPro.sh

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: SECourses installer failed!"
    exit 1
fi

echo ""
echo "===== INSTALLER FINISHED ====="
echo ""

cd /workspace/SECourses_Premium_Upscaler_Pro || exit 1

echo "===== [3/6] Updating SECourses repository ====="

git pull

echo ""
echo "===== [4/6] Starting SECourses ====="
echo ""

source venv/bin/activate

unset LD_LIBRARY_PATH

python secourses_app.py > /workspace/secourses.log 2>&1 &

SEC_PID=$!

echo "SECourses PID: $SEC_PID"
echo ""
echo "===== Waiting for SECourses on port 7860 ====="
echo ""

while true; do

    if curl -s http://127.0.0.1:7860 >/dev/null 2>&1; then
        echo ""
        echo "========================================"
        echo "       SECourses IS READY"
        echo "========================================"
        echo ""
        break
    fi

    if ! kill -0 "$SEC_PID" 2>/dev/null; then
        echo ""
        echo "ERROR: SECourses stopped unexpectedly!"
        echo ""
        echo "===== SECourses LOG ====="
        cat /workspace/secourses.log
        exit 1
    fi

    echo "Waiting..."
    sleep 2

done

echo "===== [5/6] Installing Cloudflared ====="
echo ""

if ! command -v cloudflared >/dev/null 2>&1; then

    echo "Cloudflared not found. Installing..."

    apt-get update
    apt-get install -y curl

    curl -L \
      https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb \
      -o /tmp/cloudflared.deb

    dpkg -i /tmp/cloudflared.deb

else

    echo "Cloudflared already installed."

fi

echo ""
echo "===== [6/6] Starting Cloudflare Tunnel ====="
echo ""
echo "========================================"
echo " PUBLIC SECourses URL WILL APPEAR BELOW"
echo "========================================"
echo ""

cloudflared tunnel --url http://127.0.0.1:7860
