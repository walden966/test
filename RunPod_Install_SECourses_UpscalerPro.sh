cd /workspace

export UV_SKIP_WHEEL_FILENAME_CHECK=1
export UV_LINK_MODE=copy

echo "Ensuring aria2c and FFmpeg are installed..."
if ! command -v aria2c >/dev/null 2>&1 \
    || ! command -v ffmpeg >/dev/null 2>&1 \
    || ! command -v ffprobe >/dev/null 2>&1; then
    SUDO=()
    if [ "$(id -u)" -ne 0 ]; then
        if command -v sudo >/dev/null 2>&1; then
            SUDO=(sudo)
        else
            echo "Root access or sudo is required to install aria2c and FFmpeg. Exiting..."
            exit 1
        fi
    fi

    if ! command -v apt-get >/dev/null 2>&1; then
        echo "apt-get is required to install aria2c and FFmpeg. Exiting..."
        exit 1
    fi

    if ! "${SUDO[@]}" apt-get update \
        || ! "${SUDO[@]}" apt-get install -y aria2 ffmpeg; then
        echo "Failed to install aria2c and FFmpeg. Exiting..."
        exit 1
    fi
fi

if ! command -v aria2c >/dev/null 2>&1; then
    echo "aria2c installation could not be verified. Exiting..."
    exit 1
fi
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
    echo "FFmpeg installation could not be verified. Exiting..."
    exit 1
fi
echo "Using $(aria2c --version | head -n 1)"
echo "Using $(ffmpeg -version | head -n 1)"

git clone https://github.com/FurkanGozukara/SECourses_Premium_Upscaler_Pro

cd SECourses_Premium_Upscaler_Pro

git reset --hard

git pull

echo "Provisioning the latest stable Python 3.12 with uv..."

VENV_DIR="venv"
VENV_PYTHON="$VENV_DIR/bin/python"
PYTHON_REQUEST="3.12"
PYTHON_VERSION_CMD='import sys; print(".".join(map(str, sys.version_info[:3])))'

export UV_PYTHON_PREFERENCE=only-managed

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- https://astral.sh/uv/install.sh | sh
    else
        echo "Neither curl nor wget is available to install uv. Exiting..."
        exit 1
    fi
fi

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null 2>&1; then
    echo "Failed to install uv. Exiting..."
    exit 1
fi

if ! uv python install "$PYTHON_REQUEST"; then
    echo "Failed to download Python $PYTHON_REQUEST. Exiting..."
    exit 1
fi

TARGET_PYTHON="$(uv python find "$PYTHON_REQUEST")"
if [ -z "$TARGET_PYTHON" ] || [ ! -x "$TARGET_PYTHON" ]; then
    echo "Could not locate the managed Python $PYTHON_REQUEST interpreter. Exiting..."
    exit 1
fi

PYTHON_FULL_VERSION="$("$TARGET_PYTHON" -c "$PYTHON_VERSION_CMD")"
echo "Using Python $PYTHON_FULL_VERSION from $TARGET_PYTHON"

if [ -x "$VENV_PYTHON" ] && [ "$("$VENV_PYTHON" -c "$PYTHON_VERSION_CMD" 2>/dev/null)" = "$PYTHON_FULL_VERSION" ]; then
    echo "Reusing existing Python $PYTHON_FULL_VERSION virtual environment."
else
    if [ -e "$VENV_DIR" ]; then
        echo "Existing virtual environment is not on Python $PYTHON_FULL_VERSION; recreating it..."
        rm -rf "$VENV_DIR"
    else
        echo "No existing virtual environment was found; creating one..."
    fi

    if ! uv venv --python "$TARGET_PYTHON" --seed "$VENV_DIR"; then
        echo "Failed to create the Python $PYTHON_FULL_VERSION virtual environment. Exiting..."
        exit 1
    fi
fi

if [ ! -x "$VENV_PYTHON" ]; then
    echo "Python 3.12 virtual environment setup failed. Exiting..."
    exit 1
fi

source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

python -m pip install uv


cd ..

uv pip install -r requirements.txt

cd SECourses_Premium_Upscaler_Pro

git clone https://github.com/FurkanGozukara/RIFE_Ultimate_Video_Upscaler RIFE

git clone https://github.com/FurkanGozukara/Video_Comparison_Slider

git clone https://github.com/FurkanGozukara/SeedVR2

git clone https://github.com/furkanGozukara/FlashVSR_plus

git clone https://github.com/FurkanGozukara/ComfyUI-FlashVSR_Stable

cd ComfyUI-FlashVSR_Stable

git reset --hard

git pull

cd ..

cd SeedVR2

git reset --hard

git pull

cd ..

cd RIFE

git reset --hard

git pull

cd ..

cd FlashVSR_plus

git reset --hard

git pull

cd ..

cd ..

echo "Models will be downloaded individually when first selected in the app."

echo .
echo .
echo .
echo .
echo Ultimate Video Image Upscaler installed check out all messages and save them before close to verify any errors or not later
