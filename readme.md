# Smart GIF Color Reducer

A Streamlit web application for compressing GIF files through intelligent color reduction while preserving animation quality. The app provides multiple compression versions with different optimization strategies.

## Features

- Uploads and previews GIF files
- Adjustable target file size
- Real-time compression preview
- Detailed compression statistics
- Easy-to-use web interface
- Preserves original frame count and animation timing

## Requirements

- Python 3.9+
- venv (Python virtual environment)
- Requirements listed in `requirements.txt`

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/renjithvsampcome/git_color_reduction.git
cd git_color_reduction
```

2. Create and activate virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
streamlit run week3/app.py
```

The app will open in your default web browser at `http://localhost:8501`

## Usage

1. Upload a GIF file using the file uploader
2. Adjust the target size using the slider
3. Click "Reduce Colors" to start compression
4. Preview the results and download the compressed GIF

