#!/usr/bin/env python3
import os
import sys
import argparse
import requests
import re
import mimetypes
from urllib.parse import urlparse, parse_qs
import json

# Try to import tqdm for progress bar, otherwise fallback
try:
    from tqdm import tqdm
except ImportError:
    # Simple dummy tqdm if not installed
    class tqdm:
        def __init__(self, total=None, unit='B', unit_scale=False, unit_divisor=1024, desc=None):
            self.total = total
            self.current = 0
            self.desc = desc
            print(f"Downloading {desc}..." if desc else "Downloading...")

        def update(self, n):
            self.current += n

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.total and self.current < self.total:
                print("Download incomplete.")
            else:
                print("Done.")

API_URL = "https://civitai.com/api/v1"
__version__ = "0.1.0"
REQUEST_TIMEOUT = 30

def get_api_key():
    """Load API key from CIVITAI_API_KEY env var or key.txt file, or return None."""
    env_key = os.environ.get("CIVITAI_API_KEY", "").strip()
    if env_key:
        return env_key
    if os.path.exists("key.txt"):
        try:
            with open("key.txt", "r") as f:
                key = f.read().strip()
                if key:
                    return key
        except Exception:
            pass
    return None

def get_model_id_from_url(url):
    """Extract model ID from Civitai URL."""
    # Matches https://civitai.com/models/123456 or https://civitai.com/models/123456/name
    match = re.search(r'models/(\d+)', url)
    if match:
        return match.group(1)
    return None

def get_version_id_from_url(url):
    """Extract modelVersionId from URL query parameters if present."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    if 'modelVersionId' in qs:
        return qs['modelVersionId'][0]
    return None

def clean_filename(name):
    """Sanitize filename by removing illegal characters."""
    return re.sub(r'[\\/*?:"<>|]', "", name)

def get_file_extension(filename):
    """Get extension including dot."""
    ext = os.path.splitext(filename)[1]
    return ext if ext else ""

def download_file(url, filepath, api_key=None):
    """Download a file with progress bar."""
    headers = {}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    
    try:
        response = requests.get(url, stream=True, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 # 1 Kibibyte
        
        mode = 'wb'
        
        with open(filepath, mode) as f:
            with tqdm(total=total_size, unit='iB', unit_scale=True, unit_divisor=1024, desc=os.path.basename(filepath)) as bar:
                for data in response.iter_content(block_size):
                    size = f.write(data)
                    bar.update(size)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        # Clean up partial file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        return False

def process_model(url, api_key):
    """Process a single model URL."""
    url = url.strip()
    if not url:
        return

    print(f"\nProcessing: {url}")
    
    model_id = get_model_id_from_url(url)
    if not model_id:
        print(f"Error: Could not extract model ID from {url}")
        return

    # Fetch model metadata
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    
    try:
        resp = requests.get(f"{API_URL}/models/{model_id}", headers=headers, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            print(f"Model ID {model_id} not found.")
            return
        resp.raise_for_status()
        model_data = resp.json()
    except Exception as e:
        print(f"Error fetching metadata: {e}")
        return

    # Determine which version to download
    target_version_id = get_version_id_from_url(url)
    target_version = None
    
    if target_version_id:
        for v in model_data.get('modelVersions', []):
            if str(v['id']) == str(target_version_id):
                target_version = v
                break
        if not target_version:
             print(f"Warning: Requested version ID {target_version_id} not found, checking others.")
    
    # Default to first version (usually latest) if not specified or not found
    if not target_version:
        versions = model_data.get('modelVersions', [])
        if not versions:
            print("Error: No versions found for this model.")
            return
        target_version = versions[0]
        print(f"Using version: {target_version.get('name', 'Unknown')}")

    model_name = model_data.get('name', f"Model_{model_id}")
    clean_name = clean_filename(model_name)
    
    model_type = model_data.get('type', 'Unknown')
    subfolder = "Other"
    if model_type == "Checkpoint":
        subfolder = "Checkpoints"
    elif model_type in ["LORA", "LoCon"]:
        subfolder = "LoRAs"
    
    # Prepare directory
    download_dir = os.path.join("downloads", subfolder, clean_name)
    os.makedirs(download_dir, exist_ok=True)
    
    # --- Download Model File ---
    files = target_version.get('files', [])
    download_url = None
    target_filename = f"{clean_name}.safetensors" # Default assumption
    
    # Strategy: Find "Model" type or Primary file
    # Prefer safetensors if multiple exist
    primary_file = next((f for f in files if f.get('primary')), None)
    
    selected_file = primary_file
    
    # If primary is not found or we want to be picky (optional logic here)
    if not selected_file and files:
        selected_file = files[0]
        
    if selected_file:
         download_url = selected_file.get('downloadUrl')
         fname = selected_file.get('name')
         if fname:
             # Use the extension from the actual filename on Civitai
             ext = get_file_extension(fname)
             if not ext:
                  ext = ".safetensors" # fallback
             target_filename = f"{clean_name}{ext}"
    else:
        # Construct fallback download URL
        # https://civitai.com/api/download/models/12345
        download_url = f"{API_URL}/download/models/{target_version['id']}"
        # We'll assume safetensors for now or rely on content-disposition if we were fancy (but we need a filename first for the path)
        # Let's check the type of model to guess extension
        model_type = model_data.get('type', 'Checkpoint')
        if model_type == 'TextualInversion':
            target_filename = f"{clean_name}.pt"
        else:
            target_filename = f"{clean_name}.safetensors"

    final_model_path = os.path.join(download_dir, target_filename)
    
    print(f"Target file: {target_filename}")
    if os.path.exists(final_model_path):
        print("Model file already exists. Skipping.")
    else:
        if download_url:
            success = download_file(download_url, final_model_path, api_key)
            if not success:
                print("Failed to download model file.")
        else:
            print("Could not determine download URL.")

    # --- Download Preview Image ---
    images = target_version.get('images', [])
    if images and len(images) > 0:
        first_image = images[0]
        image_url = first_image.get('url')
        
        if image_url:
            # Determine extension
            # Attempt to guess from URL
            path = urlparse(image_url).path
            ext = os.path.splitext(path)[1]
            if not ext:
                if 'format=png' in image_url: ext = '.png'
                elif 'format=jpeg' in image_url: ext = '.jpg'
                else: ext = '.png' # Fallback default
            
            # User request: "./downloads/modelname/modelname.png"
            # We will use the model name and the detected extension.
            # If user strictly wants .png for everything we could try, but just saving as ext is safer for metadata.
            
            image_filename = f"{clean_name}{ext}"
            final_image_path = os.path.join(download_dir, image_filename)
            
            if os.path.exists(final_image_path):
                print("Image file already exists. Skipping.")
            else:
                success = download_file(image_url, final_image_path) 
                if not success:
                    print("Failed to download image.")
    else:
        print("No images found for this version.")

    # --- Create Sidecar JSON ---
    json_filename = f"{clean_name}.json"
    final_json_path = os.path.join(download_dir, json_filename)
    
    if not os.path.exists(final_json_path):
        trained_words = target_version.get('trainedWords', [])
        activation_text = ", ".join(trained_words) if trained_words else ""
        base_model = target_version.get('baseModel', 'Unknown')
        
        sidecar_data = {
            "description": "",
            "sd version": base_model,
            "activation text": activation_text,
            "preferred weight": 0,
            "negative text": "",
            "notes": ""
        }
        
        try:
            with open(final_json_path, 'w', encoding='utf-8') as f:
                json.dump(sidecar_data, f, indent=4)
            print(f"Created sidecar: {json_filename}")
        except Exception as e:
            print(f"Error creating sidecar: {e}")
    else:
        print("Sidecar JSON already exists. Skipping.")

def main():
    parser = argparse.ArgumentParser(description="Civitai Model Downloader")
    parser.add_argument("url", nargs="?", help="Single Civitai Model URL to download")
    args = parser.parse_args()

    api_key = get_api_key()
    if api_key:
        print("API Key loaded from key.txt")
    else:
        print("No API Key found in key.txt. Proceeding anonymously.")
        print("Note: Some models or files may require an API key.")

    urls = []
    
    # If argument provided, use it
    if args.url:
        urls.append(args.url)
    # Check for download.txt if no argument
    elif os.path.exists("download.txt"):
        print("Reading URLs from download.txt...")
        with open("download.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    urls.append(line)
    else:
        print("Usage: python civitaidl.py [URL]")
        print("Or create a 'download.txt' file with URLs.")
        return

    if not urls:
        print("No URLs to process.")
        return

    print(f"Found {len(urls)} URLs to process.")
    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}]")
        process_model(url, api_key)

    print("\nBatch completed.")

if __name__ == "__main__":
    main()
