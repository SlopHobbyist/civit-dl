# Civit-DL - Project Summary

> **Note**: This entire project was written by AI (Claude Code: Sonnet 4.5, and Antigravity: Gemini 3 Pro (High)). All code, architecture decisions, and implementation details were generated through AI assistance.
>
> I am a strong advocate for never mixing generated code into real repos.
> Projects like these should clearly disclose as such.
## What It Is

A small command-line tool for downloading models from [Civitai](https://civitai.com). Given a model URL (or a list of them), it fetches the model file, a preview image, and writes a sidecar `.json` with metadata suitable for use with WebUI / Forge-style LoRA managers.

## Current Implementation

### Core Features

- Download by single URL: `python civitaidl.py https://civitai.com/models/12345`
- Batch download from `download.txt` (one URL per line, `#` comments allowed)
- Honors `?modelVersionId=` in the URL; otherwise falls back to the latest version
- Sorts files into `downloads/Checkpoints/`, `downloads/LoRAs/`, or `downloads/Other/` by model type
- Saves preview image as `<modelname>.<ext>` next to the model
- Writes a sidecar `<modelname>.json` (`sd version`, `activation text`, etc.)
- Skips files that already exist
- Progress bar via `tqdm` (falls back to plain prints if not installed)
- Civitai API key required via `key.txt` (downloads are authenticated)

### How It Works

1. Extracts the model ID (and optional version ID) from the Civitai URL.
2. Calls `GET https://civitai.com/api/v1/models/{id}` for metadata.
3. Picks the primary file from the chosen version, downloads it, then downloads the first preview image.
4. Writes the sidecar JSON.

## Supported Platforms

Anywhere Python 3.8+ runs (developed/tested on Windows). No OS-specific code.

## How to Run

```bash
pip install -r requirements.txt
```

**Required** — create a `key.txt` in the repo root containing only your Civitai API key (get one at https://civitai.com/user/account). Civitai requires authentication for model downloads. `key.txt` is gitignored.

Single URL:
```bash
python civitaidl.py "https://civitai.com/models/12345"
```

Batch — create `download.txt` with one URL per line, then:
```bash
python civitaidl.py
```

Output goes to `./downloads/<Type>/<ModelName>/`.

## License

**CC0 1.0 Universal (Public Domain)**

This work has been dedicated to the public domain under CC0 1.0 Universal.

You can:
- Use this code for any purpose (commercial, personal, educational, etc.)
- Modify and distribute it freely
- Use it without any attribution required

To the extent possible under law, the author has waived all copyright and related rights to this work.

For more information: [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)