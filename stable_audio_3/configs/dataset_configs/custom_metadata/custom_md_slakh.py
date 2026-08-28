"""
Custom metadata extractor for Slakh/BabySlakh datasets.

Extracts instrument information from file paths and optionally from metadata.yaml.
Expected path format:
    .../tracks/drums/Track00001/Drums.wav
    .../tracks/other/Track00001/Guitar.wav
"""

import os
import re
from functools import lru_cache
from pathlib import Path

import yaml


@lru_cache(maxsize=128)
def load_track_metadata(raw_data_root: str, track_id: str) -> dict:
    """Load and cache metadata.yaml for a track."""
    yaml_path = Path(raw_data_root) / track_id / "metadata.yaml"
    if yaml_path.exists():
        text = yaml_path.read_text()
        # Handle YAML that doesn't start with standard markers
        if text and text[0] not in ["{", "-"]:
            text = text
        return yaml.safe_load(text)
    return {}


def get_custom_metadata(info, audio):
    """
    Extract metadata for Slakh stems based on file path and YAML metadata.
    Always expects drum files (target audio vector for training).
    """
    filepath = info["path"]

    # Extract track ID from path (e.g., "Track00001")
    track_match = re.search(r"(Track\d+)", filepath)
    track_id = track_match.group(1) if track_match else None

    # Always drums - this function is only called for drum files
    is_drum = True
    prompt = "drums"

    return {
        "prompt": prompt,
        "is_drum": is_drum,
        "track_id": track_id,
    }