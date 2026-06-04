"""🛰️ busel MULTIMODAL — encoders for image, video, audio, PDF, docx.

Example:
    from multimodal import build_encoder_for
    enc = build_encoder_for("photo.jpg")
    blob = enc.encode_file("photo.jpg")
"""
from multimodal.encoders import (
    ImageEncoder,
    VideoEncoder,
    AudioEncoder,
    PDFEncoder,
    DocxEncoder,
    TextEncoder,
    auto_encode,
    build_encoder_for,
    list_encoders,
)

__all__ = [
    "ImageEncoder",
    "VideoEncoder",
    "AudioEncoder",
    "PDFEncoder",
    "DocxEncoder",
    "TextEncoder",
    "auto_encode",
    "build_encoder_for",
    "list_encoders",
]
