"""File classification and image-to-tensor pipeline.

``talu.file`` classifies files by kind and provides typed wrappers for
processing. Zero runtime dependencies.

Quick Start::

    import talu.file

    pipeline = talu.file.Pipeline(size=(384, 384))
    with talu.file.open("photo.jpg") as img:
        buffer = pipeline(img)
        print(buffer.shape, buffer.dtype)

Multi-page PDFs::

    with talu.file.open("report.pdf") as doc:
        for page_image in doc:
            buffer = pipeline(page_image)
"""

from ._pipeline import ModelBuffer, Pipeline
from ._types import Audio, Binary, Document, Image, Text, Video, open

__all__ = [
    "open",
    "Pipeline",
    "Image",
    "Document",
    "Audio",
    "Video",
    "Text",
    "Binary",
    "ModelBuffer",
]
