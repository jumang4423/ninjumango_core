from pathlib import Path
import argh
import numpy as np
import pydub
from PIL import Image

from riffusion.spectrogram_image_converter import SpectrogramImageConverter
from riffusion.spectrogram_params import SpectrogramParams
from riffusion.util import image_util

def image_to_audio(*, image: str, audio: str, device: str = "cuda"):
    """
    Reconstruct an audio clip from a spectrogram image.
    """
    pil_image = Image.open(image)

    # Get parameters from image exif
    img_exif = pil_image.getexif()
    assert img_exif is not None

    try:
        params = SpectrogramParams.from_exif(exif=img_exif)
    except KeyError:
        print("WARNING: Could not find spectrogram parameters in exif data. Using defaults.")
        params = SpectrogramParams()

    converter = SpectrogramImageConverter(params=params, device=device)
    segment = converter.audio_from_spectrogram_image(pil_image)

    extension = Path(audio).suffix[1:]
    segment.export(audio, format=extension)

    print(f"Wrote {audio} ({segment.duration_seconds:.2f} seconds)")
