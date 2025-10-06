import pyaudio
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline
import time
import logging


SAMPLE_RATE = 16000
DISTIL_MODEL_NAME = "distil-medium.en"
DEFAULT_DURATION = 5

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


try:
    model = WhisperModel(DISTIL_MODEL_NAME, device="cpu", compute_type="int8")
    pipeline = BatchedInferencePipeline(model=model)
    logger.info(f"Loaded ASR model: {DISTIL_MODEL_NAME}")
except Exception as e:
    logger.error(f"Failed to load ASR model: {e}")
    raise e


def record_audio(duration=DEFAULT_DURATION, sample_rate=SAMPLE_RATE):
    """
    Records audio from microphone and returns numpy int16 array.
    """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    frames = []
    p = pyaudio.PyAudio()
    stream = None

    try:
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=CHUNK)
        logger.info(f"Recording audio for {duration} seconds...")
        for _ in range(0, int(sample_rate / CHUNK * duration)):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(data)
        logger.info("Recording complete.")
        audio = np.frombuffer(b''.join(frames), dtype=np.int16)
        return audio
    except Exception as e:
        logger.error(f"Error recording audio: {e}")
        return None
    finally:
        if stream:
            try:
                stream.stop_stream()
                stream.close()
            except Exception as e:
                logger.warning(f"Error closing stream: {e}")
        p.terminate()


def transcribe_audio(audio):
    """
    Transcribes full audio using distil-medium.en.int8 model.
    Returns the transcription string.
    """
    if audio is None or len(audio) == 0:
        logger.warning("No audio provided for transcription.")
        return ""

    try:
        # Normalize audio
        audio_float = np.require(
            audio, dtype=np.float32, requirements='C') / 32768.0
        segments, _ = pipeline.transcribe(audio_float, vad_filter=True)
        full_text = " ".join([seg.text for seg in segments])
        return full_text
    except Exception as e:
        logger.error(f"Error during transcription: {e}")
        return ""


def main():
    start_time = time.perf_counter()
    audio = record_audio(duration=DEFAULT_DURATION)
    if audio is None:
        logger.error("Recording failed. Exiting.")
        return

    model_start = time.perf_counter()
    transcript = transcribe_audio(audio)
    model_end = time.perf_counter()

    logger.info(f"Transcription complete: {transcript}")
    logger.info(
        f"Total pipeline time: {time.perf_counter() - start_time:.2f}s")
    logger.info(f"ASR inference time: {model_end - model_start:.2f}s")


if __name__ == "__main__":
    main()
