import pyaudio
import numpy as np
from faster_whisper import WhisperModel, BatchedInferencePipeline

model = WhisperModel('distil-large-v3', compute_type='int8')

def record_audio(duration=5, sample_rate=16000):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    p = pyaudio.PyAudio()

    print("Recording...")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(sample_rate / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording complete.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert to numpy array
    audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio


def stream_transcription_audio(audio):

    # audio_float = audio.astype(np.float32) / 32768.0  # Normalize to -1.0 to 1.0
    audio_float = np.require(audio, dtype=np.float32,
                             requirements='C') / 32768.0

    batched_model = BatchedInferencePipeline(model=model)
    segments, info = batched_model.transcribe(audio_float, batch_size=16,vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500))

    for segment in segments:
        yield segment.start, segment.end, segment.text


if __name__ == "__main__":
    audio = record_audio()
    for start, end, text in stream_transcription_audio("audio.mp3"):
        print(f"[{start:.2f}s -> {end:.2f}s] {text}")
