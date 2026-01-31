import sounddevice as sd
import numpy as np
import time

DEVICE_INDEX = 25
SAMPLE_RATE = 48000

def callback(indata, frames, time_info, status):
    audio = np.copy(indata)
    print("Audio received:", audio.shape)

stream = sd.InputStream(
    device=DEVICE_INDEX,
    channels=1,
    samplerate=SAMPLE_RATE,
    callback=callback
)

stream.start()

print("Listening to WhatsApp call audio...")
while True:
    time.sleep(1)
