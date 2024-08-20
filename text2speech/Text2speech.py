import numpy as np
from gtts import gTTS
import io
import os

def text_to_speech_gtts(text, lang='en', slow=False):
    tts = gTTS(text=text, lang=lang, slow=slow)
    byte_io = io.BytesIO()
    tts.write_to_fp(byte_io)
    byte_io.seek(0)
    np_array = np.frombuffer(byte_io.read(), dtype=np.uint8)
    
    # Write the byte data to an mp3 file
    with open("output.mp3", "wb") as f:
        f.write(np_array)
    
    # Play the mp3 file
    if os.name == 'nt':  # For Windows
        os.system("start output.mp3")
    else:  # For macOS and Linux
        os.system("open output.mp3" if os.name == 'darwin' else "xdg-open output.mp3")
    
    return np_array

# Example usage
audio_np_array = text_to_speech_gtts("Hello, what a wonderful day")
print(audio_np_array)
