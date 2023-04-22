# What is this

``pyrvc`` is a python library that enables voice conversion using RVC models.

# How to prepare RVC model

The RVC models should be available on the web from someone else. <br>
Alternatively, you may use the following projects to create your models.

- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://github.com/ddPn08/rvc-webui

# How to use this

```python
MODEL_PATH = ".../moden_name.pth" # RVC model file
VOICE_PATH = ".../voice_name.wav" # Using pydub to load voice data.
OUTPUT_PATH = ".../output.wav"    # Using pydub to save outputs.

import pyrvc

model = pyrvc.Model(MODEL_PATH)

output = model.convert_from_file(VOICE_PATH)

model.save_as(output, OUTPUT_PATH)
```

# Requirements

- ffmpeg

- PyTorch

- CUDA (optional)
