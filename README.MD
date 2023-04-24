# What is this

``pyrvc`` is a python library that enables voice conversion using RVC models.

# How to prepare RVC model

The RVC models should be available on the web from someone else. <br>
Alternatively, you may use the following projects to create your models.

- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://github.com/ddPn08/rvc-webui

# How to use this

```python
MODEL_PATH = ".../moden.pth"
INPUT_PATH = ".../input.wav"
OUTPUT_PATH = ".../output.wav"

import pyrvc

model = pyrvc.Model(MODEL_PATH)

wave = pyrvc.Wave.from_file(INPUT_PATH)

output = model.convert(wave)

output.play()
output.save_as(OUTPUT_PATH)
```

# Requirements

- ffmpeg

- PyTorch

- CUDA (optional)
