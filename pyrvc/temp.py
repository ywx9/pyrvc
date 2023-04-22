import os

DIR = os.path.dirname(__file__)

with open(os.path.join(DIR, "hubert_base.pt"), "rb") as f:
    data = f.read()

with open(os.path.join(DIR, "hubert_base.pt.1"), "wb") as f:
    f.write(data[:len(data)//2])

with open(os.path.join(DIR, "hubert_base.pt.2"), "wb") as f:
    f.write(data[len(data)//2:])
