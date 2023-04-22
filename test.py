import os
import pyrvc

print("Input RVC model path to load...")
model = pyrvc.Model(input())

OUTPUT_DIR = "pyrvc-converted"

os.makedirs(OUTPUT_DIR, exist_ok=True)
while 1:
    print("Input voice file to convert...")
    path = input()
    if not path: break
    model.save_as(model.convert_from_file(path),
                  os.path.join(OUTPUT_DIR, os.path.basename(path)))
