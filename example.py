import pyrvc

print("Input a RVC model path...")
model = pyrvc.Model(input())

while  1:
    print("input a voice file to convert...")
    wave = input()
    if not wave: break
    wave = pyrvc.Wave.from_file(wave)
    model.convert(wave).play()
