from tts import TTS
import json
import os

infer_type = "torch"

model_dir=input("Path to a model:").strip()

with open(os.path.join(model_dir,"voices","voices.json"), encoding="utf8") as inf:
    voices=json.load(inf)

if len(voices["voices"])>1:
    print("Available voices:")
    for i,v in enumerate(voices["voices"]):
        print(i,v)
    voice=input("Choose one: ").strip()
    myvoice=voices["voices"][int(voice)]
else:
    myvoice=voices["voices"][0]

sambert_hifigan_tts = TTS(basepath=model_dir,
                              voice=myvoice, infer_type=infer_type)
txt=input("Enter text you wish spoken here:").strip()
speed=input("Speed:").strip()
if speed=="":
    speed=1.0
else:
    speed=float(speed)

outp = sambert_hifigan_tts.infer(text=txt, scale=speed)
print(f"Output saved to {outp}")

