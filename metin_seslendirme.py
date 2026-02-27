from transformers import VitsModel, AutoTokenizer
import torch
import scipy.io.wavfile as wav

model = VitsModel.from_pretrained("facebook/mms-tts-tur")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-tur")

text = "Merhaba, nasılsınız?"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    output = model(**inputs).waveform

wav.write("output.wav", rate=model.config.sampling_rate, data=output)
