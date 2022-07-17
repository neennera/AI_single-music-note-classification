import streamlit as st
import pandas as pd
import numpy as np
import pickle
import torch
from torch import nn
import torchaudio
#import program
import pickle
import copyreg
#from pydub import AudioSegment
import os
#import tempfile


class Model(nn.Module):
    def __init__(self, feat_dim=256):
        super(Model, self).__init__()

        # wav2letter architecture
        self.convo1 = self._get_conv_block(128, 256, 7)
        self.convo2 = self._get_conv_block(256, 256, 7)
        self.convo3 = self._get_conv_block(256, 512, 7)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(512*50, 12)

    def _get_activation(self, name):
      if name == "relu":
        return nn.ReLU()
      elif name == "softmax":
        return nn.Softmax(dim=1)
      elif name == "sigmoid":
        return nn.Sigmoid()
    
    def _get_conv_block(self, in_channel, out_channel, kernel_size, activation="relu"):
      return nn.Sequential(
          nn.Conv1d(in_channel, out_channel, kernel_size, padding="same"),
          nn.BatchNorm1d(out_channel),
          self._get_activation(activation)
      )

    def forward(self, x):
        x = self.convo1(x)
        x = self.convo2(x)
        x = self.convo3(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x

def resample_if_necessary(signal, sr, target_sample_rate):
  if sr != target_sample_rate:
      resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
      signal = resampler(signal)
  return signal

def mix_down_if_necessary(signal):
  if signal.shape[0] > 1:
      signal = torch.mean(signal, dim=0, keepdim=True)
  return signal

def processed(uploaded_file):
    target_sample_rate=16000.0
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=34624,
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
 
    # Create tmp file
    path = os.path.join("temp",uploaded_file.name)
    with open(path,"wb") as f: 
      f.write(uploaded_file.getbuffer())

    # Dowload tmp flie
    signal, sr = torchaudio.load(path)
    signal = resample_if_necessary(signal, sr, target_sample_rate)
    signal = mix_down_if_necessary(signal)
    data = mel_spectrogram(signal)
    #data = data[0]
    st.audio(uploaded_file)
    os.remove(path)
    return data

def predict(model, inputs) :
    model.eval()
    data = processed(inputs)
    with torch.no_grad():
        pred = model(data)
    pred_index = (pred.argmax(dim=1)).to(dtype=int)
    return class_mapping[pred_index]

class MyCustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "__main__":
            module = "program"
        return super().find_class(module, name)

class MyClass:
    def __init__(self, name):
        self.name = "__main__"

def pickle_MyClass(obj):
    assert type(obj) is MyClass
    return program.MyClass, (obj.name,)

model = Model()
model = copyreg.pickle(MyClass, pickle_MyClass)


with open('notemodel.pkl', 'rb') as f :
    model = pickle.load(f)
class_mapping = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

st.set_page_config(page_title="Note Prediction")
st.image("music-banner.jpg", caption=None, width=700, use_column_width=None, clamp=False, channels='RGB',output_format='auto')
st.title('Predition single music note')
wav = st.file_uploader("upload music file", type=['wav', 'mp3'])
submit = st.button('Predict')

if submit:
    prediction = predict(model,wav)
    st.write("your input note is ",prediction)
