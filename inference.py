#!/usr/bin/env python3

from utils.test import data_to_spec, generate_blocks, stft_to_signal
from tensorflow.keras.models import load_model
from utils.hparams import HParam
from model.models import take_model
import soundfile as sf
import os

CONFIG = "default.yaml"

## loading hyperparameters
config_path = os.path.join("config", CONFIG)
hp = HParam(config_path)

CKP_DIR = hp.train.checkpoints
ckp = take_model(CKP_DIR, hp.model.name)

ckp_path = os.path.join(CKP_DIR, ckp)

try:
    best_model = load_model(ckp_path)

except Exception as err:
    print(f"didn't find any model in {CKP_DIR}", err)


pred, sr = generate_blocks(best_model, hp)
spec = data_to_spec(pred)
signal = stft_to_signal(spec, hp)

file_name = hp.test.sample_src + "_test.wav"

sf.write(file_name, signal, sr)

print(f"inference stored in {file_name}")