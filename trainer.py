#!/usr/bin/env python3

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from model.models import build_model, take_model
from utils.hparams import HParam
from dataset.AudioData import AudioData
import os

CONFIG = "default.yaml"

## loading hyperparameters
config_path = os.path.join("config", CONFIG)
hp = HParam(config_path)

CKP_DIR = hp.train.checkpoints

## loading data
audio_data = AudioData(hp)
train_ds, test_ds = audio_data.create_train_test()

os.makedirs(CKP_DIR, exist_ok=True)

ckp = take_model(CKP_DIR, hp.model.name)

if not ckp:
    best_model = build_model(audio_data.win_size,
                             audio_data.tar_size,
                             audio_data.n_frec_2,
                             hp.model.n_cells,
                             hp.train.lr,
                             hp.model.hl)

else:
    best_model = load_model(f"{CKP_DIR}/{ckp}")

print(f"checkpoints will be stored in {CKP_DIR}")

es_cb = EarlyStopping(monitor='loss', patience=3)

checkpoint_cb = ModelCheckpoint(
    filepath=f"{CKP_DIR}/{hp.model.name}.keras",
    monitor="loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

print(train_ds.cardinality().numpy())

best_model.fit(train_ds,
               callbacks=[es_cb, checkpoint_cb],
               epochs=hp.train.epochs)


if hp.data.frac != 1:
    loss, acc = best_model.evaluate(test_ds, steps=50)
    print("final loss: %.3f" % loss)
    print("final accurate: %.3f" % acc)