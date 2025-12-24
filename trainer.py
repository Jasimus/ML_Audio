from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from .model.models import build_model, take_model
from utils.hparams import HParam
from dataset.AudioData import AudioData
import os

CKP_DIR = "checkpoints"
CONFIG = "dafault.yaml"

## loading hyperparameters
hp = HParam(f"config/{CONFIG}")

## loading data
audio_data = AudioData(hp)
train_ds, _ = audio_data.create_train_test()

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
    monitor="val_loss",
    save_best_only=True,
    save_weights_only=False,
    verbose=1
)

best_model.fit(train_ds,
               callbacks=[es_cb, checkpoint_cb],
               epochs=hp.train.epochs,
               validation_split=0.2)