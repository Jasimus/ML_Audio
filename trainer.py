from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from .model import build_model
from utils.hparams import HParam
from dataset.AudioData import AudioData
import os


## loading hyperparameters
hp = HParam("config/default.yaml")

## loading data
audio_data = AudioData(hp)
train_ds, _ = audio_data.create_train_test()


es_cb = EarlyStopping(monitor='loss', patience=3)
os.makedirs("checkpoints", exist_ok=True)

if not os.listdir('checkpoints'):
    best_model = build_model(audio_data.win_size, audio_data.tar_size, audio_data.n_frec_2, hp.model.n_cells, hp.train.lr, hp.model.hl)

print("checkpoints will be stored in 'checkpoints' directory")

checkpoint_cb = ModelCheckpoint(
    filepath="checkpoints/weights.{epoch:02d}-{loss:.4f}.weights.h5",
    save_weights_only=True,
    save_best_only=False,
    monitor="loss",
    mode="min",
    save_freq=3000,
    verbose=1
)

best_model.fit(train_ds, callbacks=[es_cb, checkpoint_cb], epochs=hp.train.epochs)