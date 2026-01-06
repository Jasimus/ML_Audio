import numpy as np
import soundfile as sf
import tensorflow as tf
import warnings
import math
import os
import glob

class AudioData():
    def __init__(self, hp):
        self.nfft = hp.data.nfft
        self.hop = hp.data.hop
        self.win_size = hp.data.win_size
        self.tar_size = hp.data.tar_size
        self.batch_size = hp.data.batch_size
        self.frac = hp.data.frac
        self.ds_size = None

        wav_files = os.path.join(hp.data.src, '*.wav')
        print("ruta de wavs",wav_files)
        self.paths = glob.glob(wav_files, recursive=True)

        if hp.data.max_nframes:
            self.max_nframes = hp.data.max_nframes

        else:
            self.max_nframes = self.count_frames()        
            ## falta escribir max_nframes en el yaml

        self.t_frames = self.max_nframes // self.hop
        
        if hp.data.ds_size:
            self.ds_size = hp.data.ds_size
        
        self.n_frec_2 = (self.nfft//2 + 1) * 2


    def create_train_test(self):
        ds = self.create_dataset()
        if not self.ds_size:
            self.ds_size = self.count_dataset(ds)
            warnings.warn("warning: you must to set hp.data.ds_size value")

        tr_size = math.floor(self.frac * self.ds_size)

        return ds.take(tr_size), ds.skip(tr_size)


    def create_dataset(self):
        ds = tf.data.Dataset.from_tensor_slices(self.paths)
        ds = ds.flat_map(lambda p: self.load_and_process(p))
        ds = ds.shuffle(1000)
        ds = ds.batch(self.batch_size)
        ds = ds.repeat()
        ds = ds.prefetch(tf.data.AUTOTUNE)

        return ds

    def count_frames(self):
        max = 0
        for path in self.paths:
            audio, _ = sf.read(path)
            if 1 < len(audio.shape):
                n_frames = audio.shape[1]
            else:
                n_frames, = audio.shape

            if max < n_frames:
                max = n_frames

        return max


    def load_audio(self, file_path):
        path = file_path.numpy().decode('utf-8')
        audio, sr = sf.read(path)
        signal = audio.astype(np.float32)
        audio_shape = signal.shape

        if len(audio_shape) > 1:
            signal = np.mean(signal, axis=1)

        max_val = np.max(signal)
        return signal/max_val


    def signal_to_stft(self, signal):
        spec = tf.signal.stft(signal, frame_length=self.nfft, frame_step=self.hop)
        return spec


    def load_and_process(self, path, shift=1):

        signal, = tf.py_function(func=self.load_audio, inp=[path], Tout=[tf.float32])
        signal.set_shape([None])
        stft_spec = self.signal_to_stft(signal)
        current_frames = tf.shape(stft_spec)[0]
        padding_a = tf.maximum(0, self.t_frames - current_frames)

        padded_spec = tf.pad(
            stft_spec,
            paddings=[[0, padding_a], [0, 0]],
            mode='CONSTANT'
        )

        return tf.data.Dataset.from_tensor_slices(self.create_sequences(padded_spec, shift))


    def load_audio_to_test(self, path):
        audio, sr = sf.read(path)
        signal = audio.astype(np.float32)
        audio_shape = signal.shape

        if len(audio_shape) > 1:
            signal = np.mean(signal, axis=1)

        max_val = np.max(signal)
        return signal/max_val


    def load_and_process_to_test(self, path, shift=1):
        signal = self.load_audio_to_test(path)
        stft_spec = self.signal_to_stft(signal)
        current_frames = tf.shape(stft_spec)[0]
        padding_a = tf.maximum(0, self.t_frames - current_frames)

        padded_spec = tf.pad(
            stft_spec,
            paddings=[[0, padding_a], [0, 0]],
            mode='CONSTANT'
        )

        return self.create_sequences(padded_spec, shift)


    def create_sequences(self, spec, shift=1):

        spec_re = tf.math.real(spec)
        spec_im = tf.math.imag(spec)

        X_win_re = tf.signal.frame(
            spec_re,
            frame_length=self.win_size,
            frame_step=shift,
            pad_end=False,
            axis=0
        )

        X_win_im = tf.signal.frame(
            spec_im,
            frame_length=self.win_size,
            frame_step=shift,
            pad_end=False,
            axis=0
        )

        y_spec_re = spec_re[self.win_size:]
        y_spec_im = spec_im[self.win_size:]

        y_win_re = tf.signal.frame(
            y_spec_re,
            frame_length=self.tar_size,
            frame_step=shift,
            pad_end=False,
            axis=0
        )

        y_win_im = tf.signal.frame(
            y_spec_im,
            frame_length=self.tar_size,
            frame_step=shift,
            pad_end=False,
            axis=0
        )

        X_win = tf.concat([X_win_re, X_win_im], axis=-1)
        y_win = tf.concat([y_win_re, y_win_im], axis=-1)

        min_seq = tf.minimum(tf.shape(X_win)[0],tf.shape(y_win)[0])

        X_seq = X_win[:min_seq]
        y_seq = y_win[:min_seq]

        return (X_seq, y_seq)
    

    def count_dataset(self, ds):
        count = 0
        for _ in ds:
            count += 1
        return count