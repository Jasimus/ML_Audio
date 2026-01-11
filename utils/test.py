import numpy as np
import tensorflow as tf
from dataset.AudioData import create_sequences
import soundfile as sf

def generate_blocks(model, hp):
  sample_test, sr = load_and_process_to_test(hp.test.sample_src)

  OW = hp.data.tar_size
  input_window = hp.data.win_size
  n_steps = hp.test.n_blocks

  initial_context = sample_test[-input_window:]
  overlap = OW // 2

  # Ventana Hann para OLA
  hann = np.hanning(OW)
  w_up = hann[:overlap]
  w_down = hann[overlap:]

  # Inicialización del buffer final
  final_spec = None

  # Tu contexto inicial (X0)
  context = tf.expand_dims(initial_context, axis=0)     # shape = (1, input_window, n_frec*2)

  for _ in range(n_steps):   # generar n bloques

      block = model.predict(context)    # (1, OW, n_frec*2)
      if final_spec is None:
          # Primer bloque: se copia entero
          final_spec = block.copy()[0]
      else:
          # Overlap-add
          T = final_spec.shape[0]

          # Zona solapada en final_spec: [T-overlap : T]
          # Zona solapada en block:       [0 : overlap]
          # Ponderación
          final_spec[T-overlap:T] = (
              final_spec[T-overlap:T] * w_down[1:, None] +
              block[:,:overlap]          * w_up[:, None]
          )

          # Parte no solapada se concatena normalmente
          final_spec = np.concatenate([
              final_spec,
              block[0,overlap:]
          ], axis=0)

      # Actualización del contexto para la siguiente predicción
      # Tomamos los últimos input_window frames del espectrograma final
      if final_spec.shape[0] >= input_window:
          context = tf.expand_dims(final_spec[-input_window:], axis=0)
      else:
          # Hasta que haya suficiente contexto (solo para las primeras iteraciones)
          pad = input_window - final_spec.shape[0]
          context = np.vstack([np.zeros((pad, final_spec.shape[1])), final_spec])
          context = tf.expand_dims(context, axis=0)

  return final_spec, sr


def data_to_spec(pred):
    _, n_frec_2 = pred.shape
    
    p_r = tf.constant(pred[:,:n_frec_2//2])
    p_i = tf.constant(pred[:,-n_frec_2//2:])

    return tf.complex(p_r, p_i)


def stft_to_signal(stft, hp):
    signal = tf.signal.inverse_stft(stft, frame_length=hp.data.nfft, frame_step=hp.data.hop)
    return signal


def signal_to_stft(self, signal):
        spec = tf.signal.stft(signal, frame_length=self.nfft, frame_step=self.hop)
        return spec


def load_audio_to_test(path):
    audio, sr = sf.read(path)
    signal = audio.astype(np.float32)
    audio_shape = signal.shape

    if len(audio_shape) > 1:
        signal = np.mean(signal, axis=1)

    max_val = np.max(signal)
    return signal/max_val, sr


def load_and_process_to_test(path, shift=1):
    
    signal, sr = load_audio_to_test(path)
    stft_spec = signal_to_stft(signal)

    return create_sequences(stft_spec, shift), sr
