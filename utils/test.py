import numpy as np
import tensorflow as tf
from dataset.AudioData import AudioData


def generate_blocks(model, hp, initial_context):
  OW = hp.data.tar_size
  input_window = hp.data.win_size
  n_steps = hp.test.n_blocks

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

  return final_spec


def data_to_spec(pred):
    _, n_frec_2 = pred.shape
    
    p_r = tf.constant(pred[:,:n_frec_2//2])
    p_i = tf.constant(pred[:,-n_frec_2//2:])

    return tf.complex(p_r, p_i)


def stft_to_signal(stft, nfft, hop):
        signal = tf.signal.inverse_stft(stft, frame_length=nfft, frame_step=hop)
        return signal

