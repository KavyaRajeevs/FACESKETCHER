"""
Sketch → photorealistic face using a Keras SavedModel in this directory.

CLI (legacy): opens a file dialog, saves outputs, shows matplotlib.
Library: use convert_sketch_bytes() from the HTTP server or other code.
"""
from __future__ import annotations

import os
from datetime import datetime
from io import BytesIO

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow_addons.layers import InstanceNormalization

# SavedModel directory (this folder, or override for a full checkpoint elsewhere)
MODEL_DIR = os.environ.get(
    "SKETCH2REAL_MODEL_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)

_g_model = None

import tensorflow as tf

class InstanceNormalization(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(
            shape=(input_shape[-1],),
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            shape=(input_shape[-1],),
            initializer="zeros",
            trainable=True,
        )

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        return self.gamma * (inputs - mean) / tf.sqrt(variance + self.epsilon) + self.beta
    


def get_generator_model():
    global _g_model
    if _g_model is None:
        _g_model = tf.keras.layers.TFSMLayer(
            MODEL_DIR,
            call_endpoint="serving_default"
        )
    return _g_model


def _image_score(img: np.ndarray) -> float:
    return float(np.std(img))


def sketch_array_to_best_real(norm_img: np.ndarray, num_samples: int = 6) -> np.ndarray:
    """
    norm_img: float array in [-1, 1], shape (128, 128, 3).
    Returns best RGB uint8 image (H, W, 3).
    """
    g_model = get_generator_model()
    generated_images = []

    for _ in range(num_samples):
        noise = np.random.normal(0, 0.05, norm_img.shape)
        noisy_input = np.clip(norm_img + noise, -1, 1)
        g_img = g_model(tf.convert_to_tensor(np.expand_dims(noisy_input, 0)))
        g_img = list(g_img.values())[0].numpy()[0]
        g_img = (g_img + 1) * 127.5
        g_img = np.clip(g_img, 0, 255).astype("uint8")
        generated_images.append(g_img)

    generated_images = sorted(
        generated_images, key=_image_score, reverse=True
    )
    return generated_images[0]


def convert_sketch_bytes(image_bytes: bytes, num_samples: int = 6) -> bytes:
    """
    Load a sketch image from raw bytes (PNG/JPEG), run the generator, return JPEG bytes.
    """
    img = load_img(BytesIO(image_bytes), target_size=(128, 128))
    arr = img_to_array(img)
    norm_img = (arr - 127.5) / 127.5
    best_rgb = sketch_array_to_best_real(norm_img, num_samples=num_samples)
    out = BytesIO()
    Image.fromarray(best_rgb).save(out, format="JPEG", quality=92)
    return out.getvalue()


def _run_cli():
    # import matplotlib.pyplot as plt
    from tkinter import Tk, filedialog

    Tk().withdraw()
    file_path = filedialog.askopenfilename(title="Select Sketch Image")
    if not file_path:
        print("No file selected")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"output_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

    with open(file_path, "rb") as f:
        data = f.read()

    img = load_img(BytesIO(data), target_size=(128, 128))
    arr = img_to_array(img)
    norm_img = (arr - 127.5) / 127.5

    generated_images = []
    g_model = get_generator_model()
    for _ in range(6):
        noise = np.random.normal(0, 0.05, norm_img.shape)
        noisy_input = np.clip(norm_img + noise, -1, 1)
        g_img = g_model(tf.convert_to_tensor(np.expand_dims(noisy_input, 0)))
        g_img = list(g_img.values())[0].numpy()[0]
        g_img = (g_img + 1) * 127.5
        g_img = np.clip(g_img, 0, 255).astype("uint8")
        generated_images.append(g_img)

    generated_images = sorted(
        generated_images, key=_image_score, reverse=True
    )
    best_3 = generated_images[:3]

    for i, img_out in enumerate(best_3):
        save_path = os.path.join(output_folder, f"output_{i + 1}.jpg")
        Image.fromarray(img_out).save(save_path, format="JPEG", quality=92)

    print(f"Saved 3 best images in folder: {output_folder}")

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 4, 1)
    # plt.imshow(arr.astype("uint8"))
    # plt.title("Input")
    # plt.axis("off")
    # for i in range(3):
    #     plt.subplot(1, 4, i + 2)
    #     plt.imshow(best_3[i])
    #     plt.title(f"Output {i + 1}")
    #     plt.axis("off")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    _run_cli()
