import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from model import build_model


def test_build_model_output_shape():
    model = build_model(img_size=(224, 224), num_classes=3)
    assert model.output_shape == (None, 3)


def test_forward_pass_output_shape_and_probabilities():
    model = build_model(img_size=(224, 224), num_classes=3)
    batch = tf.random.uniform((2, 224, 224, 3), dtype=tf.float32)
    probs = model(batch, training=False).numpy()

    assert probs.shape == (2, 3)
    np.testing.assert_allclose(probs.sum(axis=1), np.ones(2), rtol=1e-5, atol=1e-5)