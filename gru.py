from typing import List, Dict

import numpy as np
from keras import layers, models

from constants import *
from helper import check_unique_patterns
from preprocess import equally_spaced_points_patterns, is_inside_box
from ujipen.ujipen_class import UJIPen


def concat_samples(samples: Dict[str, List[List[np.ndarray]]]):
    labels = []
    data = []
    for letter in samples.keys():
        letter_ord = ord(letter) - ord('a')
        labels.extend([letter_ord] * len(samples[letter]))
        for word_sample in samples[letter]:
            word_sample = np.vstack(word_sample)
            data.append(word_sample)
    data = np.stack(data, axis=0)
    assert is_inside_box(data, box=((-1, -1), (1, 1)))
    labels = np.array(labels)
    print(f"Data: {data.shape}, labels: {labels.shape}")
    return data, labels


def train(ujipen: UJIPen, n_input=PATTERN_SIZE, n_hidden=50):
    patterns = ujipen.get_samples(fold='train')
    patterns = equally_spaced_points_patterns(patterns, total_points=n_input)
    train_data, train_labels = concat_samples(patterns)
    test_samples = equally_spaced_points_patterns(ujipen.get_samples(fold='test'), total_points=n_input)
    test_data, test_labels = concat_samples(test_samples)
    assert check_unique_patterns(patterns, n_points=n_input)
    gru = models.Sequential()
    gru.add(layers.GRU(units=n_hidden, activation='tanh', recurrent_activation='hard_sigmoid',
                       return_sequences=False, implementation=1,
                       input_shape=(n_input, 2)))
    gru.add(layers.Dense(units=np.unique(train_labels).size, activation='softmax'))

    print(gru.summary())
    gru.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    history = gru.fit(train_data, train_labels, epochs=100, batch_size=32, validation_data=(test_data, test_labels),
                      verbose=0)
    history = history.history
    accuracy_train = history['acc'][-1]
    print(f"Loss: {history['loss'][-1]:.5f}, accuracy: train={accuracy_train:.5f}, val={history['val_acc'][-1]:.5f}")
    MODELS_DIR.mkdir(exist_ok=True)
    model_path = str(MODELS_DIR / f'GRU_input-{n_input}_hidden-{n_hidden}_acc-{accuracy_train:.4f}.h5')
    gru.save(model_path)
    print(f"Saved trained model to {model_path}")


if __name__ == '__main__':
    train(ujipen=UJIPen(), n_input=30, n_hidden=100)
