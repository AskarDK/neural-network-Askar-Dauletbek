#PR 1 DAULETBEK ASKAR
"""
Реализация нейронной сети, вычисляющей логические функции (варианты 1..8).

Скрипт:
- формирует все 8 входных комбинаций (a,b,c)
- для каждого варианта:
  - строит простую нейросеть (Dense 4 -> Dense 1, sigmoid)
  - выводит веса до обучения
  - прогоняет датасет через keras-модель и через 2 реализованные вручную функции:
      1) elementwise_forward(X, weights) — поэлементно (циклы)
      2) numpy_forward(X, weights) — векторно (NumPy)
  - сравнивает результаты (max abs diff и совпадение при пороге 0.5)
  - обучает модель на всем наборе (без валидации)
  - снова получает веса, прогоняет и сравнивает результаты после обучения
- печатает результаты сравнения в терминал

Переформатировано без изменения логики, параметров и выходов.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --- фиксация сидов для воспроизводимости ---
np.random.seed(42)
tf.random.set_seed(42)


def variant_fn(variant):
    """
    Каждая функция принимает массивы a, b, c (0/1 scalar или numpy arrays) и возвращает 0/1.
    """
    if variant == 1:
        # (a and b) or (a and c)
        return lambda a, b, c: (a & b) | (a & c)
    if variant == 2:
        # (a or b) xor not(b and c)
        return lambda a, b, c: ((a | b) ^ (~(b & c) & 1))
    if variant == 3:
        # (a and b) or c
        return lambda a, b, c: (a & b) | c
    if variant == 4:
        # (a or b) and (b or c)
        return lambda a, b, c: (a | b) & (b | c)
    if variant == 5:
        # (a xor b) and (b xor c)
        return lambda a, b, c: ((a ^ b) & (b ^ c))
    if variant == 6:
        # (a and not b) or (c xor b)
        return lambda a, b, c: ((a & (~b & 1)) | (c ^ b))
    if variant == 7:
        # (a or b) and (a xor not b)
        return lambda a, b, c: ((a | b) & (a ^ (~b & 1)))
    if variant == 8:
        # (a and c and b) xor (a or not b)
        return lambda a, b, c: ((a & b & c) ^ (a | (~b & 1)))
    raise ValueError("Unknown variant")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def numpy_forward(X, weights):
    """
    Векторный проход через сеть с использованием NumPy.
    X: np.array shape (n_samples, input_dim)
    weights: список [W1, b1, W2, b2] как у model.get_weights()
    Возвращает: probs (n_samples,) — вероятности выхода (sigmoid)
    """
    W1, b1, W2, b2 = weights

    # Приведение типов к float32 для близости к Keras
    Xf = X.astype(np.float32)
    W1f = W1.astype(np.float32)
    b1f = b1.astype(np.float32)
    W2f = W2.astype(np.float32)
    b2f = b2.astype(np.float32)

    z1 = np.dot(Xf, W1f) + b1f          # (n_samples, units)
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2f) + b2f          # (n_samples, 1)
    a2 = sigmoid(z2)
    return a2.reshape(-1)


def elementwise_forward(X, weights):
    """
    Поэлементный проход (без матричных операций).
    X: np.array shape (n_samples, input_dim)
    weights: [W1, b1, W2, b2]
    Возвращает probs (n_samples,)
    """
    W1, b1, W2, b2 = weights

    # Приведение типов к float32 для сопоставимости
    Xf = X.astype(np.float32)
    W1f = W1.astype(np.float32)
    b1f = b1.astype(np.float32)
    W2f = W2.astype(np.float32)
    b2f = b2.astype(np.float32)

    n_samples = Xf.shape[0]
    units = W1f.shape[1]
    out = np.zeros((n_samples,), dtype=np.float32)

    for i in range(n_samples):
        x = Xf[i]

        # слой 1
        a1 = np.zeros((units,), dtype=np.float32)
        for j in range(units):
            s = np.float32(0.0)
            for k in range(x.shape[0]):
                s += x[k] * W1f[k, j]
            s += b1f[j]
            a1[j] = 1.0 / (1.0 + np.exp(-s))  # sigmoid

        # слой 2 (выход)
        s2 = np.float32(0.0)
        for j in range(units):
            s2 += a1[j] * W2f[j, 0]
        s2 += b2f[0]
        out[i] = 1.0 / (1.0 + np.exp(-s2))

    return out


# ----- утилиты -----
def build_model(input_dim=3, hidden_units=4):
    """
    Простая MLP: Dense(hidden_units, sigmoid) -> Dense(1, sigmoid).
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(hidden_units, activation="sigmoid"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.5),
                  loss="binary_crossentropy")
    return model


def threshold_preds(probs, thresh=0.5):
    """Бинаризация вероятностей по порогу."""
    return (probs >= thresh).astype(int)


def run_all_variants():
    # Все 8 комбинаций входов (a,b,c)
    X = np.array([
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ], dtype=np.int32)

    variants = list(range(1, 9))
    for var in variants:
        print("=" * 80)
        print(f"Вариант {var}")

        fn = variant_fn(var)
        a, b, c = X[:, 0], X[:, 1], X[:, 2]
        y = fn(a, b, c).astype(np.int32)  # shape (8,)

        print("Входы (a,b,c):")
        print(X)
        print("Целевая логическая функция - y:")
        print(y)

        # Построение модели и инициализация весов (через первый predict)
        model = build_model(input_dim=3, hidden_units=4)
        _ = model.predict(X, verbose=0)
        initial_weights = model.get_weights()

        print("\nВесовые матрицы и смещения (до обучения):")
        for idx, w in enumerate(initial_weights):
            print(f"weights[{idx}] shape={w.shape}")

        # Предсказания ДО обучения
        model_preds_before = model.predict(X, verbose=0).reshape(-1)
        numpy_preds_before = numpy_forward(X, initial_weights)
        element_preds_before = elementwise_forward(X, initial_weights)

        print("\nРезультаты ДО обучения (probabilities):")
        print("model:", np.round(model_preds_before, 6))
        print("numpy:", np.round(numpy_preds_before, 6))
        print("elem :", np.round(element_preds_before, 6))

        diff_np = np.max(np.abs(model_preds_before - numpy_preds_before))
        diff_el = np.max(np.abs(model_preds_before - element_preds_before))
        print(f"\nМакс |model - numpy| до обучения: {diff_np:.6e}")
        print(f"Макс |model - elementwise| до обучения: {diff_el:.6e}")
        print("Совпадают model vs numpy (tol 1e-6):",
              np.allclose(model_preds_before, numpy_preds_before, atol=1e-6))
        print("Совпадают model vs elementwise (tol 1e-6):",
              np.allclose(model_preds_before, element_preds_before, atol=1e-6))

        # Бинарные предсказания ДО обучения
        model_bin_before = threshold_preds(model_preds_before)
        numpy_bin_before = threshold_preds(numpy_preds_before)
        elem_bin_before = threshold_preds(element_preds_before)
        print("\nБинарные предсказания (порог 0.5) ДО обучения:")
        print("model:", model_bin_before)
        print("numpy:", numpy_bin_before)
        print("elem :", elem_bin_before)
        print("Совпадение бинарных model vs numpy:",
              np.array_equal(model_bin_before, numpy_bin_before))
        print("Совпадение бинарных model vs elem :",
              np.array_equal(model_bin_before, elem_bin_before))

        # Обучение модели на всем датасете
        print("\nОбучение модели на всем датасете...")
        model.fit(X.astype(np.float32), y.astype(np.float32),
                  epochs=2000, batch_size=8, verbose=0)

        # Веса и предсказания ПОСЛЕ обучения
        trained_weights = model.get_weights()
        print("Весовые матрицы и смещения (после обучения):")
        for idx, w in enumerate(trained_weights):
            print(f"weights[{idx}] shape={w.shape}")

        model_preds_after = model.predict(X, verbose=0).reshape(-1)
        numpy_preds_after = numpy_forward(X, trained_weights)
        element_preds_after = elementwise_forward(X, trained_weights)

        print("\nРезультаты ПОСЛЕ обучения (probabilities):")
        print("model:", np.round(model_preds_after, 6))
        print("numpy:", np.round(numpy_preds_after, 6))
        print("elem :", np.round(element_preds_after, 6))

        diff_np_after = np.max(np.abs(model_preds_after - numpy_preds_after))
        diff_el_after = np.max(np.abs(model_preds_after - element_preds_after))
        print(f"\nМакс |model - numpy| после обучения: {diff_np_after:.6e}")
        print(f"Макс |model - elementwise| после обучения: {diff_el_after:.6e}")
        print("Совпадают model vs numpy (tol 1e-6):",
              np.allclose(model_preds_after, numpy_preds_after, atol=1e-6))
        print("Совпадают model vs elementwise (tol 1e-6):",
              np.allclose(model_preds_after, element_preds_after, atol=1e-6))

        # Бинарные предсказания ПОСЛЕ обучения
        model_bin_after = threshold_preds(model_preds_after)
        numpy_bin_after = threshold_preds(numpy_preds_after)
        elem_bin_after = threshold_preds(element_preds_after)
        print("\nБинарные предсказания (порог 0.5) ПОСЛЕ обучения:")
        print("model:", model_bin_after)
        print("target:", y)
        print("numpy:", numpy_bin_after)
        print("elem :", elem_bin_after)
        print("model == target (после обучения):", np.array_equal(model_bin_after, y))
        print("Совпадение бинарных model vs numpy:",
              np.array_equal(model_bin_after, numpy_bin_after))
        print("Совпадение бинарных model vs elem :",
              np.array_equal(model_bin_after, elem_bin_after))

        print("\n\n")  # разделитель для читаемости


if __name__ == "__main__":
    run_all_variants()
