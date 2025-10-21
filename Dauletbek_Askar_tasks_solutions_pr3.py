#PR 1 DAULETBEK ASKAR
"""
- Все входные данные читаются из .npy/.npz файлов (поиск в рабочей папке practice3_sample_inputs).
- Результаты сохраняются в папку practice3_results.

Переформатированная версия (PEP 8, типизация, единообразные сообщения) без изменения логики,
пути, имён функций и форматов сохраняемых файлов.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # рендер без GUI
import matplotlib.pyplot as plt


# --- Константы путей ---
RESULT_DIR: str = "practice3_results"
INPUT_DIRS = ["practice3_sample_inputs"]

os.makedirs(RESULT_DIR, exist_ok=True)


# --- Вспомогательные утилиты ввода-вывода ---
def find_input_path(filename: str) -> Optional[str]:
    """Ищет файл по списку INPUT_DIRS и возвращает первый найденный путь или None."""
    for directory in INPUT_DIRS:
        path = os.path.join(directory, filename) if directory else filename
        if os.path.exists(path):
            return path
    return None


def try_load(path: str):
    """Безопасная загрузка .npy/.npz; возвращает None, если файла нет."""
    if not os.path.exists(path):
        return None
    return np.load(path, allow_pickle=True)


# --- Задача 1 ---
# Сумма произведений множества матриц на соответствующие векторы.
# Вход: matrices (p, n, n) и vectors (p, n) или (p, n, 1)
def sum_matrix_vector_products(matrices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    matrices: np.ndarray shape (p, n, n)
    vectors:  np.ndarray shape (p, n) или (p, n, 1)
    Возвращает np.ndarray shape (n, 1)
    """
    matrices = np.asarray(matrices)
    vectors = np.asarray(vectors)

    # Приведение (p, n, 1) -> (p, n)
    if vectors.ndim == 3 and vectors.shape[2] == 1:
        vectors = vectors.reshape(vectors.shape[0], vectors.shape[1])

    if matrices.ndim != 3:
        raise ValueError("matrices должен быть массива размерности (p, n, n)")

    p, n, _ = matrices.shape
    if not (vectors.shape[0] == p and vectors.shape[1] == n):
        raise ValueError("несовпадение форм matrices и vectors")

    # Суммируем произведения
    res = np.zeros((n,), dtype=np.result_type(matrices, vectors))
    for i in range(p):
        res += matrices[i].dot(vectors[i])
    return res.reshape(n, 1)


# --- Задача 2 ---
# Преобразование вектора целых чисел в матрицу бинарных представлений
def vector_to_binary_matrix(vector: np.ndarray, bit_order: str = "msb") -> np.ndarray:
    """
    vector:   1D массив неотрицательных целых чисел
    Возврат:  бинарная матрица shape (len(vector), n_bits)
    bit_order: 'msb' (по умолчанию) или 'lsb'
    """
    vec = np.asarray(vector).astype(np.int64).flatten()
    if vec.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    if np.any(vec < 0):
        raise ValueError("отрицательные числа не поддерживаются")

    max_val = int(vec.max())
    n_bits = max(1, max_val.bit_length())

    # Формируем битовую матрицу (msb -> lsb по умолчанию)
    shifts = np.arange(n_bits - 1, -1, -1)
    bits = ((vec.reshape(-1, 1) >> shifts) & 1).astype(np.uint8)

    if bit_order == "lsb":
        bits = bits[:, ::-1]

    return bits


# --- Задача 3 ---
# Возвращает все уникальные строки матрицы
def unique_rows(matrix: np.ndarray) -> np.ndarray:
    """
    matrix: 2D массив
    Возвращает массив с уникальными строками (лексикографический порядок)
    """
    arr = np.asarray(matrix)
    return np.unique(arr, axis=0)


# --- Задача 4 ---
# Заполнение матрицы (M,N) случайными нормальными числами.
# Вычисление математического ожидания и дисперсии по столбцам.
# Сохранение гистограмм для каждой строки.
def random_normal_matrix_and_stats(
    M: int,
    N: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    M: число строк
    N: число столбцов
    seed: для воспроизводимости

    Возвращает (means, variances) формы (N,), (N,)
    Также сохраняет гистограммы по строкам в RESULT_DIR.
    """
    rng = np.random.RandomState(seed)
    mat = rng.randn(M, N)

    means = mat.mean(axis=0)
    variances = mat.var(axis=0, ddof=0)

    # Сохраняем гистограммы для каждой строки
    for i in range(M):
        plt.figure()
        plt.hist(mat[i], bins="auto", color="C0", alpha=0.7)
        plt.title(f"Гистограмма значений строки {i}")
        plt.xlabel("Значение")
        plt.ylabel("Частота")
        plt.savefig(os.path.join(RESULT_DIR, f"task4_hist_row_{i}.png"))
        plt.close()

    # Сохраняем саму матрицу для отладки
    np.save(os.path.join(RESULT_DIR, "task4_random_normal_matrix.npy"), mat)
    return means, variances


# --- Задача 5 ---
# Заполнить матрицу (M,N) в шахматном порядке числами a и b
def checkerboard_matrix(M: int, N: int, a, b) -> np.ndarray:
    """
    Возвращает матрицу размером (M,N) с элементами a и b в шахматном порядке.
    Верхний левый элемент (0,0) равен a.
    """
    idx = np.add.outer(np.arange(M), np.arange(N))
    return np.where((idx % 2) == 0, a, b)


# --- Задача 6 ---
# Вернуть тензор (H,W,3) изображения круга заданного радиуса и цвета на черном фоне
def circle_image_tensor(height: int, width: int, radius: int, color) -> np.ndarray:
    """
    height, width: размеры изображения
    radius: радиус круга
    color: iterable из 3 значений (R,G,B) 0..255
    Возвращает uint8 тензор (height, width, 3)
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    y = np.arange(height)[:, None]
    x = np.arange(width)[None, :]
    mask = (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2
    color_arr = np.array(color, dtype=np.uint8).reshape(1, 1, 3)
    img[mask] = color_arr
    return img


# --- Задача 7 ---
# Стандартизировать тензор: (x - mean) / std
def standardize_tensor(tensor: np.ndarray) -> np.ndarray:
    """
    Стандартизует все элементы тензора по общей средней и СКО.
    Если СКО == 0, возвращает массив нулей той же формы.
    """
    arr = np.asarray(tensor, dtype=np.float64)
    mean = arr.mean()
    std = arr.std(ddof=0)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


# --- Задача 8 ---
# Выделение патча фиксированного размера с центром в данном элементе.
# При выходе за границы — дополняем значением fill.
def extract_centered_patch(
    matrix: np.ndarray,
    center_row: int,
    center_col: int,
    patch_rows: int,
    patch_cols: int,
    fill=0,
) -> np.ndarray:
    """
    matrix: 2D или многомерный массив (если каналов больше — патч по первым двум осям)
    center_row, center_col: координаты центра патча
    patch_rows, patch_cols: размеры патча
    fill: значение для дополнения при выходе за границы
    """
    mat = np.asarray(matrix)
    pr, pc = patch_rows, patch_cols
    half_r, half_c = pr // 2, pc // 2

    top, left = center_row - half_r, center_col - half_c
    bottom, right = top + pr, left + pc

    pad_top = max(0, -top)
    pad_left = max(0, -left)
    pad_bottom = max(0, bottom - mat.shape[0])
    pad_right = max(0, right - mat.shape[1])

    if any((pad_top, pad_bottom, pad_left, pad_right)):
        pad_width = ((pad_top, pad_bottom), (pad_left, pad_right))
        if mat.ndim > 2:
            pad_width += tuple((0, 0) for _ in range(mat.ndim - 2))
        mat_padded = np.pad(mat, pad_width, mode="constant", constant_values=fill)
        top_adj = top + pad_top
        left_adj = left + pad_left
        patch = mat_padded[top_adj : top_adj + pr, left_adj : left_adj + pc]
    else:
        patch = mat[top:bottom, left:right]

    return patch


# --- Задача 9 ---
# Находит наиболее частое число в каждой строке матрицы
def row_modes(matrix: np.ndarray) -> np.ndarray:
    """
    Для каждой строки возвращает наиболее часто встречающееся значение.
    При нескольких модах возвращает наименьшее для детерминированности.
    """
    mat = np.asarray(matrix)
    modes = []
    for row in mat:
        vals, counts = np.unique(row, return_counts=True)
        max_count = counts.max()
        mode_candidates = vals[counts == max_count]
        modes.append(mode_candidates.min())
    return np.array(modes)


# --- Задача 10 ---
# Сумма каналов изображения с указанными весами -> (height, width)
def weighted_sum_channels(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    image:   (height, width, numChannels)
    weights: (numChannels,)
    Возвращает матрицу (height, width) — взвешенная сумма каналов.
    """
    img = np.asarray(image)
    w = np.asarray(weights)

    if img.ndim != 3:
        raise ValueError("image должен быть 3D массивом (H,W,C)")
    if img.shape[2] != w.shape[0]:
        raise ValueError("число каналов изображения и длина вектора весов не совпадают")

    return np.tensordot(img, w, axes=([2], [0]))


# --- Основная логика ---
def main() -> None:
    inpath = find_input_path  # alias для компактности

    # Задача 1
    mpath = inpath("task1_matrices.npy")
    vpath = inpath("task1_vectors.npy")
    if mpath and vpath:
        matrices = np.load(mpath, allow_pickle=True)
        vectors = np.load(vpath, allow_pickle=True)
        res1 = sum_matrix_vector_products(matrices, vectors)
        np.save(os.path.join(RESULT_DIR, "task1_sum_matrix_vector_products.npy"), res1)
        print("Задача 1: выполнено. Результат сохранён в practice3_results/task1_sum_matrix_vector_products.npy")
    else:
        print("Задача 1: входные файлы не найдены. Пропуск.")

    # Задача 2
    p2 = inpath("task2_vector.npy")
    if p2:
        vec = np.load(p2, allow_pickle=True)
        binmat = vector_to_binary_matrix(vec)
        np.save(os.path.join(RESULT_DIR, "task2_vector_binary_matrix.npy"), binmat)
        print("Задача 2: выполнено. Бинарная матрица сохранена в practice3_results/task2_vector_binary_matrix.npy")
    else:
        print("Задача 2: входной файл не найден. Пропуск.")

    # Задача 3
    p3 = inpath("task3_matrix.npy")
    if p3:
        m3 = np.load(p3, allow_pickle=True)
        uniq = unique_rows(m3)
        np.save(os.path.join(RESULT_DIR, "task3_unique_rows.npy"), uniq)
        print("Задача 3: выполнено. Уникальные строки сохранены в practice3_results/task3_unique_rows.npy")
    else:
        print("Задача 3: входной файл не найден. Пропуск.")

    # Задача 4
    p4 = inpath("task4_params.npz")
    if p4:
        params = np.load(p4, allow_pickle=True)
        M = int(params["M"])
        N = int(params["N"])
        seed = int(params["seed"]) if "seed" in params else None
        means, vars_ = random_normal_matrix_and_stats(M, N, seed=seed)
        np.savez(
            os.path.join(RESULT_DIR, "task4_random_normal_means_vars.npz"),
            means=means,
            variances=vars_,
        )
        print("Задача 4: выполнено. Средние и дисперсии сохранены в practice3_results/task4_random_normal_means_vars.npz")
    else:
        print("Задача 4: параметры не найдены. Пропуск.")

    # Задача 5
    p5 = inpath("task5_params.npz")
    if p5:
        params = np.load(p5, allow_pickle=True)
        M = int(params["M"])
        N = int(params["N"])
        a = params["a"].item() if hasattr(params["a"], "item") else params["a"]
        b = params["b"].item() if hasattr(params["b"], "item") else params["b"]
        cb = checkerboard_matrix(M, N, a, b)
        np.save(os.path.join(RESULT_DIR, "task5_checkerboard_matrix.npy"), cb)
        print("Задача 5: выполнено. Шахматная матрица сохранена в practice3_results/task5_checkerboard_matrix.npy")
    else:
        print("Задача 5: параметры не найдены. Пропуск.")

    # Задача 6
    p6 = inpath("task6_params.npz")
    if p6:
        params = np.load(p6, allow_pickle=True)
        h = int(params["height"])
        w = int(params["width"])
        r = int(params["radius"])
        color = np.array(params["color"]).astype(np.uint8)
        img = circle_image_tensor(h, w, r, color)
        np.save(os.path.join(RESULT_DIR, "task6_circle_image.npy"), img)
        plt.imsave(os.path.join(RESULT_DIR, "task6_circle_image.png"), img)
        print("Задача 6: выполнено. Изображение круга сохранено в practice3_results/task6_circle_image.npy и .png")
    else:
        print("Задача 6: параметры не найдены. Пропуск.")

    # Задача 7
    p7 = inpath("task7_tensor.npy")
    if p7:
        t7 = np.load(p7, allow_pickle=True)
        standardized = standardize_tensor(t7)
        np.save(os.path.join(RESULT_DIR, "task7_standardized_tensor.npy"), standardized)
        print("Задача 7: выполнено. Стандартизованный тензор сохранён в practice3_results/task7_standardized_tensor.npy")
    else:
        print("Задача 7: входной файл не найден. Пропуск.")

    # Задача 8
    p8m = inpath("task8_matrix.npy")
    p8meta = inpath("task8_meta.npz")
    if p8m and p8meta:
        mat8 = np.load(p8m, allow_pickle=True)
        meta = np.load(p8meta, allow_pickle=True)
        center_row = int(meta["center_row"])
        center_col = int(meta["center_col"])
        pr = int(meta["patch_rows"])
        pc = int(meta["patch_cols"])
        fill = meta["fill"].item() if "fill" in meta else 0
        patch = extract_centered_patch(mat8, center_row, center_col, pr, pc, fill=fill)
        np.save(os.path.join(RESULT_DIR, "task8_extracted_patch.npy"), patch)
        print("Задача 8: выполнено. Патч сохранён в practice3_results/task8_extracted_patch.npy")
    else:
        print("Задача 8: входные файлы не найдены. Пропуск.")

    # Задача 9
    p9 = inpath("task9_matrix.npy")
    if p9:
        mat9 = np.load(p9, allow_pickle=True)
        modes = row_modes(mat9)
        np.save(os.path.join(RESULT_DIR, "task9_row_modes.npy"), modes)
        print("Задача 9: выполнено. Моды по строкам сохранены в practice3_results/task9_row_modes.npy")
    else:
        print("Задача 9: входной файл не найден. Пропуск.")

    # Задача 10
    p10i = inpath("task10_image.npy")
    p10w = inpath("task10_weights.npy")
    if p10i and p10w:
        img10 = np.load(p10i, allow_pickle=True)
        w10 = np.load(p10w, allow_pickle=True)
        res10 = weighted_sum_channels(img10, w10)
        np.save(os.path.join(RESULT_DIR, "task10_weighted_sum_image.npy"), res10)
        print("Задача 10: выполнено. Взвешенная сумма каналов сохранена в practice3_results/task10_weighted_sum_image.npy")
    else:
        print("Задача 10: входные файлы не найдены. Пропуск.")


if __name__ == "__main__":
    main()
