#PR 1 DAULETBEK ASKAR
import os
import numpy as np

"""
Генератор примерных входных данных для проверки задач.
Создаёт папку practice3_sample_inputs и набор .npy/.npz файлов.
Содержимое и форматы полностью совпадают с исходной версией (изменено только оформление).
"""

SAMPLE_DIR = "practice3_sample_inputs"
os.makedirs(SAMPLE_DIR, exist_ok=True)

# --- Задача 1: пример p=3, n=4 ---
p = 3
n = 4
a = np.array(
    [
        [[0, 1, 2, 3],
         [4, 5, 6, 7],
         [8, 9, 10, 11],
         [12, 13, 14, 15]],

        [[16, 17, 18, 19],
         [20, 21, 22, 23],
         [24, 25, 26, 27],
         [28, 29, 30, 31]],

        [[32, 33, 34, 35],
         [36, 37, 38, 39],
         [40, 41, 42, 43],
         [44, 45, 46, 47]],
    ],
    dtype=np.int64,
)

b = np.array(
    [
        [[0],
         [1],
         [2],
         [3]],

        [[4],
         [5],
         [6],
         [7]],

        [[8],
         [9],
         [10],
         [11]],
    ],
    dtype=np.int64,
)

np.save(os.path.join(SAMPLE_DIR, "task1_matrices.npy"), a)
np.save(os.path.join(SAMPLE_DIR, "task1_vectors.npy"), b)  # shape (p, n, 1)

# --- Задача 2: vector of integers ---
vec = np.array([0, 1, 2, 3, 5, 255, 1023], dtype=np.int64)
np.save(os.path.join(SAMPLE_DIR, "task2_vector.npy"), vec)

# --- Задача 3: matrix with duplicate rows ---
mat3 = np.array([[1, 2, 3],
                 [1, 2, 3],
                 [3, 4, 5],
                 [1, 2, 3],
                 [3, 4, 5]])
np.save(os.path.join(SAMPLE_DIR, "task3_matrix.npy"), mat3)

# --- Задача 4: params ---
M, N = 5, 20
np.savez(os.path.join(SAMPLE_DIR, "task4_params.npz"), M=M, N=N, seed=42)

# --- Задача 5: checkerboard params ---
np.savez(os.path.join(SAMPLE_DIR, "task5_params.npz"), M=6, N=7, a=10, b=-1)

# --- Задача 6: circle params ---
height, width, radius = 100, 150, 30
color = np.array([255, 0, 128], dtype=np.uint8)  # pinkish
np.savez(
    os.path.join(SAMPLE_DIR, "task6_params.npz"),
    height=height,
    width=width,
    radius=radius,
    color=color,
)

# --- Задача 7: tensor ---
tensor7 = np.random.RandomState(0).randn(4, 5, 3)
np.save(os.path.join(SAMPLE_DIR, "task7_tensor.npy"), tensor7)

# --- Задача 8: matrix + meta ---
mat8 = np.arange(1, 26).reshape(5, 5)
np.save(os.path.join(SAMPLE_DIR, "task8_matrix.npy"), mat8)
np.savez(
    os.path.join(SAMPLE_DIR, "task8_meta.npz"),
    center_row=0,
    center_col=0,
    patch_rows=3,
    patch_cols=3,
    fill=-999,
)

# --- Задача 9: matrix with ties ---
mat9 = np.array([
    [1, 2, 2, 3, 2],
    [5, 5, 6, 6, 5],
    [7, 8, 9, 9, 8],
])
np.save(os.path.join(SAMPLE_DIR, "task9_matrix.npy"), mat9)

# --- Задача 10 ---
h, w, c = 4, 5, 3
img10 = np.arange(h * w * c).reshape(h, w, c).astype(np.float64)
weights = np.array([0.1, 0.6, 0.3])
np.save(os.path.join(SAMPLE_DIR, "task10_image.npy"), img10)
np.save(os.path.join(SAMPLE_DIR, "task10_weights.npy"), weights)

print("Sample inputs created in ./practice3_sample_inputs")
