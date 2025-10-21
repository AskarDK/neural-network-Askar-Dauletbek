#PR 1 DAULETBEK ASKAR
from typing import List, Tuple, Iterable, Any
from collections import Counter, defaultdict
import math

# 1. Кол-во гласных (a, e, i, o, u)
def count_vowels(s: str) -> int:
    vowels = set("aeiou")
    return sum(1 for ch in s.lower() if ch in vowels)

# 2. Все символы встречаются ровно 1 раз?
def all_unique_chars(s: str) -> bool:
    c = Counter(s)
    return all(v == 1 for v in c.values())

# 3. Кол-во единичных битов у числа
def count_bits_one(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    return n.bit_count()

# 4. Мультипликативная устойчивость (сколько раз перемножать цифры до одной)
def multiplicative_persistence(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")
    steps = 0
    while n >= 10:
        prod = 1
        for d in str(n):
            prod *= int(d)
        n = prod
        steps += 1
    return steps

# 5. Среднеквадратическое отклонение двух векторов (RMSE)
def rmse_vectors(a: Iterable[float], b: Iterable[float]) -> float:
    a = list(a); b = list(b)
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    if not a:
        raise ValueError("Vectors must be non-empty")
    mse = sum((x - y) ** 2 for x, y in zip(a, b)) / len(a)
    return math.sqrt(mse)

# 6. (мат. ожидание, СКО) без готовых функций (СКО — генеральное)
def mean_and_std(nums: Iterable[float]) -> Tuple[float, float]:
    nums = list(nums)
    if not nums:
        raise ValueError("List must be non-empty")
    n = len(nums)
    mean = sum(nums) / n
    var = sum((x - mean) ** 2 for x in nums) / n
    std = math.sqrt(var)
    return mean, std

# 7. Разложение на простые множители как "(p**k)(q)..."
def prime_factorization_string(n: int) -> str:
    if n <= 1:
        raise ValueError("n must be > 1")
    factors = []
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            n //= d
            count += 1
        if count:
            factors.append((d, count))
        d += 1 if d == 2 else 2
    if n > 1:
        factors.append((n, 1))
    parts = []
    for p, k in factors:
        parts.append(f"({p})" if k == 1 else f"({p}**{k})")
    return "".join(parts)

# 8. По IP и маске вернуть адрес сети и широковещательный
def ip_network_and_broadcast(ip: str, mask: str) -> Tuple[str, str]:
    def to_int(addr: str) -> int:
        parts = addr.split(".")
        if len(parts) != 4:
            raise ValueError("Invalid IPv4 format")
        vals = [int(p) for p in parts]
        if any(not (0 <= v <= 255) for v in vals):
            raise ValueError("IPv4 octets must be 0..255")
        out = 0
        for v in vals:
            out = (out << 8) | v
        return out
    def to_str(x: int) -> str:
        return ".".join(str((x >> (8*i)) & 0xFF) for i in reversed(range(4)))
    ip_i = to_int(ip)
    mask_i = to_int(mask)
    net = ip_i & mask_i
    broadcast = net | (~mask_i & 0xFFFFFFFF)
    return to_str(net), to_str(broadcast)

# 9. Пирамида из кубиков: n = 1^2 + 2^2 + ... + k^2 ?
def pyramid_layers_from_cubes(n: int):
    if n < 0:
        return "It is impossible"
    total = 0
    k = 0
    while total < n:
        k += 1
        total += k * k
    return k if total == n else "It is impossible"

# 10. «Сбалансированное» число
def is_balanced_number(n: int) -> bool:
    s = str(abs(n))
    L = len(s)
    if L <= 2:
        return True
    if L % 2 == 1:
        mid = L // 2
        left = s[:mid]
        right = s[mid+1:]
    else:
        mid1 = L // 2 - 1
        mid2 = L // 2
        left = s[:mid1]
        right = s[mid2+1:]
    sum_left = sum(int(ch) for ch in left) if left else 0
    sum_right = sum(int(ch) for ch in right) if right else 0
    return sum_left == sum_right

# 11. Стратифицированное разбиение по классу в 0-й колонке с долей r и (1-r)
def stratified_split(matrix: List[List[Any]], r: float) -> Tuple[List[List[Any]], List[List[Any]]]:
    if not (0 < r < 1):
        raise ValueError("r must satisfy 0 < r < 1")
    groups = defaultdict(list)
    for row in matrix:
        if not row:
            continue
        cls = row[0]
        groups[cls].append(row)
    first, second = [], []
    for cls, rows in groups.items():
        m = len(rows)
        k = int(round(m * r))
        k = max(0, min(m, k))
        first.extend(rows[:k])
        second.extend(rows[k:])
    return first, second

# ----------------- Демонстрация -----------------
if __name__ == "__main__":
    print("Задача 1:", count_vowels("Hello"))                      # -> 5
    print("Задача 2:", all_unique_chars("abcABC"), all_unique_chars("hello"))
    print("Задача 3:", count_bits_one(0), count_bits_one(7), count_bits_one(1023))
    print("Задача 4:", multiplicative_persistence(39), multiplicative_persistence(4), multiplicative_persistence(999))
    print("Задача 5:", f"{rmse_vectors([1,2,3],[1,2,4]):.6f}")
    print("Задача 6:", mean_and_std([1,2,3,4,5]))
    print("Задача 7:", prime_factorization_string(86240))
    print("Задача 8:", ip_network_and_broadcast("192.168.1.130","255.255.255.0"))
    print("Задача 9:", pyramid_layers_from_cubes(5), pyramid_layers_from_cubes(14), pyramid_layers_from_cubes(30))
    print("Задача 10:", is_balanced_number(23441), is_balanced_number(7), is_balanced_number(1231), is_balanced_number(123456))
    M = [
        ["a", 1, 2],
        ["a", 3, 4],
        ["b", 5, 6],
        ["b", 7, 8],
        ["c", 9, 10],
        ["c", 11, 12],
    ]
    part1, part2 = stratified_split(M, 0.5)
    print("Задача 11 (часть 1):", part1)
    print("Задача 11 (часть 2):", part2)
