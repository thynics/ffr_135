import numpy as np
from typing import List, Dict, Tuple
from sklearn.svm import LinearSVC
from sklearn.exceptions import ConvergenceWarning
import warnings


def bool_inputs_pm1(n: int) -> np.ndarray:
    L = 1 << n
    X01 = ((np.arange(L, dtype=np.uint32)[:, None] >> np.arange(n, dtype=np.uint32)[::-1]) & 1).astype(np.float32)
    return 2.0 * X01 - 1.0

def sgn_pm1(b: np.ndarray) -> np.ndarray:
    return np.where(b >= 0.0, 1.0, -1.0).astype(np.float32)

# build all true-value table of n-d vector
def build_bool_function_tables(n: int) -> List[np.ndarray]:
    L = 1 << n
    out: List[np.ndarray] = []
    for u in range(1 << L):
        bits = ((u >> np.arange(L, dtype=np.uint32)) & 1).astype(np.int8)
        T = (2 * bits - 1).astype(np.int8)
        out.append(T)
    return out

# sample instead of build all
def sample_function_tables(n: int, count: int, seed: int = 0) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    L = 1 << n
    total_funcs = 1 << L
    out: List[np.ndarray] = []

    if n == 4 and count <= total_funcs:
        choice = rng.choice(total_funcs, size=count, replace=False)
        for u in choice:
            bits = ((u >> np.arange(L, dtype=np.uint32)) & 1).astype(np.int8)
            T = (2 * bits - 1).astype(np.int8)
            out.append(T)
    else:
        for _ in range(count):
            bits = rng.integers(low=0, high=2, size=L, dtype=np.int8)
            T = (2 * bits - 1).astype(np.int8)
            out.append(T)
    return out

def train_and_test(T: np.ndarray, epochs: int = 20, lr: float = 0.05, seed: int = 0) -> bool:
    L = T.shape[0]
    n = int(np.log2(L) + 1e-9)
    if (1 << n) != L:
        raise ValueError("lenght of T must be 2^n")
    X = bool_inputs_pm1(n)
    t = T.astype(np.float32)
    rng = np.random.default_rng(seed)
    w = rng.normal(loc=0.0, scale=np.sqrt(1.0 / n), size=n).astype(np.float32)
    theta = 0.0

    idx = np.arange(L, dtype=np.int32)
    for _ in range(epochs):
        rng.shuffle(idx)
        for i in idx:
            b = float(X[i] @ w - theta)
            O = 1.0 if b >= 0.0 else -1.0
            delta = t[i] - O
            if delta != 0.0:
                w += lr * delta * X[i]
                theta += -lr * delta

    O_all = sgn_pm1(X @ w - theta)
    return bool(np.all(O_all == t))



def svm_is_separable(T: np.ndarray) -> bool:
    L = T.shape[0]
    n = int(np.log2(L) + 1e-9)
    if (1 << n) != L:
        raise ValueError("size of T must be 2^n")
    X = bool_inputs_pm1(n).astype(np.float64)
    y = T.astype(np.int8)

    if np.unique(y).size == 1:
        return True

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        clf = LinearSVC(C=1.0, tol=1e-4, dual=False, max_iter=2000)
        clf.fit(X, y)

    pred = clf.predict(X).astype(np.int8)
    return bool(np.array_equal(pred, y))

def train_and_validate(n, tables, epochs=20, lr=0.05, seed=42):
    total = len(tables)
    cnt_20 = 0
    cnt_oracle = 0
    agree = 0
    for k, T in enumerate(tables):
        ok_20 = train_and_test(T, epochs=epochs, lr=lr, seed=seed + k)
        ok_orc = svm_is_separable(T)
        cnt_20 += int(ok_20)
        cnt_oracle += int(ok_orc)
        agree += int(ok_20 == ok_orc)
    frac_20 = cnt_20 / total if total else 0.0
    frac_orc = cnt_oracle / total if total else 0.0
    acc = agree / total if total else 0.0
    print(f"{n:>2} | {total:>8} | {cnt_20:>9} | {frac_20:>9.5f} | {cnt_oracle:>10} | {frac_orc:>11.5f} | {acc:>9.5f}")

def main():
    seed = 42
    epochs = 20
    lr = 0.05
    sample_n4 = 10_000
    sample_n5 = 10_000

    rng = np.random.default_rng(seed)

    tables_by_n: Dict[int, List[np.ndarray]] = {
        2: build_bool_function_tables(2),
        3: build_bool_function_tables(3),
        4: sample_function_tables(4, sample_n4, seed=seed),
        5: sample_function_tables(5, sample_n5, seed=seed),
    }

    print(f"{'n':>2} | {'total':>8} | {'20ep_sep':>9} | {'20ep_frac':>9} | {'oracle_sep':>10} | {'oracle_frac':>11} | {'accuracy':>9}")
    print("-"*80)

    for n, tables in tables_by_n.items():
        train_and_validate(n, tables)

    # try to validate n = 5 with more sample number
    for i in range(4, 10):
        sample_tb = sample_function_tables(5, 10**i)
        train_and_validate(5, sample_tb)
        
    
if __name__ == "__main__":
    main()