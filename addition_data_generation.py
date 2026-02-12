# Generate a dataset of addition problems
import random
import os
from tqdm import tqdm 

def generate_addition_data(n):
    """
    Generate a dataset of n unique 4-digit addition problems.

    The function should:
    1) Randomly sample two 4-digit integers a and b.
        Recall: 4-digit integers are integenrs in the range of [1000, 9999]
    2) Compute their sum c = a + b
    3) Ensure uniqueness of addition pairs, treating (a, b) and (b, a) as identical
    4) Repeat until n unique examples are collected
    5) Return the data formatted as strings of the form "a+b=c"
    """

    # data = []
    # pbar = tqdm(total=n, desc="Generating addition data", unit="examples")
    # try:
    #     while len(data) < n:
    #         # Randomly sample two 4-digit integers a and b
    #         a = random.randint(1000, 9999)
    #         b = random.randint(1000, 9999)
    #         c = a + b
    #         # Ensure uniqueness of addition pairs
    #         if (a, b, c) not in data and (b, a, c) not in data:
    #             data.append((a, b, c))
    #             pbar.update(1)
    # finally:
    #     pbar.close()
    # return [f"{a}+{b}={c}" for a, b, c in data]

    seen = set()          
    out = []
    pbar = tqdm(total=n, desc="Generating addition data", unit="examples")
    try:
        while len(out) < n:
            a = random.randint(1000, 9999)
            b = random.randint(1000, 9999)
            x, y = (a, b) if a <= b else (b, a)
            if (x, y) in seen:
                continue
            seen.add((x, y))
            c = a + b
            out.append(f"{a}+{b}={c}")
            pbar.update(1)
    finally:
        pbar.close()
    return out

def generate_dataset(n, filename, save_dir="data"):
    data = generate_addition_data(n)
    os.makedirs('data', exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, 'w') as f:
        f.write('\n'.join(data))
    print(f"{n} data points saved to {filepath}")
