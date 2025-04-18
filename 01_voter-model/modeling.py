import numpy as np
import random
import time
import pandas as pd

print("Uruchamianie modeling.py...", flush=True)

def get_neighbors(i, j, L):
    """
    Zwraca listę sąsiadów (góra, dół, lewo, prawo) komórki (i, j)
    w siatce o rozmiarze L x L z warunkami periodycznymi.
    """
    return [((i - 1) % L, j),
            ((i + 1) % L, j),
            (i, (j - 1) % L),
            (i, (j + 1) % L)]

def voter_model(L=100, max_iter=100_000, verbose=True):
    print(f"Rozpoczynam symulację voter_model dla L={L}...", flush=True)
    network = np.random.randint(0, 2, (L, L))
    iterations = 0
    evolution = []
    start_time = time.time()
    
    while (0 < np.sum(network) < L * L) and (iterations < max_iter):
        iterations += 1
        if iterations % 1000 == 0:
            print(f"Iteracja {iterations}: suma opinii 'tak' = {np.sum(network)}", flush=True)
        for x in range(L):
            for y in range(L):
                neighbors = get_neighbors(x, y, L)
                nx, ny = random.choice(neighbors)
                network[x, y] = network[nx, ny]
        evolution.append(np.sum(network))
    
    elapsed_time = time.time() - start_time
    if verbose:
        if iterations == max_iter:
            print(f"Model dla L={L} osiągnął max_iter={max_iter}", flush=True)
        else:
            print(f"Model dla L={L} zbiega po {iterations} iteracjach", flush=True)
    return iterations, elapsed_time, evolution

def run_simulations_and_save():
    rows = []

    # Zadanie 1
    L = 100
    print("Zadanie 1: Symulacja dla L=100", flush=True)
    iters, t_elapsed, evo = voter_model(L=L, verbose=True)
    rows.append({
        "zad": 1,
        "L": L,
        "iter": iters,
        "time": t_elapsed,
        "evolution": ";".join(map(str, evo))
    })
    
    # Zadanie 2
    print("Zadanie 2: 10 symulacji dla L=100", flush=True)
    for sim in range(10):
        print(f"Symulacja {sim+1} dla L=100", flush=True)
        iters, t_elapsed, evo = voter_model(L=L, verbose=False)
        rows.append({
            "zad": 2,
            "L": L,
            "iter": iters,
            "time": t_elapsed,
            "evolution": ";".join(map(str, evo))
        })
    
    # Zadanie 3
    print("Zadanie 3: Symulacje dla L = 10, 50, 100, 200 (po 10 symulacji)", flush=True)
    for L_val in [10, 50, 100, 200]:
        print(f"Rozpoczynam symulacje dla L = {L_val}", flush=True)
        for sim in range(10):
            print(f"Symulacja {sim+1} dla L={L_val}", flush=True)
            iters, t_elapsed, evo = voter_model(L=L_val, verbose=False)
            rows.append({
                "zad": 3,
                "L": L_val,
                "iter": iters,
                "time": t_elapsed,
                "evolution": ";".join(map(str, evo))
            })
        print(f"10 x model dla L={L_val} obliczony", flush=True)
    
    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)
    print("Wyniki zapisane do pliku 'results.csv'", flush=True)

if __name__ == "__main__":
    run_simulations_and_save()
    print("Koniec symulacji.", flush=True)
