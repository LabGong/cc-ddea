import math
import os
from typing import Tuple
import time
import torch
import numpy as np
import benchmark
import sampling
import cc_ddea


gpu_device = torch.device('cuda:2')
funcs = [
    benchmark.Ackley,
    # benchmark.Ellipsoid,
    # benchmark.Griewank,
    # benchmark.Rastrigin,
    # benchmark.Rosenbrock,
    # benchmark.Qing
]
ds = [100] # , 200, 300, 500, 1000]
n = 1000
iter_max = 100
lr_individual = 0.1
max_run = 10
pop_size = 100

dirname = os.path.abspath(os.path.join(__file__, '..', 'time', 'cc-ddea'))

if not os.path.isdir(dirname):
    os.makedirs(dirname)

def main():
    for func in funcs:
        f = func()
        for d in ds:
            ys = []
            for i in range(max_run):
                x_train = sampling.lhs_np(n, d, *f.bound())
                y_train = f(x_train).reshape(-1, 1)

                lb, ub = f.bound()
                pop = sampling.lhs_np(pop_size, d, *f.bound())
                if ub - lb >= 1000:
                    iter_sub_max = 40
                else:
                    iter_sub_max = 10

                t1 = time.time()
                x = cc_ddea.run(
                    pop=pop,
                    iter_max=iter_max,
                    iter_sub_max=iter_sub_max,
                    samples=(x_train, y_train),
                    lower_bound=lb,
                    upper_bound=ub,
                    n_group_init=10,
                    group_update_gap=8,
                    gpu_device=gpu_device,
                    lr_individual=0.1,
                    surrogate_update_gap=int(8/2),
                    n_top_rate=0.1,
                    n_random_children_rate=0.1,
                )
                t2 = time.time()
                print(f'{f.name()}, d = {d}, run = {i}, time={t2 - t1}')

                y = f(x).numpy().reshape(-1)
                ys.append(y)
            ys = np.array(ys)
            print(f'{f.name().lower()}-{d}d.csv')
            print(os.path.join(dirname, f'{f.name().lower()}-{d}d.csv'))
            np.savetxt(os.path.join(dirname, f'{f.name().lower()}-{d}d.csv'), ys)

if __name__ == '__main__':
    main()