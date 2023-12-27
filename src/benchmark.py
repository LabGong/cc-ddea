# -*- coding: utf-8 -*-
from typing import Any, Tuple, Union
import math
import numpy as np
try:
    import torch
except ModuleNotFoundError as e:
    pass

def isTorchTensor(x: Any) -> bool:
    return hasattr(x, 'numpy')

def to2d(x: Any) -> Union[np.ndarray, torch.Tensor]:
    '''convert to 2d numpy.array or 2d torch.Tensor
    '''
    # take type conversion if necessary
    if isinstance(x, np.ndarray) == False and isTorchTensor(x) == False:
        x = np.array(x)
    # convert to matrix
    if x.ndim == 1 or x.ndim == 0:
        x = x.reshape([1, -1])
    return x

class Sphere:
    '''Sphere function
    Equation:
        $$
        f(x) = \sum_{i=1}{d}x_i^2 \\
        x_i \in [-5.12, 5.12] \\
        x^* = 0 \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    Symbol:
        d: dimension
    '''

    def name(self) -> str:
        return 'Sphere'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        if isTorchTensor(x):
            return torch.sum(x ** 2, dim=1)
        return np.sum(x ** 2, axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -5.12, 5.12


class Ackley:
    '''Ackley function
    Equation:
        $$
        f(x) = -a \exp (-b \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2}) - \exp(\frac{1}{d} \sum_{i=1}^{d} \cos(c x_i)) + a + exp(1) \\
        x_i \in [-32.768, 32.768] \\
        x^* = 0 \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    Symbol:
        d: dimension
    '''

    def __init__(self, a=20, b=0.2, c=2*math.pi):
        self.a = a
        self.b = b
        self.c = c

    def name(self) -> str:
        return 'Ackley'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        d = x.shape[1]
        if isTorchTensor(x):
            part1 = -self.b * ((torch.sum(x ** 2, dim=1) / d) ** 0.5)
            part2 = torch.sum(torch.cos(self.c * x), dim=1) / d
        else:
            part1 = -self.b * ((np.sum(x ** 2, axis=1) / d) ** 0.5)
            part2 = np.sum(np.cos(self.c * x), axis=1) / d
        # attension: there is small difference between math.e ** x and np.exp(x)
        return -self.a * (math.e ** part1) - (math.e ** part2) + self.a + math.e

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -32.768, 32.768


class Rosenbrock:
    '''Rosenbrock Function
    Equation:
        $$
        f(x) = \sum_{i=1}^{d-1}[100(x_{i+1} - x_i^2)^2 + (x_i - 1)^2] \\
        x_i \in [-2.048, +2.048] \\
        x^* = [1,1,...,1] \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate
    '''

    def name(self) -> str:
        return 'Rosenbrock'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        x1 = x[:, :-1]
        x2 = x[:, 1:]
        y = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
        if isTorchTensor(x):
            return torch.sum(y, dim=1)
        return np.sum(y, axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -2.048, 2.048


class Ellipsoid:
    '''Ellipsoid function
    Equation:
        $$
        f(x) = \sum_{i=1}^{d} i * x_i^2 \\
        x_i \in [-5.12,5.12] \\
        x^* = 0 \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate
    '''

    def name(self) -> str:
        return 'Ellipsoid'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        if isTorchTensor(x):
            return torch.sum((x ** 2) * torch.arange(1, x.shape[1] + 1, device=x.device), dim=1)
        return np.sum((x ** 2) * np.arange(1, x.shape[1] + 1), axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -5.12, 5.12


class Griewank:
    '''Griewank function
    Equation:
        $$
        f(x) = \sum_{i=1}^{d} \frac{x_i^2}{4000} - \prod_{i=1}^{d}\cos\frac{x_i}{\sqrt{i}} + 1 \\
        x_i \in [-600, +600] \\
        x^* = 0 \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate
    '''

    def name(self) -> str:
        return 'Griewank'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        if isTorchTensor(x):
            i = torch.arange(1, x.shape[1] + 1, device=x.device) ** 0.5
            return torch.sum(x ** 2 / 4000, dim=1) - torch.prod(torch.cos(x / i), dim=1) + 1
        i = np.arange(1, x.shape[1] + 1) ** 0.5
        return np.sum(x ** 2 / 4000, axis=1) - np.prod(np.cos(x / i), axis=1) + 1

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -600, 600


class Rastrigin:
    '''Rastrigin function
    Equation:
        $$
        f(x) = 10d + \sum_{i=1}^d [x_i^2 + 10cos(2 \pi x_i)] \\
        x_i \in [-5.12, +5.12] \\
        x^* = 0 \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Rastrigin'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        if isTorchTensor(x):
            return 10 * x.shape[1] + torch.sum(x ** 2 - 10 * torch.cos(2 * math.pi * x), dim=1)
        return 10 * x.shape[1] + np.sum(x ** 2 - 10 * np.cos(2 * math.pi * x), axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -5.12, 5.12

class XinSheYangN4:
    '''Xin-She Yang N. 4 function
    Equation:
        $$
        f(x) = \left(\sum_{i=1}^n sin^2(x_i)-e^{-\sum_{i=1}^n x_i^2}\right) e^{-\sum_{i=1}^n{sin^2\sqrt{|x_i|}}} \\
        x_i \in [-10, +10] \\
        x^* = 0 \\
        f(x^*) = -1
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Xin-She Yang N. 4'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        if isTorchTensor(x):
            part1 = torch.sum(torch.sin(x) ** 2, dim=1) - math.e ** (-torch.sum(x ** 2, dim=1))
            part2 = math.e ** -torch.sum(torch.sin(torch.sqrt(torch.abs(x))) ** 2, dim=1)
            return part1 * part2

        part1 = np.sum(np.sin(x) ** 2, axis=1) - np.e ** (-np.sum(x ** 2, axis=1))
        part2 = np.e ** -np.sum(np.sin(np.sqrt(np.abs(x))) ** 2, axis=1)
        return part1 * part2

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -10, 10

class Shubert3:
    '''Shubert 3 Function
    Equation:
        $$
        f(x) = \sum_{i=1}^n{\sum_{j=1}^5{j sin((j+1)x_i+j)}} \\
        x_i \in [-10, +10] \\
        f(x^*) \approx -29.6733337
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Shubert 3'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        n, d = x.shape

        if isTorchTensor(x):
            ret = torch.zeros(n, dtype=x.dtype, device=x.device)
            for i in range(d):
                for j in range(1, 6):
                    ret += j * np.sin( (j + 1) * x[:, i] + j)
            return ret

        ret = np.zeros(n, dtype=x.dtype)
        for i in range(d):
            for j in range(1, 6):
                ret += j * np.sin( (j + 1) * x[:, i] + j)
        return ret

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -10, 10

class AlpineN1:
    '''Alpine N. 1 Function
    Equation:
        $$
        f(x) = \sum{i=1}^{n}|x_i sin(x_i)+0.1x_i| \\
        x_i \in [0, 10] \\
        x^* = [0, \dots, 0] \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Alpine N. 1'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)

        if isTorchTensor(x):
            return torch.sum(torch.abs(x * torch.sin(x) + 0.1 * x), dim=1)

        return np.sum(np.abs(x * np.sin(x) + 0.1 * x), axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return 0, 10

class AlpineN2:
    '''Alpine N. 2 Function
    Equation:
        $$
        f(x) = \prod_{i=1}^{d}\sqrt{x_i}\sin(x_i) \\
        x_i \in [0, 10] \\
        x^* = [7.917, \dots, 7.917] \\
        f(x^*) \approx 2.808^d
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Alpine N. 2'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)

        if isTorchTensor(x):
            return torch.prod(torch.sqrt(x) * torch.sin(x), dim=1)

        return np.prod(np.sqrt(x) * np.sin(x), axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return 0, 10

class Brown:
    '''Brown Function
    Equation:
        $$
        f(x) = \sum_{i=1}^{d-1}(x_i^2)^{(x_{i+1}^{2}+1)}+(x_{i+1}^2)^{(x_{i}^{2}+1)} \\
        x_i \in [-1, 4] \\
        x^* = [0, \dots, 0] \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Brown'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)

        x1 = x[:, :-1]
        x2 = x[:, 1:]
        ret = (x1 ** 2) **(x2 ** 2 + 1) + (x2 ** 2) ** (x1 ** 2 + 1)

        if isTorchTensor(x):
            return torch.sum(ret, dim=1)

        return np.sum(ret, axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -1, 4

class Exponential:
    '''Exponential Function
    Equation:
        $$
        f(x) = -e^{-0.5 \sum_{i=1}^n x_i^2} \\
        x_i \in [-1, 1] \\
        x^* = [0, \dots, 0] \\
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''
    def name(self) -> str:
        return 'Exponential'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)

        if isTorchTensor(x):
            return -(math.e ** (-0.5 * torch.sum(x ** 2, axis=1)))

        return -(math.e ** (-0.5 * np.sum(x ** 2, axis=1)))

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -1, 1


class Periodic:
    '''Periodic Function
    # 网站的公式给错，但是代码是对的
    Equation:
        $$
        f(x) = 1 + \sum_{i=1}^{n}{sin^2(x_i)}-0.1e^{(- \sum_{i=1}^{n}x_i^2)} \\
        x_i \in [-10, 10] \\
        x^* = [0, \dots, 0] \\
        f(x^*) = 0.9
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Periodic'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)

        if isTorchTensor(x):
            return 1 + torch.sum(torch.sin(x) ** 2, dim=1) - 0.1 * math.e ** (-torch.sum(x ** 2, dim=1))

        return 1 + np.sum( np.sin(x) ** 2, axis=1) - 0.1 * math.e ** (-np.sum(x ** 2, axis=1))

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -10, 10

class PowellSum:
    '''Powell Sum Function
    Equation:
        $$
        f(x) = \sum_{i=1}^{n}|x_i|^{i+1} \\
        x_i \in [-1, 1] \\
        x^* = [0, \dots, 0] \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Powell Sum'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        d = x.shape[-1]
        if isTorchTensor(x):
            a = torch.arange(2, d + 2, dtype=x.dtype, device=x.device)
            return torch.sum(torch.abs(x) ** a, axis=1)

        a = np.arange(2, d + 2)
        return np.sum(np.abs(x) ** a, axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -1, 1

class Qing:
    '''Qing Function
    Equation:
        $$
        f(x) = \sum_{i=1}^{n}(x^2-i)^2 \\
        x_i \in [-500, 500] \\
        x^* = [\sqrt{i}, \dots, \sqrt{i}] \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Qing'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        d = x.shape[-1]
        if isTorchTensor(x):
            i = torch.arange(1, d + 1, dtype=x.dtype, device=x.device)
            return torch.sum((x ** 2 - i) ** 2, axis=1)

        i = np.arange(1, d + 1)
        return np.sum((x ** 2 - i) ** 2, axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -500, 500

class Ridge:
    '''Ridge Function
    Equation:
        $$
        f(x) = x_1 + d\left(\sum_{i=2}^{n}x_i^2\right)^\alpha \\
        x_i \in [-5, 5] \\
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Ridge'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any], d=1, alpha=0.5) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        if isTorchTensor(x):
            return x[:, 0] + d * torch.sum(x[:, 1:] ** 2, dim=1) ** alpha

        return x[:, 0] + d * np.sum(x[:, 1:] ** 2, axis=1) ** alpha

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -5, 5

# todo: test
class Elliptic:
    '''Elliptic Function
    Equation:
        $$
        f(x) = \sum_{i=1}^d 10^{6 \frac{i - 1}{d - 1} x_i^2} \\
        x_i \in [-100, +100] \\
        x^* = 0 \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Elliptic'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        d = x.shape[-1]
        if isTorchTensor(x):
            part1 = (torch.arange(1, d + 1) - 1)
            part1 = 10 ** (6 / (d - 1) * part1)
            return torch.sum(part1 * (x ** 2), axis=1)
        part1 = (np.arange(1, d + 1) - 1) / (d - 1)
        part1 = 10 ** (6 * part1)
        return np.sum(part1 * (x ** 2), axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -100, 100

# todo: test
class SchwefelProblem12:
    '''Schwefel's Problem 1.2
    Equation:
        $$
        f(x) = \sum_{i=1}^d \left( \sum_{j=1}^i x_i \right)^2 \\
        x_i \in [-100, +100] \\
        x^* = 0 \\
        f(x^*) = 0
        $$

    Args:
        x: a row vector(1-d or 2-d) or a matrix. If `x` is a matrix, every row of it can be considered as a row vector to calculate

    '''

    def name(self) -> str:
        return 'Schwefel\'s Problem 1.2'

    def __call__(self, x: Union[np.ndarray, torch.Tensor, Any]) -> Union[np.ndarray, torch.Tensor]:
        x = to2d(x)
        d = x.shape[-1]
        if isTorchTensor(x):
            return torch.sum(torch.cumsum(x,dim=1) ** 2, dim=1)
        return np.sum(np.cumsum(x,axis=1) ** 2, axis=1)

    def bound(self) -> Union[Tuple[float, float], Tuple[np.ndarray, np.ndarray], Tuple[torch.Tensor, torch.Tensor]]:
        return -100, 100