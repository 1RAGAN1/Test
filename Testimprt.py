# Testimprt
PI = 3.14
EULER = 2.718
GRAVITY = 9.8
SPEED_OF_LIGHT = 3e8
AVOGADRO = 6.022e23
BOLTZMANN = 1.38e-23
PLANCK = 6.626e-34

def test_add(x, y):
    return x + y

def test_subtract(x, y):
    return x - y

def test_multiply(x, y):
    return x * y

def test_divide(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero")
    return x / y

def test_power(x, y):
    return x ** y

def test_sqrt(x):
    if x < 0:
        raise ValueError("Cannot take square root of negative number")
    return x ** 0.5

def test_factorial(n):
    if n < 0:
        raise ValueError("Cannot take factorial of negative number")
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def test_fibonacci(n):
    if n < 0:
        raise ValueError("Cannot take fibonacci of negative number")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

def test_is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True