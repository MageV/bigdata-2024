import time


def cache(func):
    cache = {}

    def inner(*args):
        if (args in cache):
            return cache[args]
        else:
            result = func(*args)
            cache[args] = result
            return result

    return inner


@cache
def heavy_func(a, b):
    time.sleep(5)
    return a + b
"""
Первый и второй вызов с задержкой 5 сек. 3 быстрый так как результат 5,5 закэширован
"""
print (heavy_func(5,5))
print (heavy_func(6,5))
print (heavy_func(5,5))
