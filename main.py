import time

from decorators import alter_sum

"""
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

"""
print (heavy_func(5,5))
print (heavy_func(6,5))
print (heavy_func(5,5))


x = alter_sum(1,2, test=True)
x = alter_sum([1,2,3], [1,2,3], test=True)
"""

pows=lambda a,count:[a**i for i in range(count)]

print(list(filter(lambda x:x%2==0,pows(2,10))))
