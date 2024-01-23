import functools

def print_args(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'args={args}')
        print(f'kwargs={kwargs}')
        result = func(*args, **kwargs)
        print(f'result={result}')
        return result
    return wrapper

#def sum_list(func):
#    @functools.wraps(func)
#    def wrapper(*args, **kwargs):
#        infer=list(map(sum,zip(*args)))
#        result=func(*args,**kwargs)
#        return result
#    return wrapper

@print_args
#@sum_list
def alter_sum(a,b,**kwargs):
    if(type(a)==list and type(b)==list):
        return list(map(sum,zip(a,b)))
    return a+b

x = alter_sum(1,2, test=True)
x = alter_sum([1,2,3], [1,2,3], test=True)