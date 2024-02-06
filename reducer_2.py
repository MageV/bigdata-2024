import sys

for line in sys.stdin:
    key,value,med=line.strip().split()
    print(f'{key}\t {int(value)/int(med)}')