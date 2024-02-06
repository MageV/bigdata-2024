import collections
import functools
import operator
import sys
from collections import Counter

total = dict()
med=dict()

for line in sys.stdin:
    visit = line.split('\t')
    try:
        total[visit[0]] += int(visit[1])
        med[visit[0]]+=1
    except:
        total[visit[0]]=int(visit[1])
        med[visit[0]]=1

for item in total:
    print(f'{item} \t {total[item]/med[item]}')


