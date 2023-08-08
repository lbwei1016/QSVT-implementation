import pstats

p = pstats.Stats('qsvt.prof')
p.sort_stats('time').print_stats(30)