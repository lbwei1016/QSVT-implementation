import pstats

p = pstats.Stats('test.prof')
p.sort_stats('time').print_stats(30)
