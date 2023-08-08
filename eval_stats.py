import pstats

p = pstats.Stats('11q.prof')
p.sort_stats('time').print_stats(30)
