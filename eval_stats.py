import pstats

p = pstats.Stats('./profiles_tmp/9q-nAA-0.prof')
p.sort_stats('time').print_stats(30)
