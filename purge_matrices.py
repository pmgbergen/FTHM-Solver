import os
from pathlib import Path
from plot_utils import load_data

known_data = set()

stats_dir = Path('./stats')
data_dir = Path('./matrices')

for stats_file in os.listdir(stats_dir):
    try:
        stats = load_data(stats_dir / stats_file)
    except TypeError:
        print('Skipping', stats_file)
    for ts in stats:
        for ls in ts.linear_solves:
            known_data.add(ls.matrix_id)
            known_data.add(ls.iterate_id)
            known_data.add(ls.state_id)
            known_data.add(ls.rhs_id)

found = 0
not_found = 0
for mat_file in os.listdir(data_dir):
    if mat_file in known_data:
        found += 1
        # print("Found:", mat_file)
    else:
        not_found += 1
        os.remove(data_dir / mat_file)
        print("Not found:", mat_file)


print(f'{found} / {found + not_found}')
