import os
import shutil

def group_columns(folder, csvs, queries_by_columns, columnums):
    columnum_str = ('').join(str(i) for i in columnums)
    csv_fp = open(folder + "/queries_{}_columns.csv".format(columnum_str), 'w')
    for i in columnums:
        for j in queries_by_columns[i]:
            csv_fp.write(csvs[j])

fp = open("job-light_equal_distribution_10000.csv")
csvs = fp.readlines()
queries_by_columns = [[] for i in range(20)]
folder = "varying_column_queries"
if os.path.exists(folder) is False:
    os.makedirs(folder)
for i, csv in enumerate(csvs):
    columns = csv.split('#')[2].split(',')
    columns_set = set()
    for j in range(0, len(columns), 3):
        cname = columns[j]
        if len(cname) > 0:
            columns_set.add(cname)
    columnum = len(columns_set)
    queries_by_columns[columnum].append(i)
for i in range(len(queries_by_columns)):
    group_columns(folder, csvs, queries_by_columns, [i])

group_columns(folder, csvs, queries_by_columns, [0, 1, 2, 3])
group_columns(folder, csvs, queries_by_columns, [4, 5, 6, 7])