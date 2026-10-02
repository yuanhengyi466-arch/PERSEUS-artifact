import os
import shutil

def group_tables(folder, csvs, queries_by_tables, tablenums):
    tablenum_str = ('').join(str(i) for i in tablenums)
    csv_fp = open(folder + "/queries_{}_tables.csv".format(tablenum_str), 'w')
    for i in tablenums:
        for j in queries_by_tables[i]:
            csv_fp.write(csvs[j])

fp = open("job-light_equal_distribution_10000.csv")
csvs = fp.readlines()
queries_by_tables = [[] for i in range(7)]
folder = "varying_table_queries"
if os.path.exists(folder) is False:
    os.makedirs(folder)
for i, csv in enumerate(csvs):
    tables = csv.split("#")[0].split(",")
    tablenum = len(tables)
    queries_by_tables[tablenum].append(i)
for i in range(len(queries_by_tables)):
    group_tables(folder, csvs, queries_by_tables, [i])

group_tables(folder, csvs, queries_by_tables, [1, 2, 3])
group_tables(folder, csvs, queries_by_tables, [4, 5, 6])