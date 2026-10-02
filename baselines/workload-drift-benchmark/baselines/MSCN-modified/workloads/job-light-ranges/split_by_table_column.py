import os
import shutil

def group_table_columns(folder, csvs, queries_by_table_column, tablenum, columnum):
    tablenum_str = ("").join(str(i) for i in tablenum)
    columnum_str = ("").join(str(i) for i in columnum)
    table_folder = folder + "/queries_{}_tables".format(tablenum_str)
    if os.path.exists(table_folder) is False:
        os.makedirs(table_folder)
    csv_fp = open(table_folder + "/queries_{}_columns.csv".format(columnum_str), 'w')
    for i in tablenum:
        for j in columnum:
            for k in queries_by_table_column[i][j]:
                csv_fp.write(csvs[k])

fp = open("job-light_equal_distribution_10000.csv")
csvs = fp.readlines()
queries_by_table_column = [[[] for j in range(100)] for i in range(7)]
folder = "varying_table_column_queries"
if os.path.exists(folder) is False:
        os.makedirs(folder)
for i, csv in enumerate(csvs):
    tables = csv.split("#")[0].split(',')
    tablenum = len(tables)
    columns = csv.split('#')[2].split(',')
    columns_set = set()
    for j in range(0, len(columns), 3):
        cname = columns[j]
        if len(cname) > 0:
            columns_set.add(cname)
    columnum = len(columns_set)
    queries_by_table_column[tablenum][columnum].append(i)
for i in range(7):
    for j in range(100):
        if len(queries_by_table_column[i][j]) > 0:
            group_table_columns(folder, csvs, queries_by_table_column, [i], [j])

small_table_ids = [1,2,3]
large_table_ids = [4,5]
small_column_ids = [i for i in range(4)]
large_column_ids = [i for i in range(4, 8)]
group_table_columns(folder, csvs, queries_by_table_column, small_table_ids, small_column_ids)
group_table_columns(folder, csvs, queries_by_table_column, small_table_ids, large_column_ids)
group_table_columns(folder, csvs, queries_by_table_column, large_table_ids, small_column_ids)
group_table_columns(folder, csvs, queries_by_table_column, large_table_ids, large_column_ids)
