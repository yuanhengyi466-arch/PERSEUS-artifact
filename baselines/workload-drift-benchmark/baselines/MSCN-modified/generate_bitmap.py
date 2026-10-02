import psycopg2 as pg
import time

def table2sampleid(table_name):
    t2sid = {
        'title': 'tid',
        'cast_info': 'ciid',
        'movie_companies': 'mcid',
        'movie_info_idx': 'miiid',
        'movie_info': 'miid',
        'movie_keyword': 'mkid'
        }
    return t2sid[table_name]

def ali2table(ali):
    a2t = {
        't': 'title',
        'ci': 'cast_info',
        'mc': 'movie_companies',
        'mi_idx': 'movie_info_idx',
        'mi': 'movie_info',
        'mk': 'movie_keyword',
        'title': 'title',
        'cast_info': 'cast_info',
        'movie_companies': 'movie_companies',
        'movie_info_idx': 'movie_info_idx',
        'movie_info': 'movie_info',
        'movie_keyword': 'movie_keyword'
    }
    return a2t[ali]

sample_num = 1000
conn = pg.connect(dbname='imdb', user='peizhi', port=5434)
cur = conn.cursor()
csv_file = open('./data/job-light-ranges-id/train.csv', 'r')
#sql_file = open('job-light-ranges_mm_wisdm_deepdb.sql', 'r')
csv_lines = csv_file.readlines()
#sql_lines = sql_file.readlines()
all_tables = []
all_sqls = []
outfile = open('./data/job-light-ranges-id/train.bitmaps', 'wb')
start = time.time()
for i, line in enumerate(csv_lines):
    sql = 'SELECT '
    use_tables = []
    sqls = []
    tables = line.split('#')[0].split(',')
    joins = line.split('#')[1].split(',')
    predicates = line.split('#')[2].split(',')
    #print(i, len(predicates))
    for t in tables:
        #t_name = t.split(' ')[0]
        t_name = t
        idname = table2sampleid(t_name)
        sql = 'SELECT ' + t_name + '_sample.' + idname + ' FROM ' + t_name + '_sample WHERE '
        sqls.append(sql)
        use_tables.append(t_name)
    if len(predicates) >= 3:
        for j in range(0, len(predicates), 3):
            t_col = predicates[j]
            op = predicates[j + 1]
            val = predicates[j + 2]
            t = ali2table(t_col.split('.')[0]) 
            col = t_col.split('.')[1]
            t_index = use_tables.index(t)
            sql = sqls[t_index]
            sql = sql + t + '_sample.' + col + op + val + ' AND '
            sqls[t_index] = sql
    for j in range(len(sqls)):
        if sqls[j][-2] == 'E':
            sqls[j] = sqls[j][:-7]
        else:
            sqls[j] = sqls[j][:-5]
    all_tables.append(use_tables)
    all_sqls.append(sqls)
end = time.time()
print("bitmap time: ", end - start)
        
for i in range(len(all_sqls)):
    sqls = all_sqls[i]
    tables = all_tables[i]
    table_num = len(tables)
    table_num_bin = table_num.to_bytes(4, 'little')
    outfile.write(table_num_bin)
    for j, sql in enumerate(sqls):
         cur.execute(sql)
         result = cur.fetchall()
         result_flag = [False] * sample_num 
         for sid in result:
             result_flag[sid[0] - 1] = True
         for k in range(int(sample_num / 8)):
             write_num = 0
             for l in range(8):
                 index = k * 8 + l
                 write_num += (2**l * result_flag[index])
             write_num_bin = write_num.to_bytes(1, 'little')
             outfile.write(write_num_bin)
outfile.close()
                 
                 
             
