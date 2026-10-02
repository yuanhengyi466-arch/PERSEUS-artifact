from datetime import datetime
 

name = "varying_value_distribution_queries/beta_distribution_2range/value_distribution_3_2"
fpin = open("{}.sql".format(name))
csvfile = open("{}.csv".format(name))
fpout = open("{}_keeptime.csv".format(name), 'w')
lines = fpin.readlines()
csvlines = csvfile.readlines()
for i, line in enumerate(lines):
    line = line[:-1]
    csvline = csvlines[i]
    new_tables = []
    new_joins = []
    new_predicates = []
    table_dict = {}
    gt = csvline.split('#')[-1][:-1]
    sql = line.split('||')[0].strip('\n')
    sql = sql[:-1]
    sql = sql.split(" FROM ")[1]
    tables = sql.split(" WHERE ")[0]
    join_predicates = sql.split(" WHERE ")[1]
    tables = tables.split(',')
    for table in tables:
        fullname = table.split(' as ')[0]
        fullname = fullname.replace(" ", "")
        alias = table.split(' as ')[1]
        alias = alias.replace(" ", "")
        table_dict[alias] = fullname
        new_tables.append(fullname + " " + alias)
    jps = join_predicates.split(' AND ')
    for pred in jps:
        if len(pred.split('.')) == 3:
            join = pred.replace(' ', '')
            new_joins.append(join)
        else:
            if '<=' in pred:
                op = '<='
            elif '>=' in pred:
                op = '>='
            elif '>' in pred:
                op = '>'
            elif '<' in pred:
                op = '<'
            elif '=' in pred:
                op = '='
            col = pred.split(op)[0]
            col.replace(' ', '')
            val = pred.split(op)[1]
            if 'timestamp' in val:
                val = val.split('::')[0]
                val = val[1:-1]
                #date_val = val.split(' ')[0]
                #time_val = val.split(' ')[1]
                #year = int(date_val.split('-')[0])
                #month = int(date_val.split('-')[1])
                #day = int(date_val.split('-')[2])
                #hour = int(time_val.split(':')[0])
                #minute = int(time_val.split(':')[1])
                #second = int(time_val.split(':')[2])
                #dtime = datetime(year, month, day, hour, minute, second)
                #val = str(int(round(dtime.timestamp())))
            new_predicates.append(col + ',' + op + ',' + val)
    new_tables = ','.join(new_tables)
    new_joins = ','.join(new_joins)
    new_predicates = ','.join(new_predicates)
    new_sql = new_tables + '#' + new_joins + '#' + new_predicates + '#' + str(gt)
    fpout.write(new_sql + '\n')
