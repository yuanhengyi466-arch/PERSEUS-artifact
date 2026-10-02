def addjoin(lines, joins_list):
    for line in lines:
        joins = line.split('#')[1]
        for join in joins.split(','):
            #t1 = join.split('=')[0]
            #t2 = join.split('=')[1]
            #if t1 < t2:
            #    newjoin = t1 + '=' + t2
            #else:
            #    newjoin = t2 + '=' + t1
            if join not in joins_list:
                joins_list.append(join)


fpsource = open("varying_label_queries/large_label_queries.csv")
fpadaptation = open("varying_label_queries/small_label_queries.csv")
#fppredict = open("target_prediction.csv")
joins = []
lines = fpsource.readlines()
addjoin(lines, joins)
lines = fpadaptation.readlines()
addjoin(lines, joins)
#lines = fppredict.readlines()
#addjoin(lines, joins)
fpout = open("joins_exp_paper.csv", 'w')
fpout.write("name\n")
for join in joins:
    fpout.write(join + '\n')
