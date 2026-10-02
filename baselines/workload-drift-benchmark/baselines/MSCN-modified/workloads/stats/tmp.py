import json
fp = open('joins.csv')
lines = fp.readlines()
dct = {'pl':'postlinks',
       'p':'posts',
       'b':'badges',
       'c':'comments',
       'ph':'posthistory',
       'v':'votes',
       'u':'users',
       't':'tags'}
outdict = {'relationships':[]}
for i in range(1, len(lines)):
    line = lines[i].strip()
    line = line.lower()
    tmpl = []
    for j in line.split('='):
        t = j.split('.')[0]
        c = j.split('.')[1]
        t = dct[t]
        tmpl.append(t)
        tmpl.append(c)
    outdict['relationships'].append(tmpl)
jsonstring = json.dumps(outdict)
with open('tmpout.json','w') as fp:
    json.dump(outdict, fp)


