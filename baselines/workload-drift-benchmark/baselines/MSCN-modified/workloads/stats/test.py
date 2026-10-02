fp = open("target_prediction.csv")
lines = fp.readlines()
card = []
for line in lines:
    gt = int(line.split('#')[-1])
    card.append(gt)
print(min(card))
print(max(card))
