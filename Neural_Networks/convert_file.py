import csv
from keras.utils import to_categorical
import pandas as pd
results = []
with open("joinedTestingData.csv") as csvfile:
    reader = csv.reader(csvfile) # change contents to floats
    for row in reader: # each row is a list
        results.append(row)

def get_col(arra, index):
    fin_array = []
    for row in arra:
        fin_array.append(row[index])
    return fin_array

def add_col(arra, fin_array):
    count = 0
    for row in arra:
        row.append(fin_array[count])
        count += 1
    return arra

def remove_col(arra, index):
    for row in arra:
        del row[index]

    return arra

print(results)
# results = remove_col(results, 0)
# results = remove_col(results, 0)
# results = remove_col(results, 1)
print(results[0])
print(results[1])


print("_"*50)
print("Col Check")
print("_"*50)
col = 0
colum = get_col(results, col)
print(colum[0])
print("*******")
results = remove_col(results, col)
rows = colum[1:]
cpime = 0
fonzas = []
fonzas.append(colum[0])
for row in rows:
    fonzas.append(cpime)
    cpime += 1

fin_array = []
print(fonzas)
results = remove_col(results, 0)
results = add_col(results, fonzas)
print(results)

col = 0
for cola in range(0, len(results[0])-1):
    print("*******")
    colum = get_col(results, col)
    print(colum[0])
    print("*******")
    results = remove_col(results, col)
    rows = colum[1:]
    count = 0
    row = rows[count]
    while row == 'NA':
        count += 1
        row = rows[count]
    flag = False
    try:
        float(row)
        flag = True
    except:
        flag = False

    if flag:
        print(rows)
        fin_rw = []
        for numb in rows:
            if (numb == "NA"):
                fin_rw.append(0)
            else:
                fin_rw.append(float(numb))
        rows = fin_rw
    else:
        print(rows)
        if "NA" in rows:
            print("HEREEEE")
            ns = rows.index("NA")
            print("NA: "+str(ns))
            first = pd.Categorical(rows)
            rows = first.codes.tolist()
            numbB = rows[ns]
            print(numbB)
            newR = []
            for val in rows:
                if val==numbB:
                    newR.append(0)
                else:
                    newR.append(val)
            print(rows)
            rows = newR
            print(newR)
        else:
            first = pd.Categorical(rows)
            rows = first.codes.tolist()
    rows.insert(0, colum[0])
    print(rows)
    results = add_col(results, rows)

for row in results:
    print(row)
    print(len(row))
import csv
with open("sec_real_final.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerows(results)