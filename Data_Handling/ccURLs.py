text_file = open("HTMLs.txt", "r")
full = text_file.read()


def find_all(a_str, sub):
    start = 0
    while True:
        start = a_str.find(sub, start)
        if start == -1: return
        yield start
        start += len(sub)


indices = list(find_all(full, 'href="/en/coasters/'))
endIndices = list(find_all(full, '">'))

HTMLs = list()

startingEnd = 0
for i in range(0, len(indices), 2):
    for j in range(startingEnd, len(endIndices)):
        if endIndices[startingEnd] < indices[i]:
            startingEnd += 1
        else:
            break
    HTMLs.append(full[indices[i]+6:endIndices[startingEnd]])

with open('cleaned.txt', 'w') as f:
    for item in HTMLs:
        f.write("%s\n" % item)