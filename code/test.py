a = [[1, 2, 3], [4, 5, 6]]
newList = []
finalList = []
print(len(a[0]))
for idx in range(0, len(a[0])):
    for p in range(0, len(a)):
        newList.append(a[p][idx])
    finalList.append(newList)
    newList = []
    
    
print(finalList)