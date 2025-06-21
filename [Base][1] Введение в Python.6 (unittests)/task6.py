def check(x: str, file: str):
    dic = {}
    s = list(x.split())
    for i in s:
        if i.lower() in dic.keys():
            dic[i.lower()] += 1
        else:
            dic[i.lower()] = 1
    f = open(file, "w")
    for i in sorted(dic.keys()):
        print(i,dic[i], file = f)
    f.close()
