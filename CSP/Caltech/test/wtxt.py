import os

filepath = "./anno_test_1xnew"
name = os.listdir(filepath)
name.sort()

with open("test.txt","w") as f:
    for i in name:
        i = i[0:-8] + '\n'
        f.write(i)
