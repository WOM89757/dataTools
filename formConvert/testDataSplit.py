import os
import argparse

parser = argparse.ArgumentParser(description="eg. python testDataSplit.py --source inputtest.txt --target resulttest.txt")
parser.add_argument("--source", type=str,  required=True,  help="source file eg. t.txt")
parser.add_argument("--target",type=str,  required=True, help="target file eg. text.txt")
args = parser.parse_args()

# print(args.source)


with open(args.target,"w") as f:
    for line in open(args.source): 
        res = line.rsplit("/", 1)
        # print(res[-1])
        f.write(res[-1])


# with open("20220318.v1.txt","w") as f:
#     for line in open("20220318.txt"): 
#         res = line.rsplit("/", 1)
#         # print(res[-1])
#         f.write(res[-1])


# python testDataSplit.py --source inputtest.txt --target resulttest.txt