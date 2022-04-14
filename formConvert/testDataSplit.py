import os
import argparse

parser = argparse.ArgumentParser(description="eg. python testDataSplit.py --source inputtest.txt --target resulttest.txt")
parser.add_argument("--source", type=str,  required=True,  help="source file eg. t.txt")
# parser.add_argument("--target",type=str,  required=True, help="target file eg. text.txt")
args = parser.parse_args()

# print(args.source)

target_name = args.source[:args.source.rfind('-') + 1] + 'result.txt'
with open(target_name, "w") as f:
    for line in open(args.source): 
        if line.find("####") != -1 :
            res = line.rsplit("/", 1)
            # print(res[-1], end="")
            f.write(res[-1])
print('save to ' + target_name)


# python testDataSplit.py --source inputtest.txt --target resulttest.txt