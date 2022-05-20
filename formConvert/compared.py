from itertools import count
import os
import argparse

parser = argparse.ArgumentParser(description="eg. python compared.py  --file1 20220326-result.txt  --file2 20220414-result.txt")
parser.add_argument("--file1", type=str,  required=True, default=1, help="source file eg. t.txt")
parser.add_argument("--file2",type=str,  required=True, default=2, help="target file eg. text.txt")
args = parser.parse_args()

print(args.file1)
print(args.file2)

count = 0
with open("diff_{}_{}.txt".format(args.file1[:args.file1.find('.txt')],args.file2[:args.file2.find('.txt')]), "w") as f:
    print('result save to {}'.format(f.name))
    for line1 in open(args.file1): 
        v_name = line1[:line1.find('.mp4')]
        for line2 in open(args.file2): 
            v2_name = line2[:line2.find('.mp4')]
            if v_name != v2_name:
                continue;
            if len(line1[line1.find('.mp4') + 5:len(line1) - 1]) == 2 and len(line2[line2.find('.mp4') + 5:len(line2) - 1]) == 2:
                    
                val = int(line1[line1.find('.mp4') + 5:len(line1) - 1])
                val2 = int(line2[line2.find('.mp4') + 5:len(line2) - 1])
                a1 = val % 10
                b1= int(val / 10)
                c1 = a1 + b1

                a2 = val2 % 10
                b2= int(val2 / 10)
                c2 = a2 + b2
                # if v_name == '86885095231028800018':
                #     print(c2 )
                #     print(c1 )
                #     print(c1!=c2 )
                #     print(line1!=line2 )
                #     print(line1)
                #     print(line2)
                if c1 != c2:
                    count += 1
                    f.write('{}\t{}\t{}\n'.format(v_name, val, val2))

                    break
                # print(int(line1[line1.find('.mp4') + 5:len(line1) - 1]) % 10)
                # print(len(line1[line1.find('.mp4') + 5:len(line1) - 1]))
            elif line1 != line2:
                count += 1
                f.write('{}\t{}\t{}\n'.format(v_name, int(line1[line1.find('.mp4') + 5:len(line1) - 1]), int(line2[line2.find('.mp4') + 5:len(line2) - 1])))

                # f.write('1-' + line1)
                # f.write('2-' + line2)
                break
            # elif line1 == line2:
            #     val = int(line1[line1.find('.mp4') + 5:len(line1) - 1])
            #     val2 = int(line2[line2.find('.mp4') + 5:len(line2) - 1])
            #     if val == 8 and val == val2:
            #         f.write('{}\t{}\t{}\n'.format(v_name, int(line1[line1.find('.mp4') + 5:len(line1) - 1]), int(line2[line2.find('.mp4') + 5:len(line2) - 1])))

print('find {} line is not equal '.format(count))
    #     if line1.find("####") != -1 :
    #         res = line.rsplit("/", 1)