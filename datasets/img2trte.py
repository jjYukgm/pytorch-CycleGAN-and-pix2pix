import os
import sys
import argparse

parser = argparse.ArgumentParser(description='SplitImageFolder')
# parser.add_argument('-n', '--net', type=str, help='net name', choices = ckpt_nets())
parser.add_argument('-r', '--root', type=str
    , default='/home/tako/1061/ce_deep_learn/fp/data/Animals_with_Attributes2/JPEGImages', help='root folder')
parser.add_argument('--spr', type=str
    , default='/home/tako/1061/ce_deep_learn/fp/data/Animals_with_Attributes2/testimg', help='split root folder')
parser.add_argument('-t', '--target_d', type=str
    , default='buffalo,cow,rhinoceros', help='target folder name')
parser.add_argument('-s', '--split_d', type=str
    , default='cowb,cowc,cowr', help='split folder name')
parser.add_argument('--ratio', type=float
    , default=0.1, help='split ratio')
args = parser.parse_args()


rootdir = args.root
tard = args.target_d
sprd = args.spr
splitd = args.split_d
ratio = args.ratio

barLength = 10
splitd = splitd.split(',')
for i, td in enumerate(tard.split(',')):
    fileList = []
    rd = os.path.join(rootdir,td)
    for root, subFolders, files in os.walk(rd):
        for file in files:
            f = os.path.join(root,file)
            #print(f)
            fileList.append(f)
    split_num = int(round(ratio* len(fileList)))
    print("root: {0}\nTarget: {1}\n#Img: {2}\nSplit to {3}: {4}".format(
        rootdir, td, len(fileList), splitd[i], split_num))
        
    # move file
    sd = os.path.join(sprd,splitd[i])
    try:
        os.stat(sd)
    except:
        os.mkdir(sd) 
    status=""
    for j in range(split_num):
        spFileName = os.path.basename(fileList[j])
        spFileName = os.path.join(sd, spFileName)
        os.rename(fileList[j], spFileName)
        # bar
        if j == split_num -1:
            status = "\r\n"
        progress = (j+1.) / (split_num +0.)
        block = int(round(progress*barLength))
        text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), progress*100, status)
        sys.stdout.write(text)
        sys.stdout.flush()
        