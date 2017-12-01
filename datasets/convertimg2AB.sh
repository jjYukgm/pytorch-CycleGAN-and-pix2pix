#!/bin/bash
FILE=$1

root='/home/tako/1061/ce_deep_learn/fp/data/Animals_with_Attributes2/JPEGImages'
sproot='/home/tako/1061/ce_deep_learn/fp/data/Animals_with_Attributes2/testimg'
tard='buffalo,cow,rhinoceros'
spld='cowb,cowc,cowr'
trted='cowb2c,cowc2r,cowb2c2r'
ratio=0.1

# split folder
python ./datasets/img2trte.py --root $root \
    --spr $sproot \
    -t $tard \
    -s $spld \
    --ratio $ratio || { echo "python false" && exit; }

IFS=',' read -a tard_ <<< "$tard"
IFS=',' read -a spld_ <<< "$spld"
IFS=',' read -a trted_ <<< "$trted"

tLen=${#tard_[@]}
for (( i=0; i<$((tLen-1)); i++ ));
do
    dir=./datasets/${trted_[$i]}
    mkdir $dir 
    ln -s ${root}/${tard_[$i]} $dir/trainA
    ln -s ${root}/${tard_[$((i+1))]} $dir/trainB
    ln -s ${root}/${spld_[$i]} $dir/testA
    ln -s ${root}/${spld_[$((i+1))]} $dir/testB

done

if [ "$tLen" -eq 3 ]; then
    dir=./datasets/${trted_[2]}
    mkdir $dir
    ln -s ${root}/${tard_[0]} $dir/trainA
    ln -s ${root}/${tard_[1]} $dir/trainB
    ln -s ${root}/${tard_[2]} $dir/trainC
    ln -s ${root}/${spld_[0]} $dir/testA
    ln -s ${root}/${spld_[1]} $dir/testB
    ln -s ${root}/${spld_[2]} $dir/testC
fi
