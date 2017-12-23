#!/bin/bash

root='/home/tako/datasets/oxford_pet'
sproot='/home/tako/1061/ce_deep_learn/fp/data/Animals_with_Attributes2/testimg'
tard='cat,dog'
trted='oxford_pet'


IFS=',' read -a tard_ <<< "$tard"

dir=./datasets/${trted}
mkdir $dir
ln -s ${root}/images/${tard_[0]} $dir/trainA
ln -s ${root}/images/${tard_[1]} $dir/trainB
ln -s ${root}/annotations/masks/${tard_[0]} $dir/trainmaskA
ln -s ${root}/annotations/masks/${tard_[1]} $dir/trainmaskB
