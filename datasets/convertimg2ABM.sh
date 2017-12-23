#!/bin/bash

root='/Disk2/cedl/oxford_pet'
tard='cat,dog'
trted='oxford_pet'


IFS=',' read -a tard_ <<< "$tard"

dir=./datasets/${trted}
mkdir $dir
ln -s ${root}/images/${tard_[0]} $dir/trainA
ln -s ${root}/images/${tard_[1]} $dir/trainB
ln -s ${root}/annotations/masks/${tard_[0]} $dir/trainmaskA
ln -s ${root}/annotations/masks/${tard_[1]} $dir/trainmaskB
