#!/bin/sh

rm -rf cil
mkdir cil

cp -r twitter-datasets cil

cp config.py cil
cp model.py cil
cp glove.py cil
cp prepare_dataset.py cil
cp classifier.py cil

cp vocab.pkl cil
cp cooc.pkl cil

cp setup.sh cil

tar -czvf cil.tar.gz cil
rm -rf cil

scp cil.tar.gz fgraziano@login.leonhard.ethz.ch:~
rm -rf cil.tar.gz
