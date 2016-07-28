#!/usr/bin/env bash

noCore=3
path=/data/vcc/adnlog
./utils/count.py $path/tr.csv  > fc.trva.t10.txt
converters/parallelizer-b.py -s $noCore converters/pre-b.py $path/tr.csv run.py tr.ffm
converters/parallelizer-b.py -s $noCore converters/pre-b.py $path/te.csv run.py te.ffm
./ffm-train -k 4 -t 18 -s $noCore -p te.ffm tr.ffm model
./ffm-predict te.ffm model te.out