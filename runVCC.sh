#!/usr/bin/env bash

nr_thread=$1
path=$2
./utils/count.py $path/tr.csv  > fc.trva.t10.txt
converters/parallelizer-a.py -s $nr_thread converters/pre-a.py $path/tr.csv tr.gbdt.dense tr.gbdt.sparse
converters/parallelizer-a.py -s $nr_thread converters/pre-a.py $path/te.csv te.gbdt.dense te.gbdt.sparse
./gbdt -t 30 -s $nr_thread te.gbdt.dense te.gbdt.sparse tr.gbdt.dense tr.gbdt.sparse te.gbdt.out tr.gbdt.out
converters/parallelizer-b.py -s $nr_thread converters/pre-b.py $path/tr.csv tr.gbdt.out tr.ffm
converters/parallelizer-b.py -s $nr_thread converters/pre-b.py $path/te.csv te.gbdt.out te.ffm
./ffm-train -k 4 -t 50 -s $nr_thread -p te.ffm tr.ffm model
./ffm-predict te.ffm model te.out
./utils/calibrate.py te.out te.out.cal