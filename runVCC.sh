#!/usr/bin/env bash

./utils/count.py tr.csv  > fc.trva.t10.txt
converters/parallelizer-b.py -s 4 converters/pre-b.py tr.csv te.out tr.ffm
converters/parallelizer-b.py -s 4 converters/pre-b.py te.csv te.out te.ffm
./ffm-train -k 4 -t 18 -s 4 -p te.ffm tr.ffm model
./ffm-predict te.ffm model te.out