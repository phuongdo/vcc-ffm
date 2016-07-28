#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

from common import *

parser = argparse.ArgumentParser()
parser.add_argument('-n', '--nr_bins', type=int, default=int(1e+6))
parser.add_argument('-t', '--threshold', type=int, default=int(10))
parser.add_argument('csv_path', type=str)
parser.add_argument('gbdt_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())


def gen_hashed_fm_feats(feats, nr_bins):
    feats = ['{0}:{1}:1'.format(field - 1, hashstr(feat, nr_bins)) for (field, feat) in feats]
    return feats


frequent_feats = read_freqent_feats(args['threshold'])

print(args['csv_path'])

with open(args['out_path'], 'w') as f:
    for row, line_gbdt in zip(csv.DictReader(open(args['csv_path']), delimiter=' '), open(args['gbdt_path'])):
        feats = []
        for feat in gen_feats(row):
            field = feat.split('-')[0]
            type, field = field[0], int(field[1:])
            if type == 'C' and feat not in frequent_feats:
                feat = feat.split('-')[0] + 'less'
            if type == 'C':
                field += 0  # 13>>0
            feats.append((field, feat))

        # for i, feat in enumerate(line_gbdt.strip().split()[1:], start=1):
        #     field = i + 75  #
        #     feats.append((field, str(i) + ":" + feat))

        feats = gen_hashed_fm_feats(feats, args['nr_bins'])
        f.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
