#!/usr/bin/env python3

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('dense_path', type=str)
parser.add_argument('sparse_path', type=str)
args = vars(parser.parse_args())

# target_cat_feats = ['C9-a73ee510', 'C22-', 'C17-e5ba7672', 'C26-', 'C23-32c7478e', 'C6-7e0ccccf', 'C14-b28479f6', 'C19-21ddcdc9', 'C14-07d13a8f', 'C10-3b08e48b', 'C6-fbad5c96', 'C23-3a171ecb', 'C20-b1252a9d', 'C20-5840adea', 'C6-fe6b92e5', 'C20-a458ea53', 'C14-1adce6ef', 'C25-001f3601', 'C22-ad3062eb', 'C17-07c540c4', 'C6-', 'C23-423fab69', 'C17-d4bb7bd8', 'C2-38a947a1', 'C25-e8b83407', 'C9-7cc72ec2']
# target_cat_feats = ['C1-', 'C2-', 'C3-', 'C4-', 'C5-', 'C6-', 'C7-', 'C8-', 'C9-', 'C10-', 'C11-', 'C12-', 'C13-',
#                     'C14-', 'C15-', 'C16-', 'C17-', 'C18-']


# this target variables are taken from XgBoost Tree ( contact Quang_)
#
# field:7	value:10
# field:15	value:300
# field:7	value:2
# field:13	value:1
# field:7	value:1
# field:16	value:250
# field:16	value:600
# field:2	value:27989d
# field:11	value:b6536a
# field:4	value:5
# field:2	value:387068
# field:8	value:10
# field:6	value:30685
# field:8	value:14
# field:4	value:4
# field:12	value:2913
# field:6	value:10059
# field:2	value:16f332
# field:6	value:29526

# 7,15,13,16,2,11,4,2,8,6,12



#
# C1: weekday C1
# C2: hour C2
# C3: domain (*) C3
# #C4: path
# C5: geo(*) C4
# #C6: campaignId
# C7: zoneId (*) C5
# C8: os_code (*) C6
# C9: browser_code(*) C7
# C10: age C8
# C11: sex  C9
# C12: banname(*) C10
# C13: label_id(*)  C11
# # C14: filetype
# # C15: url
# C16: width (*)  C12
# C17: height (*)  C13
# #C18: bantype


# C1: weekday
# C2: hour
# C3: domain
# C4: geo
# C5: zoneId
# C6: os_code
# C7: browser_code
# C8: age
# C9: sex
# C10: banname
# C11: label_id
# C12: width
# C13: height


strF = 'C6-10,C6-1,C12-300,C6-2,C13-250,C13-600,C3-27989d,C10-b6536a,C4-5,C3-387068,C7-10,C5-30685,C7-14,C4-4,C11-2913,C5-10059,C3-16f332,C5-29526'
target_cat_feats = strF.split(',')

with open(args['dense_path'], 'w') as f_d, open(args['sparse_path'], 'w') as f_s:
    for row in csv.DictReader(open(args['csv_path']), delimiter=' '):
        feats = []
        for j in range(1, 58):
            val = row['I{0}'.format(j)]
            if val == '':
                val = -10
            feats.append('{0}'.format(val))
        f_d.write(row['Label'] + ' ' + ' '.join(feats) + '\n')

        cat_feats = set()
        for j in range(1, 14):
            field = 'C{0}'.format(j)
            key = field + '-' + row[field]
            cat_feats.add(key)

        # print(cat_feats)
        feats = []
        for j, feat in enumerate(target_cat_feats, start=1):
            if feat in cat_feats:
                feats.append(str(j))
        f_s.write(row['Label'] + ' ' + ' '.join(feats) + '\n')
