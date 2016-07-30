# import hashlib, csv, math, os, pickle, subprocess
#
# def read_freqent_feats(threshold=10):
#     frequent_feats = set()
#     for row in csv.DictReader(open('../fc.trva.t10.txt')):
#         if int(row['Total']) < threshold:
#             continue
#         frequent_feats.add(row['Field']+'-'+row['Value'])
#     return frequent_feats
#
# frequent_feats = read_freqent_feats(10)
# for feat in frequent_feats:
#     print(feat)




# headers = ["Label"]
# for i in range(1, 73):
#     headers.append("C{}".format(i))
#
# print(" ".join(headers))
#
HEADER = "Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,I14,I15,I16,I17,I18,I19,I20,I21,I22,I23,I24,I25,I26,I27,I28,I29,I30,I31,I32,I33,I34,I35,I36,I37,I38,I39,I40,I41,I42,I43,I44,I45,I46,I47,I48,I49,I50,I51,I52,I53,I54,I55,I56,I57,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13"
arr = HEADER.split(",")
print(" ".join(arr))

# strF = 'C7-10,C15-300,C7-2,C13-1,C7-1,C16-250,C16-600,C2-27989d,C11-b6536a,C4-5,C2-387068,C8-10,C6-30685,C8-14,C4-4,C12-2913,C6-10059,C2-16f332,C6-29526'
# target_cat_feats = strF.split(',')
# print (target_cat_feats)


