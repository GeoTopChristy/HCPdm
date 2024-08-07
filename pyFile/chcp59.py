import os
import numpy as np
import pandas as pd
import time
import itertools
from itertools import permutations
import functools

# exec("os.chdir('/home/ftzheng/project/Coxeter/P9/P8_" + str(num) + "/flag" + str(flag) + "')")
# exec("os.chdir('d:/polytope/P9/g6/p" + str(num) + "')")
# exec("os.chdir('/home/ftzheng/project/Coxeter/p9/g6/p" + str(num) + "/flag" + str(flag) + "')")

num = 322  # input the number of polytope
flag = 2  # flag=2 means basis yes, flag=1 means basis no

##functions
def conver_face1(i, dicti_vert):
    face = [k for k, v in dicti_vert.items() if i in v]
    face = ''.join(face)
    return face


def conver_face2(i):
    my_dict = dicti_vert
    my_color = i
    my_face = [k for k, v in my_dict.items() if my_color in v]
    face = ''.join(my_face)
    return face


def sort_order(i):
    i.sort()
    return i


def convert_data5(x):
    str1 = x
    list1 = [list(str1)[i] for i in [1, 3, 5, 7, 9]]  ## 5维的所以取5个数
    list2 = list(map(int, list1))
    return list2


def select_columns(i):
    return [dicti[j] for j in i]


def IncreaseOne(x):
    return [i + 1 for i in x]


def alphabet(x):
    k = ''.join(sorted(x, key=str.lower))
    return k


def getIndexes(dfObj, value):
    listOfPos = []
    result = dfObj.isin([value])
    seriesObj = result.any()
    columnNames = list(seriesObj[seriesObj == True].index)
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
        for row in rows:
            listOfPos.append([row, col])
    return listOfPos


################## killing set
def spl(t):
    a = int(t / 10)
    b = t % 10
    return [a, b]


def stimultaneous_spl(t):
    c = [spl(_) for _ in t]
    d = []
    for i in range(len(t)):
        d += c[i]
    return set(d)


###Edge check
def check_nonvert_all_edge(x):
    edge_list = list(itertools.combinations(non_vert_facets[x], 4))
    edge_list = [set(list(_)) for _ in edge_list]
    judge = [_ for _ in edge_list if _ in Edge]
    if edge_list == judge:
        return x
    return "F"


def check_nonedge_all_face(x):
    face_list = list(itertools.combinations(s4_facets[x], 3))  ## s4_facets: non_face_facets
    face_list = [set(list(_)) for _ in face_list]
    judge = [_ for _ in face_list if _ in Face]
    if face_list == judge:
        return x
    return "F"


def infty2(x):
    if len(set([x[j] for j in [0, 5]]) & set(infty)) == 2 and len(set([x[j] for j in [1, 2, 3, 4]]) & set(infty)) == 0:
        return [x[j] for j in [1, 2, 3, 4]]
    if len(set([x[j] for j in [1, 4]]) & set(infty)) == 2 and len(set([x[j] for j in [0, 2, 3, 5]]) & set(infty)) == 0:
        return [x[j] for j in [0, 2, 3, 5]]
    if len(set([x[j] for j in [2, 3]]) & set(infty)) == 2 and len(set([x[j] for j in [0, 1, 4, 5]]) & set(infty)) == 0:
        return [x[j] for j in [0, 1, 4, 5]]
    return None


###killing and saving

def killer(j, df):
    col_check = select_columns(cdt[j])
    return list(df[[str(j) for j in col_check]]) not in lib


def saver(j, df):
    col_check = select_columns(cdt[j])
    return list(df[[str(j) for j in col_check]]) in lib


##########get permu

def hasha(L):
    return (''.join([str(q) for q in L]))


def getper(s):
    vert = []
    for q in range(len(Vert)):  ####len(Vert)
        vert += [sort_order([s[i - 1] for i in Vert[q]])]
    v = map(hasha, vert)
    if set(V) == set(v):
        return s


#####

# the following are per functions

def key_list_after_sym(x):
    global sort_order
    m = []
    for i in range(1, sym_cardinality + 1):
        exec("temp = sort_order([dir_" + str(i) + "[j] for j in x])")
        exec("m += [10*temp[0]+temp[1]]")
    return m


merTOstr = lambda x: ''.join([str(_) for _ in x])


def pre_generate(onedata, pos):
    pos = [i - 1 for i in pos]  # -1是由于list切片的原因
    return [onedata[i] for i in pos]


def generate(onedata):
    tempfunction = functools.partial(pre_generate, onedata)
    temp = np.apply_along_axis(tempfunction, 1, kl_columns)
    temp = [merTOstr(_) for _ in temp]
    return temp


# def conver_preresult(i):
#   return [int(j) for j in list([list(_) for _ in list(pre_result)][i][0])]

def conver_preresult(i):
    return [int(j) for j in data_after_per[i]]
alpha_list = [chr(x) for x in range(ord('A'), ord('Z') + 1)]


####
def replace7(row):
    to_modify = data.iloc[row, :]
    indexes = [i for i, x in enumerate(to_modify) if x == 7]
    replacements = [(i + 1) * 10 + 7 for i in range(len(indexes))]
    for index in range(len(indexes)):
        to_modify[indexes[index] + 1] = replacements[index]
    return ()
####

d = 9  # number of facets
dp = 36  # number of all indicies 1+2+..+d-1
dh = 10  # length of dihedral angle. eg. 3->3,4->6,5->10...

# file = open('/home/ftzheng/project/Coxeter/P8_v1/P8/4d8f.txt')
# file = open('d:/polytope/p9/5d9f.txt')
file = open('./polytopeDATA/5d9m.txt')

all_lines = file.readlines()
Vert = all_lines[num - 1]  ####-1
file.close()

Vert = list(Vert.split(" "))
Vert = Vert[:-1]
Vert = list([convert_data5(_) for _ in Vert])
print(Vert)

Vert = [IncreaseOne(_) for _ in Vert]
Vert = sort_order([sort_order(_) for _ in Vert])
print(Vert)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('hi, PyCharm')

# combinatorial data

key_list = [alpha_list[_] for _ in range(len(Vert))]
value_list = Vert
dicti_vert = {key: value for (key, value) in zip(key_list, value_list)}
value_1 = [conver_face1(i, dicti_vert) for i in range(1, d + 1)]
key_1 = range(d)
dicti_1 = {key: value for (key, value) in zip(key_1, value_1)}
permu_1 = [list(_) for _ in (itertools.combinations(range(d), 2))]

matrix = [[1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1, 1]]

for i in permu_1:
    matrix[i[0]][i[1]] = "".join(sorted(set.intersection(set(list(dicti_1[i[0]])), set(list(dicti_1[i[1]])))))
    if matrix[i[0]][i[1]] == '':
        matrix[i[0]][i[1]] = "0"

flat_matrix = list(itertools.chain(*matrix))
print("hyper-parallel number:", flat_matrix.count("0"))

df_matrix = pd.DataFrame(matrix, columns=range(1, d + 1), index=range(1, d + 1))
print("adjacent matrix:")
print(df_matrix)

listOfPositions = getIndexes(df_matrix, "0")
print("hyper-parallel pairs:", listOfPositions)
infty = [_[0] * 10 + _[1] for _ in listOfPositions]

key_allin_list = [12, 13, 14, 15, 16, 17, 18, 19,
                  23, 24, 25, 26, 27, 28, 29,
                  34, 35, 36, 37, 38, 39,
                  45, 46, 47, 48, 49,
                  56, 57, 58, 59,
                  67, 68, 69,
                  78, 79,
                  89]

lose_value = len(listOfPositions)
value_select_list = range(1, (dp - lose_value + 1))
key_select_list = [i for i in key_allin_list if i not in infty]
dicti = {key: value for (key, value) in zip(key_select_list, value_select_list)}
print("key_list", key_select_list)
print("dicti", dicti)
print("value_1", value_1)
print()

####### library

print("sphrical library", "*" * 100)

path = "./ToolPolytope"
s3lis = pd.read_csv(path + "/Slis/S3lis.txt", sep=' ', names=range(1, 4))
s3lis = s3lis.values.tolist()
ts3 = [[int(j) for j in s3lis[i]] for i in range(len(s3lis))]

s4lis = pd.read_csv(path + "/Slis/S4lis.txt", sep=' ', names=range(1, 7))
s4lis = s4lis.values.tolist()
ts4 = [[int(j) for j in s4lis[i]] for i in range(len(s4lis))]

s5lis = pd.read_csv(path + "/Slis/S5lis.txt", sep=' ', names=range(1, 11))
s5lis = s5lis.values.tolist()
ts5 = [[int(j) for j in s5lis[i]] for i in range(len(s5lis))]

s6lis = pd.read_csv(path + "/Slis/S6lis.txt", sep=' ', names=range(1, 16))
s6lis = s6lis.values.tolist()
ts6 = [[int(j) for j in s6lis[i]] for i in range(len(s6lis))]

s7lis = pd.read_csv(path + "/Slis/S7lis.txt", sep=' ', names=range(1, 21))
s7lis = s7lis.values.tolist()
ts7 = [[int(j) for j in s7lis[i]] for i in range(len(s7lis))]

print("3-7 spherical sizes:", len(ts3), len(ts4), len(ts5), len(ts6), len(ts7))

print("euclidean library", "*" * 99)
e3lis = pd.read_csv(path + "/Elis/E3lis.txt", sep=' ', names=range(1, 4))
e3lis = e3lis.values.tolist()
te3 = [[int(j) for j in e3lis[i]] for i in range(len(e3lis))]

e4lis = pd.read_csv(path + "/Elis/E4lis.txt", sep=' ', names=range(1, 7))
e4lis = e4lis.values.tolist()
te4 = [[int(j) for j in e4lis[i]] for i in range(len(e4lis))]

e5lis = pd.read_csv(path + "/Elis/E5lis.txt", sep=' ', names=range(1, 11))
e5lis = e5lis.values.tolist()
te5 = [[int(j) for j in e5lis[i]] for i in range(len(e5lis))]

e6lis = pd.read_csv(path + "/Elis/E6lis.txt", sep=' ', names=range(1, 16))
e6lis = e6lis.values.tolist()
te6 = [[int(j) for j in e6lis[i]] for i in range(len(e6lis))]

e7lis = pd.read_csv(path + "/Elis/E7lis.txt", sep=' ', names=range(1, 21))
e7lis = e7lis.values.tolist()
te7 = [[int(j) for j in e7lis[i]] for i in range(len(e7lis))]

print("3-7 euclidean sizes:", len(te3), len(te4), len(te5), len(te6), len(te7))

print("lanner library", "*" * 102)
l4lis = pd.read_csv(path + "/Lanner/L4lis.txt", sep=' ', names=range(1, 7))
l4lis = l4lis.values.tolist()
tl4 = [[int(j) for j in l4lis[i]] for i in range(len(l4lis))]

l5lis = pd.read_csv(path + "/Lanner/L5lis.txt", sep=' ', names=range(1, 11))
l5lis = l5lis.values.tolist()
tl5 = [[int(j) for j in l5lis[i]] for i in range(len(l5lis))]

tl5_basis = [[2]]
print("4-5 lanner sizes:", len(tl4), len(tl5), "5 basis sizes:", len(tl5_basis))

print("infty library", "*" * 103)
ti4 = [[2] * 4]
print("infty2 sizes:", len(ti4))
print("*" * 117)

###### combinatorial
print()
Nv = len(Vert)
Nr = len(key_select_list)
Nl = len(ts5)  # 带t的是验证集，不带t是测试集
print("number of vert:", Nv, "number of non-hyp-parallel:", Nr, "number of spherical candidates:", Nl)

print()

###### killing set
print("killer set:", "*" * 100)
virtual_vert_set = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 5))]  # 四元组
vert_set = [set(_) for _ in Vert]  # 顶点
non_vert_set = [_ for _ in virtual_vert_set if _ not in vert_set]  ##四元组里非顶点集
non_vert = [sort_order(_) for _ in [list(_) for _ in non_vert_set]]  ##排序且转list后的非顶点集
pre_s5 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_vert]  # 十进制非顶点集
s5 = [i for i in pre_s5 if list(set(i) & set(infty)) == []]  ##无平行对的十进制非顶点集
non_vert_facets = [stimultaneous_spl(_) for _ in s5]
s5cols = [select_columns(_) for _ in s5]
print("s5", "(", len(s5), ")", ":", non_vert_facets)

pre_e5 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
          list(itertools.combinations(range(1, d + 1), 5))]
e5 = [i for i in pre_e5 if list(set(i) & set(infty)) == []]
e5_facets = [stimultaneous_spl(_) for _ in e5]
e5cols = [select_columns(_) for _ in e5]
print("e5", "(", len(e5), "):", e5_facets)

Edge = []
for i in Vert:
    Edge += [list(j) for j in list(itertools.combinations(i, 4))]
Edge.sort()
Edge = list(Edge for Edge, _ in itertools.groupby(Edge))
Edge = [set(_) for _ in Edge]
######
virtual_edge_set = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 4))]  # 三元组
edge_set = [set(_) for _ in Edge]  # 边
non_edge_set = [_ for _ in virtual_edge_set if _ not in edge_set]  ##三元组里非顶点集
non_edge = [sort_order(_) for _ in [list(_) for _ in non_edge_set]]  ##排序且转list后的非边集
pre_s4 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_edge]  # 十进制非边集
s4 = [i for i in pre_s4 if list(set(i) & set(infty)) == []]  ##无平行对的十进制非顶点集
s4_facets = [stimultaneous_spl(_) for _ in s4]
s4cols = [select_columns(_) for _ in s4]
print("s4", "(", len(s4), "):", s4_facets)

pre_e4 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
          list(itertools.combinations(range(1, d + 1), 4))]
e4 = [i for i in pre_e4 if list(set(i) & set(infty)) == []]
e4_facets = [stimultaneous_spl(_) for _ in e4]
e4cols = [select_columns(_) for _ in e4]
print("e4", "(", len(e4), "):", e4_facets)

temp_i4 = [infty2(_) for _ in pre_e4]
i4 = [x for x in temp_i4 if x is not None]  # 23 45 36..
# print("i41", i4)
i4 = [sort_order(_) for _ in i4]
# print("i42", i4)
i4_cols = [select_columns(_) for _ in i4]  # 1,2,28..
# print("i4_cols", i4_cols)
i4_facets = [stimultaneous_spl(_) for _ in i4]  # {2,3,4,5}..
print("i4", "(", len(i4), "):", i4_facets)

Face = []
for i in Vert:
    Face += [list(j) for j in list(itertools.combinations(i, 3))]
Face.sort()
Face = list(Face for Face, _ in itertools.groupby(Face))
Face = [set(_) for _ in Face]

virtual_face_set = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 3))]  # 三元组
face_set = [set(_) for _ in Face]  # 面
non_face_set = [_ for _ in virtual_face_set if _ not in face_set]  ##三元组里非顶2-面(3个， 每个4维, 4-1-1（3-1）个1
non_face = [sort_order(_) for _ in [list(_) for _ in non_face_set]]  ##排序且转list后的非边集
pre_s3 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_face]  # 十进制非边集
s3 = [i for i in pre_s3 if list(set(i) & set(infty)) == []]  ##无平行对的十进制非顶点集
s3_facets = [stimultaneous_spl(_) for _ in s3]
s3cols = [select_columns(_) for _ in s3]
print("s3", "(", len(s3), "):", s3_facets)

pre_e3 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
          list(itertools.combinations(range(1, d + 1), 3))]
e3 = [i for i in pre_e3 if list(set(i) & set(infty)) == []]
e3_facets = [stimultaneous_spl(_) for _ in e3]
e3cols = [select_columns(_) for _ in e3]
print("e3", "(", len(e3), "):", e3_facets)

pre_se6 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
           list(itertools.combinations(range(1, d + 1), 6))]
se6 = [i for i in pre_se6 if list(set(i) & set(infty)) == []]
se6_facets = [stimultaneous_spl(_) for _ in se6]
se6cols = [select_columns(_) for _ in se6]
print("se6", "(", len(se6), "):", se6_facets)

pre_se7 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
           list(itertools.combinations(range(1, d + 1), 7))]
se7 = [i for i in pre_se7 if list(set(i) & set(infty)) == []]
se7_facets = [stimultaneous_spl(_) for _ in se7]
se7cols = [select_columns(_) for _ in se7]
print("se7", "(", len(se7), "):", se7_facets)

###lanner
# 4-lanner
select_lanner = [check_nonedge_all_face(_) for _ in range(len(s4_facets))]
trick = ["F"]
l_index = [_ for _ in select_lanner if _ not in trick]
l4 = [s4[_] for _ in l_index]
l4_cols = [select_columns(_) for _ in l4]
l4_facets = [stimultaneous_spl(_) for _ in l4]
print("l4", "(", len(l4), "):", l4_facets)

# 5-lanner


select_lanner = [check_nonvert_all_edge(_) for _ in range(len(non_vert_facets))]
trick = ["F"]
l_index = [_ for _ in select_lanner if _ not in trick]
l5 = [s5[_] for _ in l_index]
l5_cols = [select_columns(_) for _ in l5]
l5_facets = [stimultaneous_spl(_) for _ in l5]
print("l5", "(", len(l5), "):", l5_facets)

l5_bf = ([_ + 1 for _ in range(len(value_1)) if len(value_1[_]) == 5])
l5_basis = []
l5_basis_facets = []

for k in l5_bf:
    temp = sort_order([i + 1 for i in range(d) if matrix[k - 1][i] != 1 if matrix[k - 1][i] != '0'] +
                      [i + 1 for i in range(d) if matrix[i][k - 1] != 1 if matrix[i][k - 1] != '0'])
    l5_basis_facets += [temp]
    temp2 = [sort_order([k, q]) for q in temp]
    l5_basis += [10 * i + j for [i, j] in temp2]

l5_basis = [[i] for i in sort_order(l5_basis)]
print("l5_basis", "(", len(l5_bf), "):", [[l5_bf[i], l5_basis_facets[i]] for i in range(len(l5_bf))])
### 找cover  作ks_list (killing&saving list)

vert = []
link = []
cover = []
for i in range(Nv):
    vert = vert + [[10 * i + j for i, j in list(itertools.combinations(Vert[i], 2))]]  ##vert是Vert的十进制权重

cover = cover + [sort_order(list(set(vert[0]).union(set(vert[1]))))]
link = [list(set(vert[0]).intersection(set(vert[1])))]

for k in range(2, Nv):
    link = link + [list(set(cover[k - 2]).intersection(set(vert[k])))]
    cover = cover + [sort_order(list(set(cover[k - 2]).union(set(vert[k]))))]

cover_list = [vert[0]] + cover

print("link:", link)
print("cover_list:", cover_list)

### 造ks_list

range_s3 = range(len(s3))
range_s4 = range(len(s4))
range_s5 = range(len(s5))
range_se6 = range(len(se6))
range_se7 = range(len(se7))
range_e3 = range(len(e3))
range_e4 = range(len(e4))
range_e5 = range(len(e5))
range_l4 = range(len(l4))
range_l5 = range(len(l5))
range_l5_basis = range(len(l5_basis))  ##一个一个2验算
range_i4 = range(len(i4))

### cancel_list

cancel_list = []
for i in range(len(Vert)):
    temp = []
    temps3 = []
    temps4 = []
    temps5 = []
    tempse6 = []
    tempse7 = []
    tempe3 = []
    tempe4 = []
    tempe5 = []
    templ4 = []
    templ5 = []
    templ5_basis = []
    tempi4 = []

    for j in range_s3:
        temp1 = s3[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            temps3 += [j]
            range_s3 = [_ for _ in range_s3 if _ not in temps3]

    for j in range_s4:
        temp1 = s4[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            temps4 += [j]
            range_s4 = [_ for _ in range_s4 if _ not in temps4]

    for j in range_s5:
        temp1 = s5[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            temps5 += [j]
            range_s5 = [_ for _ in range_s5 if _ not in temps5]

    for j in range_se6:
        temp1 = se6[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            tempse6 += [j]
            range_se6 = [_ for _ in range_se6 if _ not in tempse6]

    for j in range_se7:
        temp1 = se7[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            tempse7 += [j]
            range_se7 = [_ for _ in range_se7 if _ not in tempse7]

    for j in range_e3:
        temp1 = e3[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            tempe3 += [j]
            range_e3 = [_ for _ in range_e3 if _ not in tempe3]

    for j in range_e4:
        temp1 = e4[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            tempe4 += [j]
            range_e4 = [_ for _ in range_e4 if _ not in tempe4]

    for j in range_e5:
        temp1 = e5[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            tempe5 += [j]
            range_e5 = [_ for _ in range_e5 if _ not in tempe5]

    for j in range_l4:
        temp1 = l4[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            templ4 += [j]
            range_l4 = [_ for _ in range_l4 if _ not in templ4]

    for j in range_l5:
        temp1 = l5[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            templ5 += [j]
            range_l5 = [_ for _ in range_l5 if _ not in templ5]

    for j in range_l5_basis:
        temp1 = l5_basis[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            templ5_basis += [j]
            range_l5_basis = [_ for _ in range_l5_basis if _ not in templ5_basis]

    for j in range_i4:
        temp1 = i4[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            tempi4 += [j]
            range_i4 = [_ for _ in range_i4 if _ not in tempi4]

    temp += [
        [i, temps3, temps4, temps5, tempse6, tempse7, tempe3, tempe4, tempe5, tempse6, tempse7, tempi4, templ5_basis,
         templ4, templ5]]
    cancel_list += temp

print("cancel_list:", cancel_list)
print()

condition_all = (s3, s4, s5, se6, se7, e3, e4, e5, se6, se7, i4, l5_basis, l4, l5)  ## cols
library_all = (ts3, ts4, ts5, ts6, ts7, te3, te4, te5, te6, te7, ti4, tl5_basis, tl4, tl5)  ## admitting angles
kind_all = ("s3", "s4", "s5", "s6", "s7", "e3", "e4", "e5", "e6", "e7", "i4", "l5_basis", "l4", "l5")  ## type

condition = [condition_all[x - 1] for x in [12, 13, 14]]
library = [library_all[x - 1] for x in [12, 13, 14]]
kind = [kind_all[x - 1] for x in [12, 13, 14]]

print("pasting + round 1 killing:", "*" * 100)
select_cancel_list = [[i[j] for j in [0, 12, 13, 14]] for i in cancel_list]
print("select_cancel_list:", select_cancel_list)

tstar = time.time()
T = []
t1 = time.time()
sel_cols = select_columns(vert[0])
df_1 = np.zeros([Nl, Nr])  # 【剩余量】
df_1[:, [j - 1 for j in sel_cols]] = pd.DataFrame(ts5, dtype=np.int8, columns=list(range(1, dh + 1)))  # 【10个二面角】
data = pd.DataFrame(df_1, dtype=np.int8, columns=[str(_) for _ in list(range(1, Nr + 1))])
t2 = time.time()
Ttemp = ["df1:", data.shape, t2 - t1]
T += [Ttemp]
print(Ttemp)

for k in range(len(select_cancel_list[0]) - flag - 1, len(select_cancel_list[0])):
    if select_cancel_list[0][k] != []:
        cdt = condition[k - 1]  # cdt,lib,kind都是i-1
        lib = library[k - 1]
        for j in select_cancel_list[0][k]:  ###这里是取0,后面是参数选取
            t1 = time.time()
            saving = functools.partial(saver, j)
            data[str(Nr + 1)] = data.apply(saving, axis=1)
            # print(data)
            data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
            t2 = time.time()
            Ttemp = ["df1", kind[k - 1], select_cancel_list[0][k], data.shape[0], t2 - t1]
            T += [Ttemp]
            print(Ttemp)

t1 = time.time()
sel_cols = select_columns(vert[1])
df_2 = np.zeros([Nl, Nr])  # 【剩余量】
df_2[:, [j - 1 for j in sel_cols]] = pd.DataFrame(ts5, dtype=np.int8, columns=list(range(1, dh + 1)))
df_2 = pd.DataFrame(df_2, dtype=int, columns=[str(name) for name in list(range(1, Nr + 1))])
# merging by linking
tem = sort_order(select_columns(link[0]))
print(tem)
tem1 = list(set(range(1, Nr + 1)) - set(tem))  # 【剩余量】+1
data = pd.merge(df_2, data, on=[str(_) for _ in tem])  # 【剩余量】*2+1
# selecting result columns and form data
for j in tem1:
    data[str(j)] = data[str(j) + "_x"] + data[str(j) + "_y"]
data = data[[str(_) for _ in range(1, Nr + 1)]]
t2 = time.time()
Ttemp = ["df2:", data.shape, t2 - t1]
T += [Ttemp]
print(Ttemp)

for k in range(len(select_cancel_list[0]) - flag - 1, len(select_cancel_list[0])):
    if select_cancel_list[1][k] != []:  ###1
        cdt = condition[k - 1]  # cdt,lib,kind都是i-1
        lib = library[k - 1]
        for j in select_cancel_list[1][k]:  ###这里是取0,后面是参数选取 ###1
            t1 = time.time()
            saving = functools.partial(saver, j)
            data[str(Nr + 1)] = data.apply(saving, axis=1)
            # print(data)
            data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
            t2 = time.time()
            Ttemp = ["df2", [kind[k - 1], select_cancel_list[1][k], data.shape[0], t2 - t1]]  ###1
            T += [Ttemp]
            print(Ttemp)


for i in range(2, Nv):  # 【Verice-2】
    t1 = time.time()
    sel_cols = select_columns(vert[i])
    temp = "df_" + str(i + 1)
    exec(temp + "=np.zeros([Nl,Nr])")  # 【剩余量】
    exec(temp + "[:,[ j-1 for j in sel_cols]]=pd.DataFrame(ts5,dtype=np.int8,columns=list(range(1,dh+1)))")
    exec(temp + "=pd.DataFrame(" + temp + ",dtype=np.int8,columns=[str(_) for _ in list(range(1,Nr+1))])")  # 【剩余量】+1
    tem = sort_order(select_columns(link[i - 1]))
    tem1 = list(set(range(1, Nr + 1)) - set(tem))
    # exec("data=pd.merge(" + temp + ",data,on=[str(_) for _ in tem])")  # 【剩余量】*2+1
    exec("data_temp=pd.merge(" + temp + ",data,on=[str(_) for _ in tem])");
    data = locals()['data_temp']
    for j in tem1:
        data[str(j)] = data[str(j) + "_x"] + data[str(j) + "_y"]
    data = data[[str(_) for _ in range(1, Nr + 1)]]
    t2 = time.time()
    Ttemp = ["df", i + 1, data.shape, t2 - t1]
    T += [Ttemp]
    print(Ttemp)

    if data.shape[0] == 0:
        break

    for k in range(len(select_cancel_list[0]) - flag - 1, len(select_cancel_list[0])):
        if select_cancel_list[i][k] != []:  ###1
            cdt = condition[k - 1]  # cdt,lib,kind都是i-1
            lib = library[k - 1]
            for j in select_cancel_list[i][k]:  ###这里是取0,后面是参数选取 ###1
                t1 = time.time()
                saving = functools.partial(saver, j)
                data[str(Nr + 1)] = data.apply(saving, axis=1)
                # print(data)
                data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
                t2 = time.time()

                Ttemp = ["df", i + 1, [kind[k - 1], select_cancel_list[i][k], data.shape[0], t2 - t1]]  ###1
                T += [Ttemp]
                print(Ttemp)



tend = time.time()
print("total time consuming:", tend - tstar)
print("data round1:", data.shape)

#### get permutation
S9 = list(permutations(range(1, 10), 9))  # 40320
S9 = [list(i) for i in S9]

###Vert is defined before
tstart = time.time()
V = [hasha(_) for _ in Vert]
rare_per = [getper(_) for _ in S9]
per = pd.DataFrame([i for i in rare_per if i is not None], dtype=np.int8, columns=list(range(1, d + 1)))
###已经给出per之后可以把这个注掉
exec("per.to_csv('output/P9_322/P9_" + str(num) + "_per.txt',header=None, index=None, sep=' ', mode='a')")
tend = time.time()
print("get per time consuming2:", tend - tstart)
print("per:", per)

### after per
# after per
# per = pd.read_csv("P8_10_per.txt", sep=' ', names=range(1, d + 1))
# exec("per = pd.read_csv('P8_" + str(num) + "_per.txt', sep=' ', names=range(1, d + 1))")
# exec("os.chdir('d:/polytope/p8_" + str(num) + "10/test'")
data.columns = range(1, Nr + 1)  # 更改column名是因为之前per和change7是在另一个文档按照这个列名写的
sym_cardinality = int(per.shape[0])

for i in range(sym_cardinality):
    sym_key_list = range(1, int(per.shape[1]) + 1)
    sym_value_list = list(per.iloc[i])
    nam = "dir_" + str(i + 1)
    exec(nam + "={key: value for (key, value) in zip(sym_key_list, sym_value_list)}")

kl_spl = [spl(_) for _ in key_select_list]
kl_matrix = np.transpose(np.array([key_list_after_sym(_) for _ in kl_spl]))
kl_columns = np.apply_along_axis(select_columns, 1, kl_matrix)

tstart = time.time()
data_array = np.array(data)
pre_result = set(frozenset(i) for i in [generate(_) for _ in data_array])
print(len(pre_result))
tend = time.time()
print("permu time consuming1:", tend - tstart)

tstart = time.time()
pre_temp = [list(_) for _ in list(pre_result)]
pre_temp = [pre_temp[i][0] for i in range(len(pre_result))]
data_after_per = [list(j) for j in pre_temp]
data_after_per = [conver_preresult(_) for _ in range(len(pre_temp))]

tend = time.time()
print("permu time consuming2:", tend - tstart)

data = pd.DataFrame(data_after_per, dtype=int,
                    columns=[str(_) for _ in range(1, Nr + 1)])  # columns name 不用str,否则replace函数的切片指标会差1

exec("data.to_csv('output/P9_322/P9_" + str(num) + "_LSIEr1_per.txt', header=None, index=None, sep=' ', mode='a')")

#round 2

print()

condition = [condition_all[x - 1] for x in [1, 6, 7, 11]]
library = [library_all[x - 1] for x in [1, 6, 7, 11]]
kind = [kind_all[x - 1] for x in [1, 6, 7, 11]]

print("pasting+round 2 killing:", "*" * 100)
select_cancel_list = [[i[j] for j in [0, 1, 6, 7, 11]] for i in cancel_list]
print("select_cancel_list:", select_cancel_list)

tstart = time.time()

for i in range(Nv):
    for k in range(1, len(select_cancel_list[0])):
        if select_cancel_list[i][k] != []:  ###1
            cdt = condition[k - 1]  # cdt,lib,kind都是i-1
            lib = library[k - 1]
            for j in select_cancel_list[i][k]:  ###这里是取0,后面是参数选取 ###1
                t1 = time.time()
                killing = functools.partial(killer, j)
                data[str(Nr + 1)] = data.apply(killing, axis=1)
                # print(data)
                data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
                t2 = time.time()
                Ttemp = ["df", i + 1, ":", kind[k - 1], select_cancel_list[i][k], data.shape[0], t2 - t1]
                T += [Ttemp]  ###1
                print(Ttemp)



tend = time.time()
print("total time consuming:", tend - tstar)
print("data round 2:", data.shape)

exec("data.to_csv('output/P9_322/P9_" + str(num) + "_LSIEr2_per.txt', header=None, index=None, sep=' ', mode='a')")

# round 3
print()
print("pasting+round 3 killing:", "*" * 100)
select_cancel_list = [[i[j] for j in [0, 2, 3, 4, 5, 8, 9, 10]] for i in cancel_list]
print("select_cancel_list:", select_cancel_list)

condition = [condition_all[x - 1] for x in [2, 3, 4, 5, 8, 9, 10]]
library = [library_all[x - 1] for x in [2, 3, 4, 5, 8, 9, 10]]
kind = [kind_all[x - 1] for x in [2, 3, 4, 5, 8, 9, 10]]

tstart = time.time()

for i in range(Nv):
    for k in range(1, len(select_cancel_list[0])):
        if select_cancel_list[i][k] != []:  ###1
            cdt = condition[k - 1]  # cdt,lib,kind都是i-1
            lib = library[k - 1]
            for j in select_cancel_list[i][k]:  ###这里是取0,后面是参数选取 ###1
                t1 = time.time()
                killing = functools.partial(killer, j)
                data[str(Nr + 1)] = data.apply(killing, axis=1)
                # print(data)
                data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
                t2 = time.time()
                Ttemp = ["df", i + 1, ":", kind[k - 1], select_cancel_list[i][k], data.shape[0], t2 - t1]
                T += [Ttemp]  ###1
                print(Ttemp)

tend = time.time()
print("total time consuming:", tend - tstar)
print("data round 3:", data.shape)

exec("data.to_csv('output/P9_322/P9_" + str(num) + "_LSIEr3_per.txt', header=None, index=None, sep=' ', mode='a')")

# chenage 7

tstart = time.time()
data.columns = range(1, Nr + 1)
temp = [replace7(_) for _ in range(len(data))]
tend = time.time()
print("change7 time consuming:", tend - tstart)

exec("data.to_csv('output/P9_322/P9_" + str(num) + "_LSIE_per_change7.txt', header=None, index=None, sep=' ', mode='a')")



