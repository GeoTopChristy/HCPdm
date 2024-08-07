import os
import numpy as np
import pandas as pd
import time
import itertools
from itertools import permutations
import functools
import networkx as nx


####functions

num=1


def convert_data(x):
    str1 = x
    d = int((len(x) - 1) / 2)
    list1 = [list(str1)[i] for i in [2 * j - 1 for j in range(1, d + 1)]]
    list2 = list(map(int, list1))
    return (list2)


def alphabet(x):
    k = ''.join(sorted(x, key=str.lower))
    return k


def IncreaseOne(x):
    return [i + 1 for i in x]


def sort_order(i):
    i.sort()
    return i


def conver_face1(i, dicti_vert):
    face = [k for k, v in dicti_vert.items() if i in v]
    face = ''.join(face)
    return face


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


def select_columns(i):
    return [dicti[j] for j in i]


##lanner_connected_saver_killer

def recap0(x):
    if x == 2:
        return 0
    else:
        return 1


def spl_tuple(t):
    a = int(t / 10)
    b = t % 10
    return (a, b)


def node_generate_lanner(x):
    d = len(x)
    G = nx.Graph()
    recap = [recap0(_) for _ in list(x)]
    if d == 3:
        dicti = {12: 1, 13: 2, 23: 3}
        recap_dicti = [list(dicti.keys())[_] for _ in range(d) if recap[_] == 1]
    if d == 6:
        dicti = {12: 1, 13: 2, 14: 3, 23: 4, 24: 5, 34: 6}
        recap_dicti = [list(dicti.keys())[_] for _ in range(d) if recap[_] == 1]

        # print(recap_dicti)
    add_edges = [spl_tuple(_) for _ in recap_dicti]

    # print(add_edges)
    G.add_edges_from(add_edges)
    # print(G.nodes)

    N = 3 * d - 6

    adM = np.zeros([N, N])

    for col_label, row_label in G.edges():
        adM[row_label - 1, row_label - 1] = 1
        adM[col_label - 1, col_label - 1] = 1
        adM[col_label - 1, row_label - 1] = 1
        adM[row_label - 1, col_label - 1] = 1


    A = range(1, N + 1)
    edge = np.zeros([N, N]).astype(int)
    deg = np.zeros([1, N]).astype(int)

    for i in range(0, N):
        for j in range(0, N):
            if i == j:
                continue
            if (adM[i, j] == 1):
                edge[i, deg[0, i]] = j + 1
                deg[0, i] += 1

    g = [[j + 1, [i for i in edge[j] if i != 0]] for j in range(N)]

    return (g)

    # nodes = [[1, [7]], [2, [3]], [3, [7]], [4, [6]], [5, [6]], [6, [5]], [7, [3]]]


def count_components(nodes):
    sets = {}
    for node in nodes:
        sets[node[0]] = DisjointSet()
    for node in nodes:
        for vtx in node[1]:
            sets[node[0]].union(sets[vtx])
    return len(set(x.find() for x in sets.values()))  ##python3 sets.itervalues-->setsvalues


class DisjointSet(object):
    def __init__(self):
        self.parent = None

    def find(self):
        if self.parent is None: return self
        return self.parent.find()

    def union(self, other):
        them = other.find()
        us = self.find()
        if them != us:
            us.parent = them


##killing and saving

def killer(j, df):
    col_check = select_columns(cdt[j])
    return list(df[[str(j) for j in col_check]]) not in lib


def saver(j, df):
    col_check = select_columns(cdt[j])
    angleForCheckDF = df[[str(j) for j in col_check]]
    if count_components(node_generate_lanner(list(angleForCheckDF))) == 1:
        return list(angleForCheckDF) in lib
    else:
        return (1 == 1)


#####all_connected
def node_generate(x):
    G = nx.Graph()
    recap = [recap0(_) for _ in list(data.iloc[x])]
    recap_dicti = [list(dicti.keys())[_] for _ in range(21) if recap[_] == 1]

    # print(recap_dicti)
    add_edges = [spl_tuple(_) for _ in recap_dicti]

    # print(add_edges)
    G.add_edges_from(add_edges)
    # print(G.nodes)

    adM = np.zeros([7, 7]).astype(int)

    # print(adM)

    for col_label, row_label in G.edges():
        adM[row_label - 1, row_label - 1] = 1
        adM[col_label - 1, col_label - 1] = 1
        adM[col_label - 1, row_label - 1] = 1
        adM[row_label - 1, col_label - 1] = 1

    N = adM.shape[0]
    A = range(1, N + 1)
    edge = np.zeros([N, N]).astype(int)
    deg = np.zeros([1, N]).astype(int)

    for i in range(0, N):
        for j in range(0, N):
            if i == j:
                continue
            if (adM[i, j] == 1):
                edge[i, deg[0, i]] = j + 1
                deg[0, i] += 1

    g = [[j + 1, [i for i in edge[j] if i != 0]] for j in range(7)]

    return (g)


### data
key_data_list = range(1, 17)
value_data_list = [1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 27, 28, 29, 30, 31]
data_dicti = {key: value for (key, value) in zip(key_data_list, value_data_list)}

AdmissiblePVert = {1: [[6, 1, 5, 2, 4, 3]],
                   2: [[6, 4, 3, 5, 2], [6, 4, 1, 5, 0], [6, 2, 1, 3, 0]],
                   3: [[5, 4, 3, 6, 2], [6, 4, 1, 5, 0]],
                   4: [[5, 4, 3, 6, 2], [6, 4, 1, 5, 0]],
                   5: [[5, 4, 3, 6, 2], [5, 4, 1, 6, 0]],
                   6: [[5, 4, 3, 6, 2], [5, 4, 1, 6, 0]],
                   7: [[5, 4, 3, 6, 2]],
                   8: [[5, 4, 3, 6, 2]],
                   9: [[5, 4, 3, 6, 2]],
                   10: [[5, 4, 3, 6, 2]],
                   11: [[5, 4, 3, 6, 2]],
                   12: [],
                   13: [],
                   14: [],
                   15: [],
                   16: []}




def change_cols(x):
    return ([10 * i + j for i, j in [sort_order(k) for k in [list(_) for _ in list(itertools.combinations(x, 2))]]])


key_selcols_list = range(1, 17)
value_selcols_list = [[change_cols(IncreaseOne(_)) for _ in AdmissiblePVert[i]] for i in range(1, 17)]

sel_cols_dicti = {key: value for (key, value) in zip(key_selcols_list, value_selcols_list)}

Rank3EllipticInPVertices = {1: [[1, 2, 3], [1, 3, 5], [3, 5, 6], [2, 3, 6], [1, 2, 4], [1, 4, 5], [4, 5, 6], [2, 4, 6]],
                            2: [[2, 3, 6], [2, 4, 6], [2, 3, 4], [3, 5, 6], [4, 5, 6], [3, 4, 5],
                                [0, 1, 4], [0, 1, 6], [0, 4, 6], [1, 4, 5], [1, 5, 6], [4, 5, 6],
                                [0, 1, 2], [0, 1, 6], [0, 2, 6], [1, 2, 3], [1, 3, 6], [2, 3, 6]],
                            3: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6],
                                [0, 1, 4], [0, 1, 6], [0, 4, 6], [1, 4, 5], [1, 5, 6], [4, 5, 6]],
                            4: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6],
                                [0, 1, 4], [0, 1, 6], [0, 4, 6], [1, 4, 5], [1, 5, 6], [4, 5, 6]],
                            5: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6],
                                [0, 1, 4], [0, 1, 5], [0, 4, 5], [1, 4, 6], [1, 5, 6], [4, 5, 6]],
                            6: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6],
                                [0, 1, 4], [0, 1, 5], [0, 4, 5], [1, 4, 6], [1, 5, 6], [4, 5, 6]],
                            7: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6]],
                            8: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6]],
                            9: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6]],
                            10: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6]],
                            11: [[2, 3, 4], [2, 3, 5], [2, 4, 5], [3, 4, 6], [3, 5, 6], [4, 5, 6]],
                            12: [],
                            13: [],
                            14: [],
                            15: [],
                            16: []}

#
Node4Rank2InPVertices = {1: [[1, 2, 5, 6], [1, 3, 4, 6], [2, 3, 4, 5]],
                         2: [[0, 2, 3, 5]],
                         3: [[0, 2, 5, 6]],
                         4: [[0, 2, 5, 6]],
                         5: [],
                         6: [],
                         7: [],
                         8: [],
                         9: [],
                         10: [],
                         11: [],
                         12: [],
                         13: [],
                         14: [],
                         15: [],
                         16: []}
#
Node3Rank2InPVertices = {1: [],
                         2: [[3, 4, 6], [1, 4, 6], [1, 2, 6]],
                         3: [[3, 4, 5], [1, 4, 6]],
                         4: [[3, 4, 5], [1, 4, 6]],
                         5: [[3, 4, 5], [1, 4, 5]],
                         6: [[3, 4, 5], [1, 4, 5]],
                         7: [[3, 4, 5]],
                         8: [[3, 4, 5]],
                         9: [[3, 4, 5]],
                         10: [[3, 4, 5]],
                         11: [[3, 4, 5]],
                         12: [],
                         13: [],
                         14: [],
                         15: [],
                         16: []}

Node2Rank1InPVertices = {1: [[1, 6], [2, 5], [3, 4]],
                         2: [[2, 5], [0, 5]],
                         3: [[2, 6], [0, 5]],
                         4: [[2, 6], [0, 5]],
                         5: [[2, 6], [0, 6]],
                         6: [[2, 6], [0, 6]],
                         7: [[2, 6]],
                         8: [[2, 6]],
                         9: [[2, 6]],
                         10: [[2, 6]],
                         11: [[2, 6]],
                         12: [],
                         13: [],
                         14: [],
                         15: [],
                         16: []}

####### library

print("sphrical library", "*" * 90)

path = "./ToolPolytope"
# path = "c:/CoxeterPolytope/toolPolytope"
# path = "/home/ftzheng/project/Coxeter/toolPolytope"
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

print("3-6 spherical sizes:", len(ts3), len(ts4), len(ts5), len(ts6))

print("euclidean library", "*" * 90)
e3lis = pd.read_csv(path + "/ElisV/E3lis.txt", sep=' ', names=range(1, 4))
e3lis = e3lis.values.tolist()
te3 = [[int(j) for j in e3lis[i]] for i in range(len(e3lis))]  ###更新e3，用旧的就可以

e4lis = pd.read_csv(path + "/ElisV/E4lis.txt", sep=' ', names=range(1, 7))  ### 更新e4
e4lis = e4lis.values.tolist()
te4 = [[int(j) for j in e4lis[i]] for i in range(len(e4lis))]

tse4k = ts4 + te4
###242+30

e5lis = pd.read_csv(path + "/ElisV/E5lis.txt", sep=' ', names=range(1, 11))  ### 更新e5
e5lis = e5lis.values.tolist()
te5 = [[int(j) for j in e5lis[i]] for i in range(len(e5lis))]

e6lis = pd.read_csv(path + "/ElisV/E6lis.txt", sep=' ', names=range(1, 16))  ### 更新e6
e6lis = e6lis.values.tolist()
te6 = [[int(j) for j in e6lis[i]] for i in range(len(e6lis))]

print("3-6 euclidean sizes:", len(te3), len(te4), len(te5), len(te6))

print("size of vertices block", "*" * 80)

e4alis = pd.read_csv(path + "/ElisV/EA3lis.txt", sep=' ', names=range(1, 7))  ###更改文件名
e4alis = e4alis.values.tolist()
te4a = [[int(j) for j in e4alis[i]] for i in range(len(e4alis))]
tv4 = ts4 + te4a
####242+27（只考虑连通的椭圆图）


e5alis = pd.read_csv(path + "/ElisV/EA2A1lis.txt", sep=' ', names=range(1, 11))  ###更改文件名
e5alis = e5alis.values.tolist()
tv5 = [[int(j) for j in e5alis[i]] for i in range(len(e5alis))]

e6alis = pd.read_csv(path + "/ElisV/E3A1lis.txt", sep=' ', names=range(1, 16))  ###更改文件名
e6alis = e6alis.values.tolist()
tv6 = [[int(j) for j in e6alis[i]] for i in range(len(e6alis))]

Nl4 = len(tv4)  # 带t的是套集，得落在套子里
Nl5 = len(tv5)
Nl6 = len(tv6)

print("numbers of vertice-candidates:", Nl4, Nl5, Nl6)

print("lanner library", "*" * 90)  ## L4是3-simplex

l3lis = pd.read_csv(path + "/Lanner/L3lis.txt", sep=' ', names=range(1, 4))
l3lis = l3lis.values.tolist()
tl3 = [[int(j) for j in l3lis[i]] for i in range(len(l3lis))]

l4lis = pd.read_csv(path + "/Lanner/L4lis.txt", sep=' ', names=range(1, 7))
l4lis = l4lis.values.tolist()
tl4 = [[int(j) for j in l4lis[i]] for i in range(len(l4lis))]

print("3-4 lanner sizes:", len(tl3), len(tl4))

flag = 2
# exec("os.chdir('c:/CoxeterPolytope/pi47/" + str(num) + "')")


d = 7  # number of facets
Nr = 21  # number of all indicies 1+2+..+d-1
dh4 = 6  # length of dihedral angle. eg. 3->3,4->6,5->10.
dh5 = 10
dh6 = 15

alpha_list = []
alpha_list = [chr(x) for x in range(ord('A'), ord('Z') + 1)] + [chr(x) for x in range(ord('a'), ord('z') + 1)]

#file = open('c:/CoxeterPolytope/pi47/pi47.txt')
file = open('./polytopeDATA/pi47.txt')

all_lines = file.readlines()
Vert = all_lines[data_dicti[num] - 1]
file.close()

Vert = list(Vert.split(" "))
Vert = list([convert_data(_) for _ in Vert])
print("Vert:", Vert)

Vert = [IncreaseOne(_) for _ in Vert]
Vert = [sort_order(_) for _ in Vert]

# combinatorial data

key_list = [alpha_list[_] for _ in range(len(Vert))]
value_list = Vert
dicti_vert = {key: value for (key, value) in zip(key_list, value_list)}
value_1 = [conver_face1(i, dicti_vert) for i in range(1, d + 1)]
key_1 = range(d)
dicti_1 = {key: value for (key, value) in zip(key_1, value_1)}
permu_1 = [list(_) for _ in (itertools.combinations(range(d), 2))]

matrix = [[1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1]]

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
print("ultra-parallel pairs:", listOfPositions)
infty = [_[0] * 10 + _[1] for _ in listOfPositions]

key_allin_list = [12, 13, 14, 15, 16, 17,
                  23, 24, 25, 26, 27,
                  34, 35, 36, 37,
                  45, 46, 47,
                  56, 57,
                  67]
lose_value = len(listOfPositions)
value_select_list = range(1, Nr + 1)  ###
key_select_list = key_allin_list  ###
dicti = {key: value for (key, value) in zip(key_select_list, value_select_list)}
loseCols = [i - 1 for i in select_columns(infty)]
print("key_select_list", key_select_list)
print("dicti", dicti)
print("value_1", value_1)
print()

Nv = len(Vert)
Ndisjoint = Nr - lose_value
print("number of vert:", Nv, "number of ultra-parallel:", lose_value)

print("s3", "*" * 90)

Edge = []
Vert4 = [i for i in Vert if len(i) == 4]
for i in Vert4:
    Edge += [list(j) for j in list(itertools.combinations(i, 3))]

addedge = [IncreaseOne(_) for _ in Rank3EllipticInPVertices[num]]
Edge = Edge + addedge
Edge.sort()
Edge = list(Edge for Edge, _ in itertools.groupby(Edge))

virtual_edge_set = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 3))]
edge_set = [set(_) for _ in
            Edge]
non_edge_set = [_ for _ in virtual_edge_set if _ not in edge_set]
non_edge = [sort_order(_) for _ in [list(_) for _ in non_edge_set]]
pre_s3 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_edge]
s3 = [i for i in pre_s3 if list(set(i) & set(infty)) == []]
non_edge_facets = [stimultaneous_spl(_) for _ in s3]
s3cols = [select_columns(_) for _ in s3]
print("s3", "(", len(s3), "):", [[i - 1 for i in j] for j in non_edge_facets])

print("e3", "*" * 90)

e3excep = [IncreaseOne(_) for _ in Node3Rank2InPVertices[num]]
com_e3 = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 3))]

exp_set = [set(_) for _ in e3excep]

left_set = [i for i in com_e3 if i not in exp_set]
# print(len(left_set))
pre_e3 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in left_set]

e3 = [i for i in pre_e3 if list(set(i) & set(infty)) == []]
e3_facets = [stimultaneous_spl(_) for _ in e3]
e3cols = [select_columns(_) for _ in e3]
print("e3", "(", len(e3), "):", [[i - 1 for i in j] for j in e3_facets])

virtual_vert_set = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 4))]
vert4 = [set(_) for _ in Vert4]
non_vert_set = [_ for _ in virtual_vert_set if _ not in vert4]
non_vert = [sort_order(_) for _ in [list(_) for _ in non_vert_set]]
# print(non_vert)
# print(cube_dicti[num])
# cubedata= [IncreaseOne(_) for _ in cube_dicti[num]]
# non_vert_cube = [i for i in non_vert if i not in cubedata]
# print(non_vert_cube)

pre_s4 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_vert]
s4 = [i for i in pre_s4 if list(set(i) & set(infty)) == []]
# s4 = [i for i in pre_s4 if len(set.intersection(set(i), set(infty))) == 0]
non_vert_facets = [stimultaneous_spl(_) for _ in s4]
s4cols = [select_columns(_) for _ in s4]
print("s4", "(", len(s4), ")", ":", [[i - 1 for i in j] for j in non_vert_facets])

print("e4", "*" * 90)

virtual_vert_set = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 4))]
vert4 = [set(_) for _ in Vert4]
non_vert_set = [_ for _ in virtual_vert_set if _ not in vert4]
non_vert = [sort_order(_) for _ in [list(_) for _ in non_vert_set]]
# print(non_vert)
# print(cube_dicti[num])
cubedata = [IncreaseOne(_) for _ in Node4Rank2InPVertices[num]]
non_vert_cube = [i for i in non_vert if i not in cubedata]

pre_e4 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
          non_vert_cube]
e4 = [i for i in pre_e4 if list(set(i) & set(infty)) == []]
non_vert_cube_facets = [stimultaneous_spl(_) for _ in e4]
e4cols = [select_columns(_) for _ in e4]
print("e4", "(", len(e4), ")", ":", [[i - 1 for i in j] for j in non_vert_cube_facets])

print("l4", "*" * 90)
quasi_edge = Edge + [IncreaseOne(_) for _ in Node3Rank2InPVertices[num]]
quasi_edge_set = [set(_) for _ in quasi_edge]

temp = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_vert_cube]
templ4 = [i for i in temp if list(set(i) & set(infty)) == []]
non_vert_f = [stimultaneous_spl(_) for _ in templ4]

###Edge check
def check_nonvert_all_edge(x):
    edge_list = list(itertools.combinations(non_vert_f[x], 3))
    edge_list = [set(list(_)) for _ in edge_list]
    judge = [_ for _ in edge_list if _ in quasi_edge_set]
    if edge_list == judge:
        return x
    return "F"

select_lanner = [check_nonvert_all_edge(_) for _ in range(len(non_vert_f))]
trick = ["F"]
l_index = [_ for _ in select_lanner if _ not in trick]
l4 = [templ4[_] for _ in l_index]
l4_cols = [select_columns(_) for _ in l4]
l4_facets = [stimultaneous_spl(_) for _ in l4]
print("l4", "(", len(l4), "):", [[i - 1 for i in j] for j in l4_facets])

print("l3", "*" * 90)

###去掉距离对
l3_facets_pre = [list(_) for _ in non_edge_facets]
deleted = [IncreaseOne(_) for _ in Node3Rank2InPVertices[num]]
l3_facets = [_ for _ in l3_facets_pre if _ not in deleted]
l3_facets = [sort_order(_) for _ in l3_facets]
l3 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
      l3_facets]
l3_cols = [select_columns(_) for _ in l3]
print(l3_cols)
print("l3", "(", len(l3), "):", [[i - 1 for i in j] for j in l3_facets])

print("s5", "*" * 90)

pre_s5 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
          list(itertools.combinations(range(1, d + 1), 5))]
s5 = [i for i in pre_s5 if list(set(i) & set(infty)) == []]
s5_facets = [stimultaneous_spl(_) for _ in s5]
s5cols = [select_columns(_) for _ in s5]
print("s5", "(", len(s5), "):", [[i - 1 for i in j] for j in s5_facets])

print("e5", "*" * 90)

com_e5 = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 5))]
Vert5 = [i for i in Vert if len(i) == 5]
vert_set = [set(_) for _ in Vert5]
non_vert = [i for i in com_e5 if i not in vert_set]
pre_e5 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_vert]
e5 = [i for i in pre_e5 if list(set(i) & set(infty)) == []]
non_vert_facets = [stimultaneous_spl(_) for _ in e5]
e5cols = [select_columns(_) for _ in e5]
print("e5", "(", len(e5), ")", ":", [[i - 1 for i in j] for j in non_vert_facets])

print("s6", "*" * 90)

pre_s6 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
          list(itertools.combinations(range(1, d + 1), 6))]
s6 = [i for i in pre_s6 if list(set(i) & set(infty)) == []]
s6_facets = [stimultaneous_spl(_) for _ in s6]
s6cols = [select_columns(_) for _ in s6]
print("s6", "(", len(s6), "):", [[i - 1 for i in j] for j in s6_facets])

print("e6", "*" * 90)

com_e6 = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 6))]
Vert6 = [i for i in Vert if len(i) == 6]
vert_set = [set(_) for _ in Vert6]
non_vert = [i for i in com_e6 if i not in vert_set]
pre_e6 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_vert]
e6 = [i for i in pre_e6 if list(set(i) & set(infty)) == []]
non_vert_facets = [stimultaneous_spl(_) for _ in e6]
e6cols = [select_columns(_) for _ in e6]
print("e6", "(", len(e6), ")", ":", [[i - 1 for i in j] for j in non_vert_facets])


#___________________________________________________________________________________________
##cover
vert = []
link = []
cover = []
for i in range(Nv):
    vert = vert + [[10 * i + j for i, j in list(itertools.combinations(Vert[i], 2))]]

cover = cover + [sort_order(list(set(vert[0]).union(set(vert[1]))))]
link = [list(set(vert[0]).intersection(set(vert[1])))]

for k in range(2, Nv):
    link = link + [list(set(cover[k - 2]).intersection(set(vert[k])))]
    cover = cover + [sort_order(list(set(cover[k - 2]).union(set(vert[k]))))]

cover_list = [vert[0]] + cover

print("link:", link)
print("cover_list:", cover_list)

### ks_list

range_s3 = range(len(s3))
range_s4 = range(len(s4))
range_s5 = range(len(s5))
range_s6 = range(len(s6))

range_e3 = range(len(e3))
range_e4 = range(len(e4))
range_e5 = range(len(e5))
range_e6 = range(len(e6))
# range_i4 = range(len(i4))
# range_l4_basis = range(len(l4_basis))
range_l3 = range(len(l3))
range_l4 = range(len(l4))

cancel_list = []
for i in range(len(Vert)):
    temp = []
    temps3 = []
    temps4 = []
    temps5 = []
    temps6 = []

    tempe3 = []
    tempe4 = []
    tempe5 = []
    tempe6 = []
    templ3 = []
    templ4 = []
    #     templ4_basis = []
    #     tempi4 = []

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

    for j in range_s6:
        temp1 = s6[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            temps6 += [j]
            range_s6 = [_ for _ in range_s6 if _ not in temps6]

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

    for j in range_e6:
        temp1 = e6[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            tempe6 += [j]
            range_e6 = [_ for _ in range_e6 if _ not in tempe6]

    for j in range_l3:
        temp1 = l3[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            templ3 += [j]
            range_l3 = [_ for _ in range_l3 if _ not in templ3]

    for j in range_l4:
        temp1 = l4[j]
        judge_temp = [k for k in temp1 if k in cover_list[i]]
        if len(temp1) == len(judge_temp):
            templ4 += [j]
            range_l4 = [_ for _ in range_l4 if _ not in templ4]

    temp += [
        [i, temps3, temps4, temps5, temps6, tempe3, tempe4, tempe5, tempe6, templ3, templ4]]
    cancel_list += temp

print("cancel_list:", cancel_list)

print()

condition_all = (s3, s4, s5, s6, e3, e4, e5, e6, l3, l4)  ## cols
library_all = (ts3, ts4, ts5, ts6, te3, te4, te5, te6, tl3, tl4)  ## admitting angles
kind_all = ("s3", "s4", "s5", "s6", "e3", "e4", "e5", "e6", "l3", "l4")  ## type

condition = [condition_all[x - 1] for x in [1, 5, 2, 6, 9, 10]]
library = [library_all[x - 1] for x in [1, 5, 2, 6, 9, 10]]
kind = [kind_all[x - 1] for x in [1, 5, 2, 6, 9, 10]]

print("pasting + round 1 killing:", "*" * 100)

select_cancel_list = [[i[j] for j in [0, 1, 5, 2, 6, 9, 10]] for i in cancel_list]
print("select_cancel_list:", select_cancel_list)

exec("os.chdir('./output/P7_" + str(num) + "')")
#______________________________________________________________________________


T = []
t1 = time.time()
if len(vert[0])==6:
    sel_cols = select_columns(vert[0])
else:
    sel_cols= select_columns(sel_cols_dicti[num][0])
r=len(Vert[0])
exec("Nl=Nl"+str(r)+"")
df_1 = np.zeros([Nl, Nr])
exec("dh=dh" +str(r)+ "")
exec("tv=tv" +str(r)+ "")

df_1[:, [j - 1 for j in sel_cols]] = pd.DataFrame(tv, dtype=np.int8, columns=list(range(1, dh + 1)))
non=[[8]*lose_value]*Nl
df_1[:, loseCols] = pd.DataFrame(non, dtype=np.int8, columns=list(range(1, lose_value+1)))
data = pd.DataFrame(df_1, dtype=np.int8, columns=[str(_) for _ in list(range(1, Nr + 1))])

df_1[:, [j - 1 for j in sel_cols]] = pd.DataFrame(tv, dtype=np.int8, columns=list(range(1, dh + 1)))
data = pd.DataFrame(df_1, dtype=np.int8, columns=[str(_) for _ in list(range(1, Nr + 1))])
t2 = time.time()
Ttemp = ["df1:", data.shape, t2 - t1]
T += [Ttemp]
print(Ttemp)

for k in range(len(select_cancel_list[0]) - flag, len(select_cancel_list[0])):
    if select_cancel_list[0][k] != []:
        cdt = condition[k - 1]  # cdt,lib,kind i-1

        lib = library[k - 1]

        for j in select_cancel_list[0][k]:
            t1 = time.time()
            saving = functools.partial(saver, j)
            data[str(Nr + 1)] = data.apply(saving, axis=1)
            data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
            t2 = time.time()
            Ttemp = ["df1", kind[k - 1], select_cancel_list[0][k], data.shape[0], t2 - t1]
            T += [Ttemp]
            print(Ttemp)

for k in range(1, len(select_cancel_list[0])- flag):
    if select_cancel_list[0][k] != []:
        cdt = condition[k - 1]  # cdt,lib,kind i-1
        lib = library[k - 1]
        for j in select_cancel_list[0][k]:
            t1 = time.time()
            killing = functools.partial(killer, j)
            data[str(Nr + 1)] = data.apply(killing, axis=1)
            # print(data)
            data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
            t2 = time.time()
            Ttemp = ["df1", kind[k - 1], select_cancel_list[0][k], data.shape[0], t2 - t1]
            T += [Ttemp]
            print(Ttemp)


t1 = time.time()
if len(vert[1])==6:
    sel_cols = select_columns(vert[1])
else:
    sel_cols= select_columns(sel_cols_dicti[num][1])
r=len(Vert[1])
exec("Nl=Nl"+str(r)+"")
df_2 = np.zeros([Nl, Nr])
exec("dh=dh" +str(r)+ "")
exec("tv=tv" +str(r)+ "")
df_2[:, [j - 1 for j in sel_cols]] = pd.DataFrame(tv, dtype=np.int8, columns=list(range(1, dh + 1)))
df_2 = pd.DataFrame(df_2, dtype=int, columns=[str(name) for name in list(range(1, Nr + 1))])


# merging by linking
tem = sort_order(select_columns(link[0]))
print(tem)
tem1 = list(set(range(1, Nr + 1)) - set(tem))
data = pd.merge(df_2, data, on=[str(_) for _ in tem])
# selecting result columns and form data
for j in tem1:
    data[str(j)] = data[str(j) + "_x"] + data[str(j) + "_y"]
data = data[[str(_) for _ in range(1, Nr + 1)]]
t2 = time.time()
Ttemp = ["df2:", data.shape, t2 - t1]
T += [Ttemp]
print(Ttemp)

for k in range(len(select_cancel_list[0]) - flag, len(select_cancel_list[0])):
    if select_cancel_list[1][k] != []:
        cdt = condition[k - 1]
        lib = library[k - 1]
        for j in select_cancel_list[1][k]:
            t1 = time.time()
            saving = functools.partial(saver, j)
            data[str(Nr + 1)] = data.apply(saving, axis=1)
            # print(data)
            data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
            t2 = time.time()
            Ttemp = ["df2", [kind[k - 1], select_cancel_list[1][k], data.shape[0], t2 - t1]]  ###1
            T += [Ttemp]
            print(Ttemp)

for k in range(1, len(select_cancel_list[0])- flag):
    if select_cancel_list[1][k] != []:  ###1
        cdt = condition[k - 1]
        lib = library[k - 1]
        for j in select_cancel_list[1][k]:
            t1 = time.time()
            killing = functools.partial(killer, j)
            data[str(Nr + 1)] = data.apply(killing, axis=1)
            # print(data)
            data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
            t2 = time.time()
            Ttemp = ["df2", [kind[k - 1], select_cancel_list[1][k], data.shape[0], t2 - t1]]  ###1
            T += [Ttemp]
            print(Ttemp)


tstar = time.time()

for i in range(2, Nv):  # 【Verice-2】
    t1 = time.time()
    if len(vert[i])==6:
        sel_cols = select_columns(vert[i])
    else:
        sel_cols= select_columns(sel_cols_dicti[num][i])
    #sel_cols = select_columns(vert[0]) ###对应关系在sel_cols处改
    r=len(Vert[i])
    exec("Nl=Nl"+str(r)+"")
    temp = "df_" + str(i + 1)
    exec(temp + "=np.zeros([Nl,Nr])")  # 【剩余量】
    exec("dh=dh" +str(r)+ "")
    exec("tv=tv" +str(r)+ "")
    exec(temp + "[:,[ j-1 for j in sel_cols]]=pd.DataFrame(tv,dtype=np.int8,columns=list(range(1,dh+1)))")
    exec(temp + "=pd.DataFrame(" + temp + ",dtype=np.int8,columns=[str(_) for _ in list(range(1,Nr+1))])")  # 【剩余量】+1
    tem = sort_order(select_columns(link[i - 1]))
    tem1 = list(set(range(1, Nr + 1)) - set(tem))
    exec("data=pd.merge(" + temp + ",data,on=[str(_) for _ in tem])")  # 【剩余量】*2+1
    for j in tem1:
        data[str(j)] = data[str(j) + "_x"] + data[str(j) + "_y"]
    data = data[[str(_) for _ in range(1, Nr + 1)]]
    t2 = time.time()
    Ttemp = ["df", i + 1, data.shape, t2 - t1]
    T += [Ttemp]
    print(Ttemp)

    if data.shape[0] == 0:
        break

    for k in range(len(select_cancel_list[0]) - flag, len(select_cancel_list[0])):
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

    for k in range(1, len(select_cancel_list[0]) - flag):
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

                Ttemp = ["df", i + 1, [kind[k - 1], select_cancel_list[i][k], data.shape[0], t2 - t1]]  ###1
                T += [Ttemp]
                print(Ttemp)



tend = time.time()
print("total time consuming:", tend - tstar)
print("data round1:", data.shape)

#_____________________________________________________________________________________________________
##connectedness
# %%time
dataNumberRound1=data.shape[0]
con_list=[count_components(node_generate(_)) for _ in range(data.shape[0])]
# print(len([i+1 for i in range(data.shape[0]) if con_list[i]==1]))
con_labs_add1=[i+1 for i in range(dataNumberRound1) if con_list[i]==1]
con_labs=[i for i in range(dataNumberRound1) if con_list[i]==1]
data=data.iloc[con_labs]

print("data connected:", data.shape)


##graph isomorphic

permu7 = [list(_) for _ in list(itertools.permutations([1, 2, 3, 4, 5, 6, 7], 7))]
sym_cardinality = len(permu7)

for i in range(sym_cardinality):
    sym_key_list = range(1, 8)
    sym_value_list = permu7[i]
    nam = "dir_" + str(i + 1)
    exec(nam + "={key: value for (key, value) in zip(sym_key_list, sym_value_list)}")

kl_spl = [spl(_) for _ in key_select_list]


def key_list_after_sym(x):
    m = []
    for i in range(1, sym_cardinality + 1):
        exec("temp = sort_order([dir_" + str(i) + "[j] for j in x])")
        exec("m += [10*temp[0]+temp[1]]")
    return m


kl_matrix = np.transpose(np.array([key_list_after_sym(_) for _ in kl_spl]))
kl_columns = np.apply_along_axis(select_columns, 1, kl_matrix)

merTOstr = lambda x: ''.join([str(_) for _ in x])


def pre_generate(onedata, pos):
    pos = [i - 1 for i in pos]  #
    return [onedata[i] for i in pos]


def generate(onedata):
    tempfunction = functools.partial(pre_generate, onedata)
    temp = np.apply_along_axis(tempfunction, 1, kl_columns)
    temp = [merTOstr(_) for _ in temp]
    return temp


data_array = np.array(data)

tstar = time.time()

per_value= [frozenset(i) for i in [generate(_) for _ in data_array]]
tend = time.time()
print("permu time consuming:", tend - tstar)

per_key=range(len(per_value))

dicti_per={key: value for (key, value) in zip(per_key, per_value)}
temp={val:key for key,val in dicti_per.items()}
res={val:key for key, val in temp.items()}

lis_per=res.keys()
lis_per_order=sorted(lis_per)
data_after_per=[list(data_array[i]) for i in lis_per_order]
print("data equivalent:",len(data_after_per))

data = pd.DataFrame(data_after_per, dtype=int, columns=[str(_) for _ in range(1, Nr + 1)])


#___________________________________________________________
##round2killing


#round 2

print()

condition = [condition_all[x - 1] for x in [3, 4, 7, 8]]
library = [library_all[x - 1] for x in [3, 4, 7, 8]]
kind = [kind_all[x - 1] for x in [3, 4, 7, 8]]

print("pasting+round 2 killing:", "*" * 100)
select_cancel_list = [[i[j] for j in [0, 3, 4, 7, 8]] for i in cancel_list]
print("select_cancel_list:", select_cancel_list)



tstar = time.time()

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



####
def replace7(row):
    to_modify = data.iloc[row, :]
    indexes = [i for i, x in enumerate(to_modify) if x == 7]
    replacements = [(i + 1) * 10 + 7 for i in range(len(indexes))]
    for index in range(len(indexes)):
        to_modify[indexes[index] + 1] = replacements[index]
    return ()
####



####
def replace8(row):
    to_modify = data.iloc[row, :]
    indexes = [i for i, x in enumerate(to_modify) if x == 8]
    replacements = [(i + 1) * 10 + 8 for i in range(len(indexes))]
    for index in range(len(indexes)):
        to_modify[indexes[index] + 1] = replacements[index]
    return ()
####



# chenage 7

tstart = time.time()
data.columns = range(1, Nr + 1)
temp = [replace7(_) for _ in range(len(data))]
temp = [replace8(_) for _ in range(len(data))]
tend = time.time()
print("change7/8 time consuming:", tend - tstart)

exec("data.to_csv('P7_" + str(num) + "_LSE_con_per_change7_new.txt', header=None, index=None, sep=' ', mode='a')")

