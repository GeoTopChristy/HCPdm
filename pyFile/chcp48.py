import numpy as np
import pandas as pd
import time
import itertools
import functools

def run48(Vert):
    '''
    :param Vert:
    :return:
    '''
    print('PyCharm start')

    #global functions

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

    # the following are per functions

    def key_list_after_sym(x):
        m = []
        for i in range(1, sym_cardinality + 1):
            exec("temp = sort_order([dir_" + str(i) + "[j] for j in x])")
            exec("m += [10*temp[0]+temp[1]]")
        return m

    merTOstr = lambda x: ''.join([str(_) for _ in x])

    def pre_generate(onedata, pos):
        pos = [i - 1 for i in pos]  # -1 due to a uniform indexing
        return [onedata[i] for i in pos]

    def generate(onedata):
        tempfunction = functools.partial(pre_generate, onedata)
        temp = np.apply_along_axis(tempfunction, 1, kl_columns)
        temp = [merTOstr(_) for _ in temp]
        return temp

    def conver_preresult(i):
        return [int(j) for j in data_after_per[i]]

    def replace7(row):
        to_modify = data.iloc[row, :]
        indexes = [i for i, x in enumerate(to_modify) if x == 7]
        replacements = [(i + 1) * 10 + 7 for i in range(len(indexes))]
        for index in range(len(indexes)):
            to_modify[indexes[index] + 1] = replacements[index]
        return ()


    #local functions

    def select_columns(i):
        return [dicti[j] for j in i]

    def infty2(x):
        if len(set([x[j] for j in [0, 5]]) & set(infty)) == 2 and len(
                set([x[j] for j in [1, 2, 3, 4]]) & set(infty)) == 0:
            return [x[j] for j in [1, 2, 3, 4]]
        if len(set([x[j] for j in [1, 4]]) & set(infty)) == 2 and len(
                set([x[j] for j in [0, 2, 3, 5]]) & set(infty)) == 0:
            return [x[j] for j in [0, 2, 3, 5]]
        if len(set([x[j] for j in [2, 3]]) & set(infty)) == 2 and len(
                set([x[j] for j in [0, 1, 4, 5]]) & set(infty)) == 0:
            return [x[j] for j in [0, 1, 4, 5]]
        return None


    def check_nonvert_all_edge(x):
        edge_list = list(itertools.combinations(non_vert_facets[x], 3))
        edge_list = [set(list(_)) for _ in edge_list]
        judge = [_ for _ in edge_list if _ in Edge]
        if edge_list == judge:
            return x
        return "F"

    def killer(j, df):
        col_check = select_columns(cdt[j])
        return list(df[[str(j) for j in col_check]]) not in lib

    def saver(j, df):
        col_check = select_columns(cdt[j])
        return list(df[[str(j) for j in col_check]]) in lib


    # data
    alpha_list = [chr(x) for x in range(ord('A'), ord('Z') + 1)]

    d = 8  # number of facets
    dp = 28  # number of all indexes 1+2+..+d-1
    dh = 6  # length of dihedral angle. eg. 3->3,4->6,5->10...



    print(Vert)

    key_list = [alpha_list[_] for _ in range(len(Vert))]
    value_list = Vert
    dicti_vert = {key: value for (key, value) in zip(key_list, value_list)}
    value_1 = [conver_face1(i, dicti_vert) for i in range(1, d + 1)]
    key_1 = range(d)
    dicti_1 = {key: value for (key, value) in zip(key_1, value_1)}
    permu_1 = [list(_) for _ in (itertools.combinations(range(d), 2))]

    matrix = [[1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]]

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

    key_allin_list = [12, 13, 14, 15, 16, 17, 18,
                      23, 24, 25, 26, 27, 28,
                      34, 35, 36, 37, 38,
                      45, 46, 47, 48,
                      56, 57, 58,
                      67, 68,
                      78]
    lose_value = len(listOfPositions)
    value_select_list = range(1, (dp - lose_value + 1))
    key_select_list = [i for i in key_allin_list if i not in infty]
    dicti = {key: value for (key, value) in zip(key_select_list, value_select_list)}
    print("key_list", key_select_list)
    print("dicti", dicti)

    print()

    #library

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

    print("euclidean library", "*" * 100)
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

    print("lanner library", "*" * 100)
    l4lis = pd.read_csv(path + "/Lanner/L4lis.txt", sep=' ', names=range(1, 7))
    l4lis = l4lis.values.tolist()
    tl4 = [[int(j) for j in l4lis[i]] for i in range(len(l4lis))]
    print("4 lanner sizes:", len(tl4))

    tl4_basis = [[2]]
    print("4 lanner basis sizes:", len(tl4_basis))

    print("infty library", "*" * 100)
    ti4 = [[2] * 4]
    print("4 infty2 sizes:", len(ti4))
    print("*" * 150)

    Nv = len(Vert)
    Nr = len(key_select_list)
    Nl = len(ts4)  # ts4 is verify set and s4 is test set
    print("number of vert:", Nv, "number of non-hyp-parallel:", Nr, "number of spherical candidates:", Nl)

    print()
	
    #generate test sets

    print("killer set:", "*" * 100)
    virtual_vert_set = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 4))]
    vert_set = [set(_) for _ in Vert]
    non_vert_set = [_ for _ in virtual_vert_set if _ not in vert_set]
    non_vert = [sort_order(_) for _ in [list(_) for _ in non_vert_set]]
    pre_s4 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_vert]
    s4 = [i for i in pre_s4 if list(set(i) & set(infty)) == []]
    non_vert_facets = [stimultaneous_spl(_) for _ in s4]
    s4cols = [select_columns(_) for _ in s4]
    print("s4", "(", len(s4), ")", ":", non_vert_facets)

    pre_e4 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
              list(itertools.combinations(range(1, d + 1), 4))]
    e4 = [i for i in pre_e4 if list(set(i) & set(infty)) == []]
    e4_facets = [stimultaneous_spl(_) for _ in e4]
    e4cols = [select_columns(_) for _ in e4]
    print("e4", "(", len(e4), "):", e4_facets)

    temp_i4 = [infty2(_) for _ in pre_e4]
    i4 = [x for x in temp_i4 if x is not None] ## 23 45 36..
    ## print("i41", i4)
    i4 = [sort_order(_) for _ in i4]
    ## print("i42", i4)
    i4_cols = [select_columns( _) for _ in i4]  ## 1,2,28..
    ## print("i4_cols", i4_cols)
    i4_facets = [stimultaneous_spl(_) for _ in i4]  ## {2,3,4,5}..
    print("i4", "(", len(i4), "):", i4_facets)

    Edge = []
    for i in Vert:
        Edge += [list(j) for j in list(itertools.combinations(i, 3))]
    Edge.sort()
    Edge = list(Edge for Edge, _ in itertools.groupby(Edge))
    Edge = [set(_) for _ in Edge]

    virtual_edge_set = [set(list(_)) for _ in list(itertools.combinations(range(1, d + 1), 3))]
    edge_set = [set(_) for _ in Edge]
    non_edge_set = [_ for _ in virtual_edge_set if _ not in edge_set]
    non_edge = [sort_order(_) for _ in [list(_) for _ in non_edge_set]]
    pre_s3 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in non_edge]
    s3 = [i for i in pre_s3 if list(set(i) & set(infty)) == []]
    non_edge_facets = [stimultaneous_spl(_) for _ in s3]
    s3cols = [select_columns( _) for _ in s3]
    print("s3", "(", len(s3), "):", non_edge_facets)

    pre_e3 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
              list(itertools.combinations(range(1, d + 1), 3))]
    e3 = [i for i in pre_e3 if list(set(i) & set(infty)) == []]
    e3_facets = [stimultaneous_spl(_) for _ in e3]
    e3cols = [select_columns( _) for _ in e3]
    print("e3", "(", len(e3), "):", e3_facets)

    select_lanner = [check_nonvert_all_edge(_) for _ in range(len(non_vert_facets))]
    trick = ["F"]
    l_index = [_ for _ in select_lanner if _ not in trick]
    l4 = [s4[_] for _ in l_index]
    l4_cols = [select_columns( _) for _ in l4]
    l4_facets = [stimultaneous_spl(_) for _ in l4]
    print("l4", "(", len(l4), "):", l4_facets)

    l4_bf = ([_ + 1 for _ in range(len(value_1)) if len(value_1[_]) == 4])
    l4_basis = []
    l4_basis_facets = []

    for k in l4_bf:
        temp = sort_order([i + 1 for i in range(d) if matrix[k - 1][i] != 1 if matrix[k - 1][i] != '0'] +
                          [i + 1 for i in range(d) if matrix[i][k - 1] != 1 if matrix[i][k - 1] != '0'])
        l4_basis_facets += [temp]
        temp2 = [sort_order([k, q]) for q in temp]
        l4_basis += [10 * i + j for [i, j] in temp2]

    l4_basis = [[i] for i in sort_order(l4_basis)]

    print("l4_basis", "(", len(l4_bf), "):", [[l4_bf[i], l4_basis_facets[i]] for i in range(len(l4_bf))])


    pre_se5 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
               list(itertools.combinations(range(1, d + 1), 5))]
    se5 = [i for i in pre_se5 if list(set(i) & set(infty)) == []]
    se5_facets = [stimultaneous_spl(_) for _ in se5]
    se5cols = [select_columns( _) for _ in se5]
    print("se5", "(", len(se5), "):", se5_facets)

    pre_se6 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
               list(itertools.combinations(range(1, d + 1), 6))]
    se6 = [i for i in pre_se6 if list(set(i) & set(infty)) == []]
    se6_facets = [stimultaneous_spl(_) for _ in se6]
    se6cols = [select_columns( _) for _ in se6]
    print("se6", "(", len(se6), "):", se6_facets)

    pre_se7 = [[10 * i + j for i, j in list(itertools.combinations(k, 2))] for k in
               list(itertools.combinations(range(1, d + 1), 7))]
    se7 = [i for i in pre_se7 if list(set(i) & set(infty)) == []]
    se7_facets = [stimultaneous_spl(_) for _ in se7]
    se7cols = [select_columns( _) for _ in se7]
    print("se7", "(", len(se7), "):", se7_facets)

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

    range_s3 = range(len(s3))
    range_s4 = range(len(s4))
    range_se5 = range(len(se5))
    range_se6 = range(len(se6))
    range_se7 = range(len(se7))
    range_e3 = range(len(e3))
    range_e4 = range(len(e4))
    range_l4 = range(len(l4))
    range_l4_basis = range(len(l4_basis))
    range_i4 = range(len(i4))

    cancel_list = []
    for i in range(len(Vert)):
        temp = []
        temps3 = []
        temps4 = []
        tempse5 = []
        tempse6 = []
        tempse7 = []
        tempe3 = []
        tempe4 = []
        templ4 = []
        templ4_basis = []
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

        for j in range_se5:
            temp1 = se5[j]
            judge_temp = [k for k in temp1 if k in cover_list[i]]
            if len(temp1) == len(judge_temp):
                tempse5 += [j]
                range_se5 = [_ for _ in range_se5 if _ not in tempse5]

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

        for j in range_l4:
            temp1 = l4[j]
            judge_temp = [k for k in temp1 if k in cover_list[i]]
            if len(temp1) == len(judge_temp):
                templ4 += [j]
                range_l4 = [_ for _ in range_l4 if _ not in templ4]

        for j in range_l4_basis:
            temp1 = l4_basis[j]
            judge_temp = [k for k in temp1 if k in cover_list[i]]
            if len(temp1) == len(judge_temp):
                templ4_basis += [j]
                range_l4_basis = [_ for _ in range_l4_basis if _ not in templ4_basis]

        for j in range_i4:
            temp1 = i4[j]
            judge_temp = [k for k in temp1 if k in cover_list[i]]
            if len(temp1) == len(judge_temp):
                tempi4 += [j]
                range_i4 = [_ for _ in range_i4 if _ not in tempi4]

        temp += [
            [i, temps3, temps4, tempse5, tempse6, tempse7, tempe3, tempe4, tempse5, tempse6, tempse7, templ4, templ4_basis, tempi4]]
        cancel_list += temp

    print("cancel_list:", cancel_list)

    print()

    condition_all = (s3, s4, se5, se6, se7, e3, e4, se5, se6, se7, l4, l4_basis, i4)  ## cols
    library_all = (ts3, ts4, ts5, ts6, ts7, te3, te4, te5, te6, te7, tl4, tl4_basis, ti4)  ## admitting angles
    kind_all = ("s3", "s4", "s5", "s6", "s7", "e3", "e4", "e5", "e6", "e7", "l4", "l4_basis", "i4")  ## type

    condition = [condition_all[x - 1] for x in [1, 6, 7, 13, 12, 11]]
    library = [library_all[x - 1] for x in [1, 6, 7, 13, 12, 11]]
    kind = [kind_all[x - 1] for x in [1, 6, 7, 13, 12, 11]]

    print("pasting + round 1 killing:", "*" * 100)

    select_cancel_list = [[i[j] for j in [0, 1, 6, 7, 13, 12, 11]] for i in cancel_list]
    print("select_cancel_list:", select_cancel_list)

    tstar = time.time()

    T = []
    t1 = time.time()
    sel_cols = select_columns(vert[0])
    df_1 = np.zeros([Nl, Nr])
    df_1[:, [j - 1 for j in sel_cols]] = pd.DataFrame(ts4, dtype=np.int8, columns=list(range(1, dh + 1)))  # 【6个二面角】
    data = pd.DataFrame(df_1, dtype=np.int8, columns=[str(_) for _ in list(range(1, Nr + 1))])
    t2 = time.time()
    Ttemp = ["df1:", data.shape, t2 - t1]
    T += [Ttemp]
    print(Ttemp)

    for k in range(len(select_cancel_list[0]) - 2, len(select_cancel_list[0])):
        if select_cancel_list[0][k] != []:
            cdt = condition[k - 1]
            lib = library[k - 1]
            for j in select_cancel_list[0][k]:
                t1 = time.time()
                saving = functools.partial(saver, j)
                data[str(Nr + 1)] = data.apply(saving, axis=1)
                # print(data)
                data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
                t2 = time.time()
                Ttemp = ["df1", kind[k - 1], select_cancel_list[0][k], data.shape[0], t2 - t1]
                T += [Ttemp]
                print(Ttemp)

    for k in range(1, len(select_cancel_list[0]) - 2):
        if select_cancel_list[0][k] != []:
            cdt = condition[k - 1]
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
    sel_cols = select_columns(vert[1])
    df_2 = np.zeros([Nl, Nr])
    df_2[:, [j - 1 for j in sel_cols]] = pd.DataFrame(ts4, dtype=np.int8, columns=list(range(1, dh + 1)))
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

    for k in range(len(select_cancel_list[0]) - 2, len(select_cancel_list[0])):
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
                Ttemp = ["df2", [kind[k - 1], select_cancel_list[1][k], data.shape[0], t2 - t1]]
                T += [Ttemp]
                print(Ttemp)

    for k in range(1, len(select_cancel_list[0]) - 2):
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

    for i in range(2, Nv):
        t1 = time.time()
        sel_cols = select_columns(vert[i])
        temp = "df_" + str(i + 1)
        exec(temp + "=np.zeros([Nl,Nr])")
        exec(temp + "[:,[ j-1 for j in sel_cols]]=pd.DataFrame(ts4,dtype=np.int8,columns=list(range(1,dh+1)))")
        exec(temp + "=pd.DataFrame(" + temp + ",dtype=np.int8,columns=[str(_) for _ in list(range(1,Nr+1))])")
        tem = sort_order(select_columns(link[i - 1]))
        tem1 = list(set(range(1, Nr + 1)) - set(tem))
        exec("data_temp=pd.merge(" + temp + ",data,on=[str(_) for _ in tem])"); data=locals()['data_temp']
        for j in tem1:
            data[str(j)] = data[str(j) + "_x"] + data[str(j) + "_y"]
        data = data[[str(_) for _ in range(1, Nr + 1)]]
        t2 = time.time()
        Ttemp = ["df", i + 1, data.shape, t2 - t1]
        T += [Ttemp]
        print(Ttemp)

        if data.shape[0] == 0:
            break

        for k in range(len(select_cancel_list[0]) - 2, len(select_cancel_list[0])):
            if select_cancel_list[i][k] != []:
                cdt = condition[k - 1]
                lib = library[k - 1]
                for j in select_cancel_list[i][k]:
                    t1 = time.time()
                    saving = functools.partial(saver, j)
                    data[str(Nr + 1)] = data.apply(saving, axis=1)
                    # print(data)
                    data = data.loc[data[str(Nr + 1)] == True][[str(_) for _ in range(1, Nr + 1)]]
                    t2 = time.time()

                    Ttemp = ["df", i + 1, [kind[k - 1], select_cancel_list[i][k], data.shape[0], t2 - t1]]
                    T += [Ttemp]
                    print(Ttemp)

        for k in range(1, len(select_cancel_list[0]) - 2):
            if select_cancel_list[i][k] != []:
                cdt = condition[k - 1]
                lib = library[k - 1]
                for j in select_cancel_list[i][k]:
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

    # after per
    per = pd.read_csv("P8_17_per.txt", sep=' ', names=range(1, d + 1))
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
    # data_after_per = [conver_preresult(_) for _ in range(len(pre_result))]
    tend = time.time()
    print("permu time consuming2:", tend - tstart)

    data = pd.DataFrame(data_after_per, dtype=int,
                        columns=[str(_) for _ in range(1, Nr + 1)])  # columns name 不用str,否则replace函数的切片指标会差1
    data.to_csv('output/P8_17/P8_17_LSIEr1_per.txt', header=None, index=None, sep=' ', mode='a')

    # round 2
    print()
    print("pasting+round 2 killing:", "*" * 100)
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
    print("data round 2:", data.shape)

    data.to_csv('output/P8_17/P8_17_LSIEr2_per.txt', header=None, index=None, sep=' ', mode='a')

    # chenage 7

    tstart = time.time()
    data.columns = range(1, Nr + 1)
    temp = [replace7(_) for _ in range(len(data))]
    tend = time.time()
    print("change7 time consuming:", tend - tstart)

    data.to_csv('output/P8_17/P8_17_LSIE_per_change7.txt', header=None, index=None, sep=' ', mode='a')
    return


if __name__ == '__main__':
    Vert = [[1, 2, 4, 6], [1, 3, 4, 6], [2, 3, 4, 6], [1, 2, 5, 6], [1, 3, 5, 6], [2, 3, 5, 6],
            [1, 2, 5, 7], [1, 3, 5, 7], [1, 2, 4, 7], [1, 3, 4, 7], [2, 3, 5, 8], [2, 5, 7, 8],
            [3, 5, 7, 8], [2, 3, 4, 8], [2, 4, 7, 8], [3, 4, 7, 8]]
    run48(Vert)