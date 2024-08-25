# Multinode simulation but all contained in 1 function

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def id(node):
    return node[0]

def edge(matrix, fromnode, tonode): # Edge which flows i to j
    i = id(fromnode)
    i = int(i-1)
    j = id(tonode)
    j = int(j-1)
    return matrix[i][j]

#s, e, i and r
def N(node):
    return sum(node[1::])

def S(node):
    return node[1]

def E(node):
    return node[2]

def I(node):
    return node[3]

def R(node):
    return node[4]

def delta_seir(node, params_dict):
    
    beta = params_dict["beta"]
    c = params_dict["c"]
    sigma = params_dict["sigma"]
    gamma = params_dict["gamma"]

    dS = - ((beta * c) * S(node) * I(node)) / N(node)
    dE = ((beta * c) * S(node) * I(node)) / N(node) - sigma * E(node)
    dI = sigma * E(node) - gamma * I(node)
    dR = gamma * I(node)

    return np.array([0, dS, dE, dI, dR])

def delta_out(fromnode, tonode, flow_matrix, univ_matrix, partial_matrix):

    F = edge(flow_matrix, fromnode, tonode)
    U = edge(univ_matrix, fromnode, tonode)
    P = edge(partial_matrix, fromnode, tonode)

    FS = (U * S(fromnode) * F)/N(fromnode)
    FE = (U * E(fromnode) * F)/N(fromnode)
    FI = (U * P * I(fromnode) * F)/N(fromnode)
    FR = (U * R(fromnode) * F)/N(fromnode)

    return np.array([0, FS, FE, FI, FR])

def infected_enroute(fromnode, tonode, flow_matrix, freq_matrix, univ_matrix, partial_matrix, params_dict):
    
    A = params_dict["A"]
    beta = params_dict["beta"]
    r = params_dict["r"]

    F = edge(flow_matrix, fromnode, tonode)
    f = edge(freq_matrix, fromnode, tonode)
    U = edge(univ_matrix, fromnode, tonode)
    P = edge(partial_matrix, fromnode, tonode)

    s = (U * F * S(fromnode))/N(fromnode)
    i = (U * P * F * I(fromnode))/N(fromnode)

    L = s * (1-np.exp((-i * np.pi * np.square(r))/A)) * beta
    # later replace beta with 1-exp(-kT)
    
    delta_array = delta_out(fromnode, tonode, flow_matrix, univ_matrix, partial_matrix)
    
    delta_array[1] = delta_array[1] - L
    delta_array[2] = delta_array[2] + L

    return delta_array

def delta_travel(node, nodes, flow_matrix, freq_matrix, univ_matrix, partial_matrix, params_dict):
    intake = [0, 0, 0, 0, 0]
    outtake = [0, 0, 0, 0, 0]
    for n in [x for x in nodes if not (x==node).all()]:
        intake += infected_enroute(n, node, flow_matrix, freq_matrix, univ_matrix, partial_matrix, params_dict)
        outtake += delta_out(node, n, flow_matrix, univ_matrix, partial_matrix)
    return intake - outtake

def update_node_status(node, nodes, flow_matrix, freq_matrix, univ_matrix, partial_matrix, params_dict):
    node += delta_seir(node, params_dict)
    node += delta_travel(node, nodes, flow_matrix, freq_matrix, univ_matrix, partial_matrix, params_dict)
    return node


def sim(nodes, time_horizon, flow_matrix, freq_matrix, univ_matrix, partial_matrix, params_dict):
    
    nodes_num = len(nodes)
    rowslist = []

    # Initial Values
    for i in range(0, nodes_num):
        initrow = [{"t": 0, "S": S(nodes[i]), "E": E(nodes[i]), "I": I(nodes[i]), "R":R(nodes[i])}]
        rowslist.append(initrow)


    for t in range(1, time_horizon):
        for j in range(0, nodes_num):
            nodes[j] = update_node_status(nodes[j], nodes, flow_matrix, freq_matrix, univ_matrix, partial_matrix, params_dict)
            insertrow = {"t": t, "S": S(nodes[j]), "E": E(nodes[j]), "I": I(nodes[j]), "R":R(nodes[j])}
            rowslist[j].append(insertrow)
        
    dflist = []

    for k in range(0, nodes_num):
        df = pd.DataFrame(rowslist[k])
        df["N"] = df["S"] + df["E"] + df["I"] + df["R"]
        df["C"] = df["I"] + df["R"]
        dflist.append(df)

    return dflist

def ESI(df_list, t_h):
    cumsum = 0
    sumpop = 0
    for df in df_list:
        cum = df.query("t == @t_h")["C"].item()
        cumsum += cum
        pop = df.query("t == @t_h")["N"].item()
        sumpop += pop

    return cumsum/sumpop

def disruption_constF(df_list, t_h, F_matrix, P_matrix, U_matrix): # constant flow rate
    total_flow = 0
    disrupted = 0

    lst = list(range(1, len(df_list) + 1))

    for i in lst:
        fromid = i-1
        df = df_list[fromid]

        exclu_lst = lst[:]
        exclu_lst.remove(i)


        for j in exclu_lst:
            toid = j-1
            F = F_matrix[fromid][toid]
            P = P_matrix[fromid][toid]
            U = U_matrix[fromid][toid]

            total_flow += F * t_h

            df["i"] = df.apply(lambda x: np.divide(x.I, x.N), axis=1)
            df["Irestrict"] = df["i"].apply(lambda x: F*(1-P*U)*x)
            df["otherrestrict"] = df["i"].apply(lambda x: F*(1-U)*(1-x))
            disrupted += df["Irestrict"].sum() + df["otherrestrict"].sum()

        df = None
    
    return disrupted/total_flow

def plot_SEIR(df, ax, title):
    ax.set_title(title)
    ax.plot(df["t"], df["S"], color="blue")
    ax.plot(df["t"], df["E"], color="orange")
    ax.plot(df["t"], df["I"], color="red")
    ax.plot(df["t"], df["R"], color="green")
    ax.plot(df["t"], df["C"], color="red", linestyle="dashed")