# Import your files here...
from collections import defaultdict
import math
import re

# Question 1

def read_files(State_File, Symbol_File, Query_File):
    with open(State_File) as f:
        states = defaultdict(int)
        N = int(f.readline())
        for i in range(N):
            states[i] = f.readline().strip()
        state_transit = defaultdict(lambda: defaultdict(int))
        line = f.readline()
        while line != '':
            toks = line.strip().split()
            state_transit[int(toks[0])][int(toks[1])] = int(toks[2])
            line = f.readline()
    for key_s in states:
        states[key_s] = 0
        for key_tr in state_transit[key_s]:
            states[key_s] += state_transit[key_s][key_tr]
    with open(Symbol_File) as f:
        symbols = defaultdict(int)
        M = int(f.readline())
        for i in range(M):
            symbols[f.readline().strip()] = i
        symbol_emission = defaultdict(lambda: defaultdict(int))
        line = f.readline()
        while line != '':
            toks = line.strip().split()
            symbol_emission[int(toks[0])][int(toks[1])] = int(toks[2])
            line = f.readline()      
    symbol_prob = defaultdict(int)
    for state_id in symbol_emission:
        symbol_prob[state_id] = 0
        for key_symbol in symbol_emission[state_id]:
            symbol_prob[state_id] += symbol_emission[state_id][key_symbol]
    with open(Query_File) as f:
        queries = []
        for line in f.readlines():
            queries += [re.split(r"([,\-\(\)\/\&\s*])", line.strip())]
            while ' ' in queries[-1]:
                queries[-1].remove(' ')
            while '' in queries[-1]:
                queries[-1].remove('')
            
    return states, state_transit, symbols, symbol_emission, symbol_prob, queries, N, M

def caculate_prob(Aij, Ai, Bij, Bi, N, M):
    return (Aij + 1) / (Ai + N-1) * (Bij + 1) / (Bi + M + 1)

# Q1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    states, state_transit, symbol, symbol_emission, symbol_prob, queries, N, M = read_files(State_File, Symbol_File, Query_File)
    result = []
    for line in queries:
        L = []
        viterbi_prob = defaultdict(lambda: defaultdict(int))
        viterbi_state = defaultdict(lambda: defaultdict(int))
        for i in range(len(line)):
            o = line[i]
            for s in range(N-2):
                pre_state = -1
                if o not in symbol:
                    symbol[o] = -1
                if i == 0:
                    viterbi_prob[i][s] = caculate_prob(state_transit[N-2][s], states[N-2], symbol_emission[s][symbol[o]], symbol_prob[s], N, M)
                else:
                    for pre_s in range(N-2):
                        p = caculate_prob(state_transit[pre_s][s], states[pre_s], symbol_emission[s][symbol[o]], symbol_prob[s], N, M) * \
                             viterbi_prob[i-1][pre_s]
                        if viterbi_prob[i][s] < p:
                            viterbi_prob[i][s] = p
                            viterbi_state[i][s] = pre_s
        max_p = 0
        max_s = -1
        for s in range(N-2):
            viterbi_prob[i][s] *= (state_transit[s][N-1] + 1) / (states[s] + N-1)
            if max_p < viterbi_prob[i][s]:
                max_p = viterbi_prob[i][s]
                max_s = s
        L = [max_s]
        for i in range(len(viterbi_state), -1, -1):
            L.append(viterbi_state[i][L[-1]])
        result += [[N-2] + list(reversed(L))[1:] + [N-1] + [math.log(max_p)]]
        
    return result

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k):
    states, state_transit, symbol, symbol_emission, symbol_prob, queries, N, M = read_files(State_File, Symbol_File, Query_File)
    result = []
    for line in queries:
        L = []
        viterbi_prob = defaultdict(lambda :defaultdict(defaultdict))
        for i in range(len(line)):
            o = line[i]
            for s in range(N-2):
                pre_state = -1
                if o not in symbol:
                    symbol[o] = -1
                if i == 0:
                    viterbi_prob[i][s][0] = (caculate_prob(state_transit[N-2][s], states[N-2], symbol_emission[s][symbol[o]], symbol_prob[s], N, M),[N-2,s])
                else:
                    top_k_p = defaultdict(lambda: defaultdict(tuple))
                    top_k_states = defaultdict(lambda: defaultdict(tuple))
                    for pre_s in range(N-2):
                        for inner_k in range(min(len(viterbi_prob[i-1][pre_s]),k)):
                            top_k_p[pre_s,inner_k] = caculate_prob(state_transit[pre_s][s], states[pre_s], symbol_emission[s][symbol[o]], symbol_prob[s], N, M) * \
                                     viterbi_prob[i-1][pre_s][inner_k][0]
                            top_k_states[pre_s,inner_k] = viterbi_prob[i-1][pre_s][inner_k][1] + [s]
                            if i == len(line) - 1:
                                top_k_p[pre_s,inner_k] *= (state_transit[s][N-1] + 1) / (states[s] + N-1)
                                top_k_states[pre_s,inner_k] += [N-1]
                    p = sorted(top_k_p.items(), key=lambda p: p[1],reverse=True)
                    for inner_k in range(k):
                        if inner_k < len(p):
                            viterbi_prob[i][s][inner_k] = (p[inner_k][1],top_k_states[p[inner_k][0][0],p[inner_k][0][1]])
        # sort the last column to find top-k
        top_k_list = []
        for i in range(N-2):
            for j in range(len(viterbi_prob[len(line)-1][i])):
                top_k_list.append(viterbi_prob[len(line)-1][i][j])
        top_k_list = sorted(top_k_list, key=lambda L:L[0], reverse=True)
        for num in range(k):
            result += [top_k_list[num][1] + [math.log(top_k_list[num][0])]]
    return result


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    states, state_transit, symbol, symbol_emission, symbol_prob, queries, N, M = read_files(State_File, Symbol_File, Query_File)
    result = []
    symbol_p = defaultdict(int)
    symbol_v = defaultdict(int)
    for s in symbol_emission:
        symbol_v[s] = len(symbol_emission[s])
        symbol_p[s] = 1 / (symbol_prob[s] + symbol_v[s])
        
    
    for line in queries:
        L = []
        viterbi_prob = defaultdict(lambda: defaultdict(int))
        viterbi_state = defaultdict(lambda: defaultdict(int))
        for i in range(len(line)):
            o = line[i]
            for s in range(N-2):
                pre_state = -1
                if o not in symbol:
                    symbol[o] = -1
                if i == 0:
                    if symbol_emission[s][symbol[o]] > 0:
                        viterbi_prob[i][s] = (symbol_emission[s][symbol[o]] / symbol_prob[s] - symbol_p[s]) * (state_transit[N-2][s] + 1) / (N - 1 + states[N-2])
                    else:
                        viterbi_prob[i][s] = symbol_v[s] * symbol_p[s] / (M + 1 - symbol_v[s]) * (state_transit[N-2][s] + 1) / (N - 1 + states[N-2])
                else:
                    for pre_s in range(N-2):
                        if symbol_emission[s][symbol[o]] > 0:
                            p = (symbol_emission[s][symbol[o]] / symbol_prob[s] - symbol_p[s]) * (state_transit[pre_s][s] + 1) / (N - 1 + states[pre_s]) * \
                             viterbi_prob[i-1][pre_s]
                        else:
                            p = symbol_v[s] * symbol_p[s] / (M + 1 - symbol_v[s]) * (state_transit[pre_s][s] + 1) / (N - 1 + states[pre_s]) * \
                             viterbi_prob[i-1][pre_s]
                        if viterbi_prob[i][s] < p:
                            viterbi_prob[i][s] = p
                            viterbi_state[i][s] = pre_s
        max_p = 0
        max_s = -1
        for s in range(N-2):
            viterbi_prob[i][s] *= (state_transit[s][N-1] + 1) / (states[s] + N-1)
            if max_p < viterbi_prob[i][s]:
                max_p = viterbi_prob[i][s]
                max_s = s
        L = [max_s]
        for i in range(len(viterbi_state), -1, -1):
            L.append(viterbi_state[i][L[-1]])
        result += [[N-2] + list(reversed(L))[1:] + [N-1] + [math.log(max_p)]]
        
    return result
