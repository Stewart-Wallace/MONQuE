import numpy as np
from os import getcwd,listdir
from itertools import permutations
import sys
import random
import json

def uniform_search(dim,verbose = False):
    directory = getcwd()
    directory_content = listdir(directory)

    for x in directory_content:
        i = ''.join(ch for ch in x.split(r"\\")[-1] if not ch.isdigit())

        if i == "opertators__complexity_.npy" and x[:-4]+'_unitary_set.npy' not in directory_content:
            if verbose:
                print("Constructing Uniform Transformation: \t" + x)


            operators = np.load(x, allow_pickle=True)
            op_mats = np.array(list(operators[1]), dtype=np.complex128)
            op_str = list(operators[0])

            uni_matrices = []
            for i in permutations(list(range(2**dim))):
                start = np.zeros((2**dim,2**dim))
                for n,y in enumerate(i):
                    start[n,y] = 1
                if np.allclose(np.dot(start,np.conj(start).T),np.eye(2**dim)):
                    uni_matrices.append(start.copy())

            complete_unitary_matrices = uni_matrices.copy()
            indx_of_incomplete_mtx = []
            for n,u in enumerate(uni_matrices):
                for ops in op_mats:
                    if np.dot(u,np.dot(ops,np.conj(u).T)) in op_mats:
                        pass
                    else:
                        indx_of_incomplete_mtx.append(n)
            indx_of_incomplete_mtx.reverse()
            for i in indx_of_incomplete_mtx:
                complete_unitary_matrices.pop(i)

            trans_dict = []
            for n,u in enumerate(uni_matrices):
                if np.allclose(np.eye(2**dim),u):
                    pass
                else:
                    trans_dict.append({})
                    failed = False
                    for m,s in zip(op_mats,op_str):
                        U = [i for i,m_0 in enumerate(op_mats) if np.allclose(np.dot(u,np.dot(m,np.conj(u).T)),m_0)]
                        if U == []:
                            if verbose:
                                print('Lindblad missing for rotation with a given uniatary matrix.')
                            failed = True
                        else:
                            trans_dict[-1][s] = op_str[U[0]]
                            
                    if failed:
                        trans_dict.pop()
                if verbose and n%3 == 0 and n != 0:
                    print("Transforms Constructed: \t\t" + str(100*n/len(uni_matrices)) + r"%")

            with open(x[:-4]+'_unitary_set.npy','w')as f:
                if verbose:
                    print(str(len(trans_dict)) + ' Uniform Transoformations found for the given Library of terms')
                json.dump(trans_dict,f)
        elif i == "opertators__complexity_.npy":
            if verbose:
                print("Unitary Transformations Found:\t\t" +x[:-4]+'_unitary_set.npy')

#uniform_search(verbose=True)