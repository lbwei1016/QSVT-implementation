#from qiskit import QuantumCircuit
#from qiskit.circuit import QuantumRegister, ClassicalRegister
#from qiskit.extensions import UnitaryGate
from qiskit.quantum_info import Statevector

import numpy as np

# from qsvt.algorithms import linear_solver
from qsvt.algorithms_beta import linear_solver
from qsvt.helper import total_variation, gen_random_matrix

from qiskit import transpile
from qiskit_aer import AerSimulator
#from qiskit.visualization import plot_histogram

import time
import getopt, sys

# import memory_profiler

# @memory_profiler.profile
def exec():

    TOTAL_TIME = 0
    AA_On = False
    set_degree = 0
    # matrix size exponent
    N = 1


    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "hN:ad:"

    # Long options
    long_options = ["help", "Num_of_qubits_for_matrix=", "AA", "set_degree"]

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-h", "--help"):
                help_msg = """
    -h: show help
    -N <Number of qubits for matrix>: specify matrix size
    -a: Use AA
                """
                print(help_msg)
                
            elif currentArgument in ("-N", "--Num_of_qubits_for_matrix"):
                N = int(currentValue)
                print(f'N = {currentValue}')
                
            elif currentArgument in ("-a", "--AA"):
                AA_On = True
                print('AA is on')
            elif currentArgument in ("-d", "--set_degree"):
                set_degree = int(currentValue)
                print(f"set_degree = {set_degree}")
                
    except getopt.error as err:
        # output error, and return with an error code
        print (str(err))

    #########################################################################

    #N = 9
    st = time.time()
    A = gen_random_matrix(50-1e-9, 2**(N), N)
    ed = time.time()
    print(f'generate matrix time spent: {ed - st} sec')
    print('==================================')


    # A = np.array([
    #     [1, -1/3],
    #     [-1/3, 1]
    # ])

    A_norm = np.linalg.norm(A)
    A /= A_norm
    print(f'A:\n{A}')

    print(f'calculating condition number...')
    st = time.time()
    kappa = np.linalg.cond(A)
    ed = time.time()
    print(f'time spent for calculating condition number: {ed - st} sec')
    print(f'kappa: {kappa}')

    W, S, Vd = np.linalg.svd(A)
    print(f'SVD of normalized A:\n\tW:\n{W}\n\tS:\n{S}\n\tVd:\n{Vd}')

    st = time.time()
    if not AA_On:
        qc = linear_solver(A, set_degree=set_degree)
    else:
        qc = linear_solver(A, set_degree=set_degree, amplify='AA')
    # qc = linear_solver(A, eps=0.01, set_kappa=True)
    # qc = linear_solver(A, set_kappa=True)
    # qc = linear_solver(A, set_kappa=True, amplify='sign')
    # qc = linear_solver(A, amplify='sign')
    # qc = linear_solver(A, real_only=False)
    # qc = linear_solver(A, amplify='chebyshev')
    # qc = linear_solver(A, amplify='sign')
    # qc = linear_solver(A, amplify='AA')
    ed = time.time()
    TOTAL_TIME += (ed - st)
    print(f'prepare circuit spends: {ed - st} sec')

    # print(f'circuit depth: {qc.depth()}')
    # qc.draw('mpl')

    print('==================================')

    st = time.time()
    state = Statevector(qc)
    ed = time.time()
    print(f'prepare state snapshot spends: {ed - st} sec')

    n = qc.num_qubits
    print(f'number of qubits: {n}')

    # for AA or not
    if AA_On:
        measure_qubits = [n - 3, n - 2]
    else:
        measure_qubits = [n - 2, n - 1]

    exp_outcome = "00"

    # for no AA and no real_only
    # measure_qubits = [n - 1]
    # exp_outcome = "0"

    st = time.time()
    while True:
        outcome, mstate = state.measure(measure_qubits)
        if outcome == exp_outcome: break
    ed = time.time()
    print(f'post-measurement state: {mstate}')
    print(f'post-selection spends: {ed - st} sec')

    # for AA: 3 ancilla qubits
    st = time.time()
    if AA_On:
        res = np.linalg.solve(A, np.array([1] + [0] * (2 ** (n - 3) - 1)))
    else:
        # for no AA: 2 ancilla qubits
        res = np.linalg.solve(A, np.array([1] + [0] * (2 ** (n - 2) - 1)))

    # for no AA and no real_only: 1 ancilla qubits
    # res = np.linalg.solve(A, np.array([1] + [0] * (2 ** (n - 1) - 1)))
    ed = time.time()
    res /= np.linalg.norm(res)
    print(f'res: {res}')
    print(f'Classically solving Ax=b time spent: {ed - st} sec')


    # Calculate total variation
    print('==================================')
    P = np.array([mstate[i] for i in range(2 ** N)])
    P = np.array([np.linalg.norm(x)**2 for x in P])
    #print(f'P: {P}')
    # res = [-0.63012604,  0.070014,    0.070014,    0.77015405]
    Q = np.array([x ** 2 for x in res])
    #print(f'Q: {Q}')

    print(f'kappa: {kappa}')
    print(f'total_variation (exact): {total_variation(P, Q)}')


    # cr = ClassicalRegister(len(measure_qubits))
    # qc.add_register(cr)
    # qc.measure(measure_qubits, cr)
    print('==================================')
    qc.measure_all()
    print(f'qc depth: {qc.depth()}')

    # # It seems that even if 'GPU' is specified, GPU is not used at all.
    # # Since QSVT involves large multi-qubit gates (block-encoding), "extended_stabilizer" is not efficient.
    # # sim = AerSimulator(method='extended_stabilizer')
    # # sim = AerSimulator(method='statevector')
    # sim = AerSimulator(method='density_matrix')
    # # sim = AerSimulator(method='statevector', device='GPU')

    # st = time.time()
    # transpiled_circuit = transpile(qc, sim)
    # # transpiled_circuit = transpile(qc, sim, optimization_level=3)
    # ed = time.time()
    # print(f'transpilation spends: {ed - st} sec')
    # TOTAL_TIME += (ed - st)
    # # transpiled_circuit = transpile(qc, sim, optimization_level=3)
    # print(f'transpiled qc depth: {transpiled_circuit.depth()}')


    # # run job
    # shots = 10000
    # st = time.time()
    # job = sim.run(transpiled_circuit, shots=shots, dynamic=True, blocking_enable=True, blocking_qubits=10)
    # ed = time.time()

    # # print(f'run job spends: {ed - st} sec')
    # TOTAL_TIME += (ed - st)
    # # Get the results and display them
    # exp_result = job.result()
    # exp_counts = exp_result.get_counts()
    
    
    # plot_histogram(exp_counts)

    # Calculate total variance
    # experiment count
    #print(f'exp_counts: {exp_counts}')

    # st = time.time()
    # valid_count = np.zeros(shape=(2 ** N))
    # for data in exp_counts:
    #     # print(f'data: {data[:]}')
    #     if data[:2] == '00':
    #         # print(int(data[2:], base=2))
    #         valid_count[int(data[2:], base=2)] = exp_counts[data]
    # valid_count /= shots
    # valid_count /= np.linalg.norm(valid_count)
    # #print(f'valid_count: {valid_count}')

    # Q = np.array([x ** 2 for x in res])
    # tot_var = total_variation(valid_count, Q)
    # ed = time.time()
    # print(f'total var. (exact) time spent: {ed - st}')
    # #print(f'Q: {Q}')

    # print(f'kappa: {kappa}')
    # print(f'total_variation (exp): {tot_var}')
    # print('==================================')

    print(f'total execution time (exclude snapshot): {TOTAL_TIME} sec')


exec()