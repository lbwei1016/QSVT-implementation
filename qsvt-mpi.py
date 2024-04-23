from qiskit.quantum_info import Statevector

import numpy as np

from qsvt.algorithms import linear_solver
from qsvt.helper import total_variation, random_matrix

from qiskit import transpile, QuantumCircuit, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import time
import getopt, sys

# import memory_profiler
# @memory_profiler.profile


OVERALL_TIME = 0
AA_On = False
set_degree = 0
# simulation_method = 'statevector'
MPI_ON = False
SIMULATION = False
# matrix size exponent
N = 1

def parse_cmd_parameters():
    # Remove 1st argument from the
    # list of command line arguments
    argumentList = sys.argv[1:]

    # Options
    options = "hN:ad:ms"

    # Long options
    long_options = ["help", "num-of-qubits-for-matrix=", "AA", "set-degree", 'mpi', 'self-sampling']

    global OVERALL_TIME
    global AA_On
    global set_degree
    global MPI_ON
    global SIMULATION 
    global N

    try:
        # Parsing argument
        arguments, values = getopt.getopt(argumentList, options, long_options)
        
        # checking each argument
        for currentArgument, currentValue in arguments:

            if currentArgument in ("-h", "--help"):
                help_msg = """
    -h: Show help
    -N <Number of qubits for matrix>: specify matrix size
    -a: Use AA
    -m: enable MPI
    -s: Run simulation (with AerSimulator), rather than self-sample the statevetor
                """
                print(help_msg)
                exit(0)
                
            elif currentArgument in ("-N", "--num-of-qubits-for-matrix"):
                N = int(currentValue)
                print(f'N = {currentValue}')
                
            elif currentArgument in ("-a", "--AA"):
                AA_On = True
                print('AA is on')
            elif currentArgument in ("-d", "--set-degree"):
                set_degree = int(currentValue)
                print(f"set_degree = {set_degree}")
            elif currentArgument in ('-m', '--mpi'):
                MPI_ON = True
                print(f'MPI is on with {MPI_ON} nodes')
            elif currentArgument in ('-s', 'self-sampling'):
                SIMULATION = True
                print('Experiment with AerSimulator')
                
    except getopt.error as err:
        # output error, and return with an error code
        print(str(err))


def gen_random_unitary() -> np.ndarray:
    st = time.time()
    A = random_matrix(N)
    ed = time.time()
    print(f'generate matrix time spent: {ed - st} sec')
    print('==================================')

    A_norm = np.linalg.norm(A)
    A /= A_norm
    print(f'A:\n{A}')

    print(f'calculating condition number...')
    kappa = np.linalg.cond(A)

    return A


def create_circuit(A: np.ndarray) -> QuantumCircuit:
    global OVERALL_TIME

    st = time.time()
    if not AA_On:
        qc = linear_solver(A, set_degree=set_degree)
    else:
        qc = linear_solver(A, set_degree=set_degree, amplify='AA')
    ed = time.time()

    OVERALL_TIME += (ed - st)

    print(f'prepare circuit spends: {ed - st} sec')
    print(f'circuit depth: {qc.depth()}')
    # qc.draw('mpl')

    print('==================================')

    return qc


def prepare_snapshot(A: np.ndarray, qc: QuantumCircuit):
    global OVERALL_TIME

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
        res = np.linalg.solve(A, np.array([1] + [0] * (2 ** N - 1)))
    else:
        # for no AA: 2 ancilla qubits
        res = np.linalg.solve(A, np.array([1] + [0] * (2 ** N - 1)))

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
    Q = np.array([x ** 2 for x in res])

    # print(f'kappa: {kappa}')
    print(f'total_variation (exact): {total_variation(P, Q)}')
    print('==================================')

    return Q


def self_sampling(qc: QuantumCircuit, real_distribution: np.ndarray, shots: int=10000):
    # global OVERALL_TIME
    
    st = time.time()
    state = Statevector(qc)
    P = np.array([np.linalg.norm(x)**2 for x in state])
    
    n = qc.num_qubits
    a = np.random.choice(a=list(range(2 ** n)), p=P, size=shots)
    a = np.sort(a)
    unique_elements, counts = np.unique(a, return_counts=True)
    counts = dict(zip(unique_elements, counts))

    ed = time.time()
    print(f'sampling time: {ed - st} sec')
    

    st = time.time()
    SIZE = 2 ** N
    valid_count = np.zeros(shape=(SIZE))
    tmp = 0
    for data in counts:
        if data < SIZE:
            tmp += counts[data]
            valid_count[int(data)] = counts[data]
    valid_count /= shots
    valid_count /= np.sum(valid_count)
    # print(f'tmp: {tmp}')
    print(f'valid_count: {valid_count}; sum: {np.sum(valid_count)}')

    tot_var = total_variation(valid_count, real_distribution)
    ed = time.time()
    print(f'total var. (exp) time spent: {ed - st}')

    # print(f'kappa: {kappa}')
    print(f'total_variation (exp): {tot_var}')

    
def simulation(qc: QuantumCircuit, shots: int=10000) -> list:
    global OVERALL_TIME

    qc.measure_all()
    print(f'qc depth: {qc.depth()}')

    # It seems that even if 'GPU' is specified, GPU is not used at all.
    # Since QSVT involves large multi-qubit gates (block-encoding), "extended_stabilizer" is not efficient.
    # sim = AerSimulator(method='extended_stabilizer')
    sim = AerSimulator(method='statevector')
    # sim = AerSimulator(method='density_matrix')
    # sim = AerSimulator(method='statevector', device='GPU')

    st = time.time()
    # transpiled_circuit = transpile(qc, sim)
    transpiled_circuit = transpile(qc, sim, optimization_level=3)
    ed = time.time()
    print(f'transpilation spends: {ed - st} sec')
    OVERALL_TIME += (ed - st)
    print(f'transpiled qc depth: {transpiled_circuit.depth()}')

    # run job
    st = time.time()

    if MPI_ON:
        # 'blocking_qubits=10' is set ignorantly here
        # job = sim.run(transpiled_circuit, shots=shots, dynamic=True, blocking_enable=True, blocking_qubits=10)
        job = sim.run(transpiled_circuit, shots=shots, dynamic=True, blocking_enable=True, blocking_qubits=20)
    else:
        job = sim.run(transpiled_circuit, shots=shots, dynamic=True)


    # Get the results and display them
    exp_result = job.result()
    exp_counts = exp_result.get_counts()
    ed = time.time()
    OVERALL_TIME += (ed - st)
    print(f'run job spends: {ed - st} sec')

    if MPI_ON:
        meta = exp_result.to_dict()['metadata']
        myrank = meta['mpi_rank']
        print(f'myrank: {myrank}')
    
    print("==========================================================")

    return exp_counts


if __name__ == '__main__':
    parse_cmd_parameters()
    A = gen_random_unitary()
    qc = create_circuit(A)

    if SIMULATION or MPI_ON: 
        exp_counts = simulation(qc)
    else:
        Q = prepare_snapshot(A, qc)
        self_sampling(qc, Q)

    print(f'total execution time (exclude snapshot): {OVERALL_TIME} sec')
