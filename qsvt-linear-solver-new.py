from qiskit.quantum_info import Statevector

import numpy as np

from qsvt.algorithms import linear_solver
from qsvt.helper import total_variation, gen_random_matrix

from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram

import time
import getopt, sys



TOTAL_TIME = 0
AA_On = False
set_degree = 0
simulation_method = 'statevector'
# matrix size exponent
N = 1


# Remove 1st argument from the
# list of command line arguments
argumentList = sys.argv[1:]

# Options
options = "hN:ad:s:"

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
        elif currentArgument in ("-s", "--simulation_method"):
            if currentValue == 'm':
                simulation_method = "matrix_product_state"
            else:
                simulation_method = "statevector"  
            # simulation_method = currentValue
            print(f'simulation_method = {simulation_method}')
                
except getopt.error as err:
    # output error, and return with an error code
    print (str(err))

# print(f'simulation_method = {simulation_method}')


#########################################################################

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
# print(f'A:\n{A}')


st = time.time()
if not AA_On:
    qc = linear_solver(A, set_degree=set_degree)
else:
    qc = linear_solver(A, set_degree=set_degree, amplify='AA')
ed = time.time()

TOTAL_TIME += (ed - st)
print(f'prepare circuit spends: {ed - st} sec')



print('==================================')

st = time.time()
state = Statevector(qc)
ed = time.time()
snap_time = ed - st
print(f'prepare state snapshot spends: {snap_time} sec')

n = qc.num_qubits
print(f'number of qubits: {n}')

# for AA or not
if AA_On:
    measure_qubits = [n - 3, n - 2]
else:
    measure_qubits = [n - 2, n - 1]

exp_outcome = "00"


st = time.time()
while True:
    outcome, mstate = state.measure(measure_qubits)
    if outcome == exp_outcome: break
ed = time.time()
# print(f'post-measurement state: {mstate}')
print(f'post-selection spends: {ed - st} sec')


st = time.time()
res = np.linalg.solve(A, np.array([1] + [0] * (2 ** N - 1)))
ed = time.time()
res /= np.linalg.norm(res)

# print(f'res: {res}')
print(f'#### Classically solving Ax=b time spent: {ed - st} sec ####')


# Calculate total variation
print('==================================')
P = np.array([np.linalg.norm(mstate[i]) ** 2 for i in range(2 ** N)])
# P = np.array([np.linalg.norm(x)**2 for x in P])
Q = np.array([x ** 2 for x in res])

# print(f'kappa: {kappa}')
print(f'total_variation (exact): {total_variation(P, Q)}')

print('==================================')

################################ Self-sampling ########################################################
st = time.time()
# state = Statevector(qc)
P = np.array([np.linalg.norm(x)**2 for x in state])
shots = 100000

a = np.random.choice(a=list(range(2 ** n)), p=P, size=shots)
a = np.sort(a)
unique_elements, counts = np.unique(a, return_counts=True)
counts = dict(zip(unique_elements, counts))
# print(counts)
# plot_histogram(counts)

ed = time.time()
TOTAL_TIME += (ed - st)
print(f'sampling time: {ed - st + snap_time} sec')

################################ Simulation ######################################################

# qc.measure_all()
# print(f'qc depth: {qc.depth()}')

# sim = AerSimulator(method=simulation_method)
# # sim = AerSimulator(method='density_matrix')
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

# # Get the results and display them
# exp_result = job.result()
# counts = exp_result.get_counts()
# ed = time.time()
# TOTAL_TIME += (ed - st)
# print(f'run job spends: {ed - st} sec')

#####################################################################################

st = time.time()
SIZE = 2 ** N
valid_count = np.zeros(shape=(SIZE))
success = 0
for data in counts:
    if data < SIZE:
        success += counts[data]
        valid_count[int(data)] = counts[data]

valid_count /= shots
# valid_count /= np.linalg.norm(valid_count)
valid_count /= np.sum(valid_count)
print(f'sucess ratio: {success / shots}')
# print(f'valid_count: {valid_count};\nsum (should = 1): {np.sum(valid_count)}')

tot_var = total_variation(valid_count, Q)
ed = time.time()
# print(f'total var. (sample) time spent: {ed - st}')
print(f'total_variation (sample): {tot_var}')


###############################################################################
print('==================================')

print(f'total execution time (exclude snapshot): {TOTAL_TIME} sec')