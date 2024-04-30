# Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES
#
# SPDX-License-Identifier: BSD-3-Clause

from qiskit import QuantumCircuit, transpile
from qiskit import Aer
from mpi4py import MPI
import argparse

from qsvt_linear_solver import gen_random_unitary, create_circuit
from qsvt.algorithms import linear_solver
import numpy as np
import time

def run(n_qubits, precision, use_cusvaer):
    st = time.time()

    simulator = Aer.get_backend('aer_simulator_statevector')
    simulator.set_option('cusvaer_enable', use_cusvaer)
    simulator.set_option('precision', precision)

    A = gen_random_unitary(n_qubits)
    # circuit = create_circuit(A)
    circuit = linear_solver(A)


    # circuit = create_ghz_circuit(n_qubits)
    circuit.measure_all()
    circuit = transpile(circuit, simulator)

    # 改 basis gate 也沒用
    # old_basis = ['cp', 'cx', 'id', 'rz', 's
    # x', 'u1', 'u2', 'u3', 'unitary', 'x', 'save_state_simple', 'save_statevector', 'set_state_simple']
    # new_basis = old_basis.remove('unitary')
    # circuit = transpile(circuit, simulator, basis_gates=new_basis)

    print('After transpilation. Ready to run...')
    job = simulator.run(circuit)
    result = job.result()

    ed = time.time()

    if MPI.COMM_WORLD.Get_rank() == 0:
        # print(f'basis gates: {simulator.configuration().basis_gates}')
        print(f'precision: {precision}')
        print(result.get_counts())
        print(f'backend: {result.backend_name}')
        print(f'time spent: {ed - st}')


parser = argparse.ArgumentParser(description="Qiskit ghz.")
parser.add_argument('--nbits', type=int, default=2, help='the number of qubits')
parser.add_argument('--precision', type=str, default='single', choices=['single', 'double'], help='numerical precision')
parser.add_argument('--disable-cusvaer', default=False, action='store_true', help='disable cusvaer')

args = parser.parse_args()

run(args.nbits, args.precision, not args.disable_cusvaer)
