from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions import UnitaryGate

from qsvt.core import *
# from qsvt.helper import cvx_poly_coef
# from .Solvers.QSP_solver import QSP_Solver

import numpy as np
from warnings import warn


def amplitude_amplification(
        circ: QuantumCircuit,
        state: str,
        positions: list,
        transformation = 'chebyshev'
) -> QuantumCircuit:
    """
        Description:
            Given a quantum circuit that prepares some initial state |s>, and 
            given a target state |t>, amplify the value <t|s>.

            This is usually used as a subroutine, especially for algorithms that
            needs to "post-select" some desired states. (like "block-encoding"
            and LCU).
        Args:
            circ: A quantum circuit that prepares the initial state |s>.
            state: A string specifying |t>, e.g., "0010".
            positions: A list of non-negative integers indicating which qubits
                       are correlated to this algorithm. See example below.
            transformation: A string indicating what transformation to use. 
                            So far, we have "chebyshev", "sign", and "AA".
        Return:
            qc: A quantum circuit after AA. Let "qr" be "qc's" quantum 
                register then
                    qr[0:-1]: The register we are in interest ("circ");
                    qr[-1]: Ancilla qubit for phase shift.
                which shows that "qc" is a (n+1)-qubit circuit. ("circ" is a n-qubit register.)
        Note:
            Although this algorithm is based on QSVT, we do "not" use the primitive
            "qsvt.core.QSVT()". This is because "qsvt.core.QSVT()" utilizes block-
            encoding, which itself needs to be amplified! Thus, this implementation
            is isolated, to ensure performance.

            However, all three transformations ("chebyshev", "sign", and "AA") are not ideal. 
            One should try out different transformations in a simulator and check which 
            performs best. (By default, "chebyshev" is adopted, since Chebyshev polynomials 
            are the most natural transformations in QSVT, which have relatively shallow circuit depths.
            However, Chebyshev polynomial's performance is not guaranteed when the initial state
            is not in uniform superposition.)

            A final warning: amplitude amplification induces imaginary parts; be careful!
        Example:
            Description:
                Now we have a quantum register consists of 6 qubits, and we want the qubits
                indexed "0", "2", and "5" be "0", "1", and "1", respectively. Note that the 
                other qubits ("1", "3", and "4") are intact. The following script performs
                amplitude amplification over the state "011", for qubits indexed [0, 2, 5].
            script: 
                n = 6
                positions = [0, 2, 5]
                state = "011"
                circ = QuantumCircuit(n)
                circ.h([0, 2, 5]) # uniform superposition among qubit [0, 2, 5]

                qc = amplitude_amplification(circ, st, ls, 'chebyshev')

                cr = ClassicalRegister(len(state))
                qc.add_register(cr)
                qc.measure(state, cr)
                sim = AerSimulator()
                transpiled_circuit = transpile(qc, sim)

                shots = 10000
                job = sim.run(transpiled_circuit, shots=shots, dynamic=True)

                exp_result = job.result()
                exp_counts = exp_result.get_counts()
                plot_histogram(exp_counts)
            output:
                "011": roughly 94% in the plot
    """
    n = circ.num_qubits
    search_len = len(positions)
    qr = QuantumRegister(n + 1)
    qc = QuantumCircuit(qr)

    if transformation == 'chebyshev':
        d = 2 * search_len - 1
        # R convention already
        phi_seq = np.array([np.pi / 2 for _ in range(d + 1)])
        phi_seq[1] = (1 - d) * np.pi / 2
    elif transformation == 'sign': 
        # Use "Sign(x)". (Performance is not guaranteed.)
        # Usually, a "too-large" initial probability leads to bad performance.
        Wx_seq = [-0.14030619300699504, 0.31770910376491157, -0.2159687350010641, 0.5376368100800262, -0.6561965311157821, -0.570587000131174, -0.029578881455564865, 0.29810643597957565, 1.185054925225846, -0.26170748520431114, -0.06251690387218234, -0.5952101173964184, -0.3062334505793054, 0.46333819051152153, -0.9745105865317089, -0.974510586522954, 0.46333819051936453, -0.3062334505815665, 2.5463825361924144, -0.06251690388673614, -0.26170748519919007, -1.9565377283595162, 0.2981064359713328, -0.029578881442567484, -0.5705870001297568, 2.485396122476667, -2.6039558435092593, -0.21596873500304925, 0.3177091037658908, 1.430490133786587]
        d, phi_seq = convert_convention(Wx_seq)
    elif transformation == 'AA':
        # This works well on low probabilities (but performance is not guaranteed).
        # pyqsp --plot-npts=4000 --plot-positive-only --plot-magnitude --plot --seqargs=21,1.0e-20 --seqname fpsearch angles --output-json
        Wx_seq = [-1.6289766249603586, -1.454208011876622, -1.7462485302871582, -1.3357954977223736, -1.8662600481340057, -1.213725811224108, -1.9908459961512452, -1.0861701139264175, -2.1218145928902925, -0.9513642785202095, -2.260851457478332, -0.8077485591343427, -2.4093286369239166, -0.6542153401660733, -2.567999066787916, -0.49046601602792345, -2.736605610844485, -0.3174051062134276, -2.9135294062485397, -0.13739160893601188, -3.0957015611621053, -3.0957015611621053, -0.13739160893601188, -2.9135294062485397, -0.3174051062134276, -2.736605610844485, -0.49046601602792345, -2.567999066787916, -0.6542153401660733, -2.4093286369239166, -0.8077485591343427, -2.260851457478332, -0.9513642785202095, -2.1218145928902925, -1.0861701139264175, -1.9908459961512452, -1.213725811224108, -1.8662600481340057, -1.3357954977223736, -1.7462485302871582, -1.454208011876622, -1.6289766249603586]
        # pyqsp --plot-npts=4000  --plot-magnitude --plot --seqargs=10,0.8 --seqname fpsearch angles --output-json
        # Wx_seq = [-1.575769485956323, -1.5606190601810968, -1.5866845573794672, -1.5483044916717168, -1.6014016785461402, -1.52944545603768, -1.6278832953387132, -1.4869235949478299, -1.7143624281082808, -1.156048157872872, -1.156048157872872, -1.7143624281082808, -1.4869235949478299, -1.6278832953387132, -1.52944545603768, -1.6014016785461402, -1.5483044916717168, -1.5866845573794672, -1.5606190601810968, -1.575769485956323]
        d, phi_seq = convert_convention(Wx_seq)
    else:
        warn('Please specify another transformation!')
        raise Exception(NotImplemented)
    phi_seq *= -2

    
    block, aux = qr[:-1], qr[-1]
    circ_gate = circ.to_gate(label='circ')
    circ_gate_d = circ_gate.inverse()


    for k in range(1, d // 2 + 1):
        # U
        qc.append(circ_gate, qargs=block)

        ##### projector controlled phase shift operations (left SV space) #####
            ### Controlled-Pi-NOT ###
        for i in range(search_len):
            if state[i] == '0':
                qc.x(qubit=positions[i])
        # qc.mcx(control_qubits=block, target_qubit=aux)
        qc.mcx(control_qubits=positions, target_qubit=aux)
            ### Z rotation ###
        qc.rz(phi=phi_seq[2 * k + 1], qubit=aux)
            ### Controlled-Pi-NOT ###
        # qc.mcx(control_qubits=block, target_qubit=aux)
        qc.mcx(control_qubits=positions, target_qubit=aux)
        for i in range(search_len):
            if state[i] == '0':
                qc.x(qubit=positions[i])


        qc.barrier() # comment this if needed

        # U_inv
        qc.append(circ_gate_d, qargs=block)

        ##### projector controlled phase shift operations (right SV space) #####
            ### Controlled-Pi-NOT ###
        # qc.h(qubit=block)
        # qc.append(circ_gate_d, qargs=block)
        qc.x(qubit=block)
        qc.mcx(control_qubits=block, target_qubit=aux)
        qc.x(qubit=block)
        # qc.h(qubit=block)
        # qc.append(circ_gate, qargs=block)

            ### Z rotation ###
        qc.rz(phi=phi_seq[2 * k], qubit=aux)

            ### Controlled-Pi-NOT ###
        # qc.h(qubit=block)
        # qc.append(circ_gate_d, qargs=block)
        qc.x(qubit=block)
        qc.mcx(control_qubits=block, target_qubit=aux)
        qc.x(qubit=block)
        # qc.h(qubit=block)
        # qc.append(circ_gate, qargs=block)

        # qc.barrier()

    # U
    qc.append(circ_gate, qargs=block)

    ##### projector controlled phase shift operations (left SV space) #####
        ### Controlled-Pi-NOT ###
    for i in range(search_len):
        if state[i] == '0':
            qc.x(qubit=positions[i])
    # qc.mcx(control_qubits=block, target_qubit=aux)
    qc.mcx(control_qubits=positions, target_qubit=aux)
        ### Z rotation ###
    qc.rz(phi=phi_seq[1], qubit=aux)
        ### Controlled-Pi-NOT ###
    # qc.mcx(control_qubits=block, target_qubit=aux)
    qc.mcx(control_qubits=positions, target_qubit=aux)
    for i in range(search_len):
        if state[i] == '0':
            qc.x(qubit=positions[i])

    return qc


def phase_estimation(
        U: np.ndarray,
        precision: int,
        prepare_eigenvec: QuantumCircuit
) -> QuantumCircuit:
    """
        Description:
            Phase estimation subroutine. Although this algorithm is designed under 
            the framework of QSVT, after some simplification, it is actually the 
            "ordinary" phase estimation (the one with inverse QFT). Thus, no angle
            sequence is required, and the circuit is shallow. However, this means 
            that this implementation does not have some "robust" properties mentioned
            in "A Grand Unification of Quantum algorithms". Refer to that for more details.
        Args:
            U: A "2^n by 2^n" sized unitary matrix.
            precision: The number of bits to store the desired eigenvalue.
            prepare_eigenvec: A quantum circuit that prepares a eigenvector for "U".
        Return:
            qc: A quantum circuit after performing phase estimation. Let "qr" be "qc's" 
                quantum register, then
                    qr[0:precision]: Where the estimated phase is stored.
                    qr[precision:]: Register for an eigenvector.
                which shows that "qc" is a (precision + n)-qubit circuit. ("A" is "2^n by 2^n".)
        Example:
            script:
                U = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, np.e ** (2 * np.pi * 1j * (2/9))]
                ])

                prepare_eigenvec = QuantumCircuit(2)
                prepare_eigenvec.x([0, 1])

                precision = 6
                qc = algo.phase_estimation(U, precision, prepare_eigenvec)

                cr = ClassicalRegister(precision)
                qc.add_register(cr)
                qc.measure(list(range(precision)), cr)

                sim = AerSimulator()
                transpiled_circuit = transpile(qc, sim)

                shots = 1000
                job = sim.run(transpiled_circuit, shots=shots, dynamic=True)

                exp_result = job.result()
                exp_counts = exp_result.get_counts()
                plot_histogram(exp_counts)
            output:
                "001110" with roughly 83% in the plot.
    """
    # The input matrix U should be of size (2^n * 2^n).
    eigenvec_dim = int(np.log2(U.shape[0]))
    print(f'eigvec_dim: {eigenvec_dim}')

    qr = QuantumRegister(precision + eigenvec_dim)
    # cr = ClassicalRegister(precision)
    # qc = QuantumCircuit(qr, cr)
    qc = QuantumCircuit(qr)

    phase = qr[0:precision]
    eigenvec = qr[-eigenvec_dim:]

    # qc.x(eigenvec) # prepare eigenstate for U
    qc.append(prepare_eigenvec, qargs=eigenvec)

    for j in range(precision - 1, -1, -1):
        bit = precision - 1 - j
        qc.h(phase[bit])
        
        for r in range(2, precision - j + 1):
            qc.cp(
                theta = -(2 * np.pi / (2 ** r)),
                control_qubit = phase[bit - r + 1], 
                target_qubit = phase[bit]
            )

        U_gate = UnitaryGate(data=U**(2**j), label=f'$U$^{2**j}').control(num_ctrl_qubits=1)
        if eigenvec_dim == 1:
            qc.append(U_gate, [phase[bit], eigenvec])
        else:
            qc.append(U_gate, [phase[bit], *eigenvec])
        qc.h(phase[bit])

        qc.barrier()

    # qc.measure(phase, cr)

    return qc


def linear_solver(
        A: np.ndarray,      
        real_only = True,
        amplify = 'NONE',
        eps = 0.01,
        set_degree = 0
        # set_kappa = False
) -> QuantumCircuit:
    r"""
        Description:
            Given an invertible and well-conditioned (See "A Grand Unification of Quantum
            Algorithms".) matrix "A", return a circuit that computes "A^{-1}", up to some
            global phase. 
        Args:
            A: The matrix to be inverted.
            real_only: Indicate whether to fully implement "A^{-1}". (See Note for details.)
            amplify: If "amplify != 'NONE'", use "amplify" to perform AA. 
                     (amplify = 'chebyshev', 'sign', or 'AA')
            eps: Precision of the approximated polynomial (approximating 1/x).
            set_kappa: By default, the condition number (kappa) of the input matrix is assumed 
                       to be "10", and the polynomial is approximated to "1 / (2*kappa*x)". 
                       However, if "set_kappa = True", the aprroximated polynomial would follow
                       the "true" condition number of the input matrix. (Warning: It is recommended
                       that for input with small condition number, "set_kappa = False".)
        Return:
            qc: A quantum circuit that computes "A^{-1}". There are 4 possibilities regarding
                the return "qc": "real_only" or not and "amplify" or not. 

                Let "A" be a "2^n * 2^n" matrix. We have the following form:

                    [# of qubits] |   real_only = True   |   real_only = False   |   
                    --------------|----------------------|-----------------------|
                        amplified |        n + 3         |         n + 2         |
                    --------------|----------------------|-----------------------|
                    not amplified |        n + 2         |         n + 1         |

                Suppose "real_only = True" and we have amplified the circuit. Let "qr" be 
                "qc's" quantum register, then
                (below comment outdated: 2023/08/08)
                    qr[0:n]: The register we are in interest;
                    qr[n]: Ancilla qubit for block-encoding;
                    qr[n+1]: Ancilla qubit for phase shift (QSVT)
                    qr[n+2]: Ancilla qubit for LCU;
                    qr[n+3]: Ancilla qubit for phase shift (AA).
                To obtain desired outcomes, the user should "post-select 00" for "qr[n]" and
                "qr[n+2]".
        Note:
            Note that if "real_only" is set to "False", then the resulting
            circuit doesn't compute "A^{-1}" exactly: only the "real part" of the applied
            state may be correct! (The user should somehow get rid of the imaginary part.)
        Example:
            See Documentation. 
    """
    # R convention (angle sequence for 1/x; fixed)
    # d = 121 (degree)
    phi_seq_121 = np.array([0.0, 190.0663500808966, -1.570792332211534, -1.570803175587046, -1.57078533277455, -1.570813154313031, -1.5707714941803497, -1.5708319139087061, -1.570746548967131, -1.5708645354861015, -1.570704514471793, -1.5709179886472342, -1.5706373438081545, -1.571001491350469, -1.5705345583001469, -1.5711268622308265, -1.5703829360143802, -1.5713087797005645, -1.570166285962922, -1.571564998893927, -1.569865271403714, -1.5719164148231348, -1.5694574674921569, -1.5723869687052803, -1.5689174744388823, -1.573003377969213, -1.5682173717552426, -1.5737946587196114, -1.5673272644962362, -1.5747913643979465, -1.5662161662173397, -1.5760246155293127, -1.5648531373324035, -1.5775248154467165, -1.5632087158671584, -1.579320041196319, -1.5612567054040773, -1.5814340652773506, -1.558976360068157, -1.5838839313870032, -1.5563550827287989, -1.5866769793775468, -1.5533916844518771, -1.5898072585315322, -1.5501003248952576, -1.5932512682256263, -1.5465150426507057, -1.5969631792151664, -1.5426947047965607, -1.6008699021892487, -1.538727711133374, -1.6048668428324833, -1.5347354156790864, -1.6088157324178185, -1.5308725950860975, -1.612546377782502, -1.5273229730395141, -1.615864308131166, -1.5242880602670779, -1.6185655920144242, -1.521968776260039, -1.6204583585560368, -1.6204583585560368, -1.521968776260039, -1.6185655920144242, -1.5242880602670779, -1.615864308131166, -1.5273229730395141, -1.612546377782502, -1.5308725950860975, -1.6088157324178185, -1.5347354156790864, -1.6048668428324833, -1.538727711133374, -1.6008699021892487, -1.5426947047965607, -1.5969631792151664, -1.5465150426507057, -1.5932512682256263, -1.5501003248952576, -1.5898072585315322, -1.5533916844518771, -1.5866769793775468, -1.5563550827287989, -1.5838839313870032, -1.558976360068157, -1.5814340652773506, -1.5612567054040773, -1.579320041196319, -1.5632087158671584, -1.5775248154467165, -1.5648531373324035, -1.5760246155293127, -1.5662161662173397, -1.5747913643979465, -1.5673272644962362, -1.5737946587196114, -1.5682173717552426, -1.573003377969213, -1.5689174744388823, -1.5723869687052803, -1.5694574674921569, -1.5719164148231348, -1.569865271403714, -1.571564998893927, -1.570166285962922, -1.5713087797005645, -1.5703829360143802, -1.5711268622308265, -1.5705345583001469, -1.571001491350469, -1.5706373438081545, -1.5709179886472342, -1.570704514471793, -1.5708645354861015, -1.570746548967131, -1.5708319139087061, -1.5707714941803497, -1.570813154313031, -1.57078533277455, -1.570803175587046, -1.570792332211534])

    # It's the adjoint of A, not A itself!
    X = np.conj(A.T)

############ Directly calculate the desired polynomial (may be slow) ############
    # if set_kappa == True:
    #     kappa = np.linalg.cond(X)
    #     # b = int(kappa ** 2 * np.log2(kappa / eps))
    #     # deg = int(np.sqrt(b * np.log2(4 * b / eps))) + 1


    #     # Heuristically adjust deg
    #     interval = 5.0001
    #     if kappa <= 10 + interval:
    #         phi_seq = phi_seq_121
    #         # deg = 121
    #         # npts = 500
    #     # elif kappa <= 25:
    #     #     phi_seq = phi_seq_500
    #     #     # deg = int(1.2 * int(np.sqrt(b * np.log2(4 * b / eps))) + 1)
    #     #     # deg = min(deg, 500)
    #     #     # npts = 5000
    #     else:
    #         Wx_seq = []
    #         # if kappa <= 30 + interval: file = './qsvt/inv901.txt'
    #         # else: file = './qsvt/inv2501.txt'
    #         if kappa <= 15 + interval:
    #             file = './qsvt/inv601.txt'
    #         else:
    #             file = './qsvt/inv2501.txt'

    #         with open(file, 'r') as f:
    #             Wx_seq = f.read().split('\n')
    #             Wx_seq = [float(x) for x in Wx_seq]
        
    #         _, phi_seq = convert_convention(Wx_seq)

        # opts = {
        #     'npts': npts,
        #     'epsil': eps,
        #     'fscale': 1,
        #     'intervals': [1/kappa, 1],
        #     'isplot': True,
        #     'objnorm': np.inf
        # }


        # def x_inverse(x):
        #     return 1 / (2*kappa * x)
        

        # coef_full = cvx_poly_coef(x_inverse, deg, opts)
        # parity = deg % 2
        # coef = coef_full[parity::2]

        # opts = {
        #     'maxiter': 100,
        #     'criteria': 1e-12,
        #     'useReal': True,
        #     # 'useReal': False,
        #     'targetPre': True,
        #     # 'method': 'Newton'
        # }

        # Wx_seq, out = QSP_Solver(coef, parity, opts)

        # _, phi_seq = convert_convention(Wx_seq)


############ Apply pre-calculated polynomials ############
    if set_degree == 0:
        phi_seq = phi_seq_121
    else:
        Wx_seq = []
        if set_degree == 1:
            file = './qsvt/inv_k50_d601.txt'
        elif set_degree == 2:
            file = './qsvt/inv_k150_d1501.txt'
        elif set_degree == 3:
            file = './qsvt/inv_k200_d2001.txt'
        else:
            file = './qsvt/inv_k1000_d5001.txt'

        with open(file, 'r') as f:
            Wx_seq = f.read().split('\n')
            Wx_seq = [float(x) for x in Wx_seq]
    
        _, phi_seq = convert_convention(Wx_seq)

    print(f'deg: {len(phi_seq)-1}')



    # import time
    # st = time.time()
    qc = QSVT(phi_seq, X, real_only=real_only)
    # ed = time.time()
    # print(f'QSVT spends: {ed - st} sec')


    # Apply amplitude amplification (may significantly increase circuit depth)
    if amplify != 'NONE':
        n = qc.num_qubits
        if real_only:
            measure_qubits = [n - 2, n - 1]
            measure_states = "00"
        else:
            measure_qubits = [n - 2]
            measure_states = "0"

        qc = amplitude_amplification(qc, measure_states, measure_qubits, amplify)
    
    return qc