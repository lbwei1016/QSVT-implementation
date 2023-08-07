from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions import UnitaryGate

from qsvt.core_beta import *
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
        # 這有奇效，但不見得管用
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
        set_kappa = False
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
                        amplified |        n + 4         |         n + 3         |
                    --------------|----------------------|-----------------------|
                    not amplified |        n + 3         |         n + 2         |

                Suppose "real_only = True" and we have amplified the circuit. Let "qr" be 
                "qc's" quantum register, then
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
    # d = 500 (degree)
    phi_seq_500 = np.array([0.0, 785.3981503666581, -1.5707896677744382, -1.5708031005020633, -1.570789438663786, -1.5708033289530867, -1.5707892110029433, -1.5708035556670883, -1.5707889851804813, -1.5708037804794368, -1.5707887614151448, -1.570804003388877, -1.5707885395762176, -1.5708042240451108, -1.570788320143412, -1.5708044422958054, -1.570788103124698, -1.5708046579380417, -1.5707878886882625, -1.5708048710834104, -1.5707876771035165, -1.5708050810948206, -1.570787468604368, -1.570805288149573, -1.570787263111211, -1.570805491981592, -1.5707870609881083, -1.5708056924624842, -1.5707868622241599, -1.570805889428547, -1.5707866670318098, -1.5708060827141064, -1.5707864756395944, -1.5708062720043903, -1.5707862882803254, -1.5708064575472183, -1.5707861049295222, -1.570806638691675, -1.5707859258424215, -1.570806815723388, -1.5707857509704337, -1.5708069882877256, -1.5707855807703819, -1.5708071562056793, -1.570785415197078, -1.570807319577025, -1.5707852542588094, -1.5708074778487655, -1.570785098363746, -1.5708076313096657, -1.5707849474804627, -1.5708077796296551, -1.5707848017874628, -1.5708079226692442, -1.570784661418787, -1.5708080603331236, -1.570784526345382, -1.5708081926488526, -1.570784396935651, -1.5708083192404483, -1.5707842731455566, -1.570808440204222, -1.570784155118537, -1.5708085553858222, -1.5707840428895863, -1.5708086645217632, -1.5707839367300729, -1.570808767642305, -1.5707838366313072, -1.5708088647052119, -1.570783742754526, -1.570808955542581, -1.5707836549749432, -1.5708090400547474, -1.5707835736484121, -1.570809118183373, -1.5707834987868556, -1.570809189856589, -1.5707834304123762, -1.570809255031472, -1.5707833685141543, -1.5708093133438257, -1.5707833134108335, -1.570809365238476, -1.5707832650783433, -1.5708094101606906, -1.5707832234446937, -1.57080944836056, -1.5707831887741222, -1.5708094796832188, -1.5707831608817924, -1.5708095039427201, -1.57078314020654, -1.570809521253135, -1.5707831262474563, -1.570809531595619, -1.5707831194961932, -1.5708095347042235, -1.5707831199844517, -1.5708095307748824, -1.5707831274291517, -1.5708095196971195, -1.5707831421059446, -1.5708095014108294, -1.5707831640010905, -1.5708094758967774, -1.5707831932100005, -1.570809443195918, -1.570783229467448, -1.5708094030878086, -1.570783273195986, -1.570809355819229, -1.570783324154145, -1.5708093012128745, -1.570783382281396, -1.5708092394171875, -1.5707834479381002, -1.5708091702613427, -1.5707835206998357, -1.570809093709217, -1.570783600814968, -1.5708090099679055, -1.5707836881554909, -1.5708089189943921, -1.5707837828283555, -1.5708088206888595, -1.5707838847595341, -1.5708087152248842, -1.570783993901863, -1.5708086024610635, -1.570784110165444, -1.5708084824839852, -1.570784233634222, -1.5708083554327295, -1.570784364348059, -1.570808221325154, -1.5707845020982458, -1.570808080013129, -1.5707846468678932, -1.5708079316763357, -1.5707847986559575, -1.570807776459643, -1.5707849573941342, -1.5708076142509677, -1.5707851229464695, -1.5708074451573302, -1.570785295560973, -1.5708072693153736, -1.5707854746812764, -1.570807086903482, -1.570785660564919, -1.57080689752594, -1.570785853141595, -1.5708067018230911, -1.5707860521507693, -1.5708064996248514, -1.5707862574373002, -1.5708062909686078, -1.5707864691976852, -1.5708060761839864, -1.5707866872072545, -1.5708058552361979, -1.570786911180134, -1.5708056281936633, -1.5707871411007672, -1.5708053953469774, -1.5707873768696747, -1.5708051567168704, -1.5707876183794045, -1.570804912502658, -1.5707878651905547, -1.5708046629134778, -1.5707881175152152, -1.570804408020183, -1.5707883748858709, -1.5708041481932822, -1.570788637101432, -1.570803883459597, -1.570788904149496, -1.570803614186583, -1.570789175672223, -1.5708033406652326, -1.5707894512231595, -1.5708030629296241, -1.5707897307896608, -1.5708027815012777, -1.5707900140072837, -1.5708024968362888, -1.570790300326859, -1.5708022090229012, -1.5707905895044145, -1.5708019185828703, -1.5707908811157858, -1.5708016259647815, -1.5707911746503205, -1.570801331708444, -1.5707914694742364, -1.5708010362371971, -1.5707917652244519, -1.5708007404096591, -1.570792061214285, -1.5708004446452493, -1.5707923564238158, -1.5708001499583177, -1.5707926504249834, -1.5707998569346586, -1.5707929421218845, -1.5707995668140817, -1.5707932304147556, -1.5707992805694682, -1.570793514394964, -1.5707989992667193, -1.5707937926224298, -1.57079872455655, -1.5707940635716284, -1.5707984578905339, -1.5707943255118961, -1.5707982009146468, -1.5707945768133802, -1.5707979558020182, -1.5707948151769993, -1.5707977249033684, -1.5707950379719875, -1.5707975108186378, -1.5707952425367364, -1.5707973166761995, -1.5707954254192584, -1.570797146150982, -1.570795582604551, -1.5707970032532, -1.5707957096457874, -1.5707968933409664, -1.5707958009965632, -1.5707968224095044, -1.570795849790862, -1.57079679802562, -1.5707958475742445, -1.5707968295899222, -1.570795783491425, -1.5707969297284095, -1.570795642918229, -1.5707971157602127, -1.5707954048440542, -1.570797414018775, -1.5707950360006293, -1.570797868172149, -1.5707944750076266, -1.5707985695412978, -1.5707935748147162, -1.5707997888007093, -1.5707917071769741, -1.5708042949321512, -1.5707917071769741, -1.5707997888007093, -1.5707935748147162, -1.5707985695412978, -1.5707944750076266, -1.570797868172149, -1.5707950360006293, -1.570797414018775, -1.5707954048440542, -1.5707971157602127, -1.570795642918229, -1.5707969297284095, -1.570795783491425, -1.5707968295899222, -1.5707958475742445, -1.57079679802562, -1.570795849790862, -1.5707968224095044, -1.5707958009965632, -1.5707968933409664, -1.5707957096457874, -1.5707970032532, -1.570795582604551, -1.570797146150982, -1.5707954254192584, -1.5707973166761995, -1.5707952425367364, -1.5707975108186378, -1.5707950379719875, -1.5707977249033684, -1.5707948151769993, -1.5707979558020182, -1.5707945768133802, -1.5707982009146468, -1.5707943255118961, -1.5707984578905339, -1.5707940635716284, -1.57079872455655, -1.5707937926224298, -1.5707989992667193, -1.570793514394964, -1.5707992805694682, -1.5707932304147556, -1.5707995668140817, -1.5707929421218845, -1.5707998569346586, -1.5707926504249834, -1.5708001499583177, -1.5707923564238158, -1.5708004446452493, -1.570792061214285, -1.5708007404096591, -1.5707917652244519, -1.5708010362371971, -1.5707914694742364, -1.570801331708444, -1.5707911746503205, -1.5708016259647815, -1.5707908811157858, -1.5708019185828703, -1.5707905895044145, -1.5708022090229012, -1.570790300326859, -1.5708024968362888, -1.5707900140072837, -1.5708027815012777, -1.5707897307896608, -1.5708030629296241, -1.5707894512231595, -1.5708033406652326, -1.570789175672223, -1.570803614186583, -1.570788904149496, -1.570803883459597, -1.570788637101432, -1.5708041481932822, -1.5707883748858709, -1.570804408020183, -1.5707881175152152, -1.5708046629134778, -1.5707878651905547, -1.570804912502658, -1.5707876183794045, -1.5708051567168704, -1.5707873768696747, -1.5708053953469774, -1.5707871411007672, -1.5708056281936633, -1.570786911180134, -1.5708058552361979, -1.5707866872072545, -1.5708060761839864, -1.5707864691976852, -1.5708062909686078, -1.5707862574373002, -1.5708064996248514, -1.5707860521507693, -1.5708067018230911, -1.570785853141595, -1.57080689752594, -1.570785660564919, -1.570807086903482, -1.5707854746812764, -1.5708072693153736, -1.570785295560973, -1.5708074451573302, -1.5707851229464695, -1.5708076142509677, -1.5707849573941342, -1.570807776459643, -1.5707847986559575, -1.5708079316763357, -1.5707846468678932, -1.570808080013129, -1.5707845020982458, -1.570808221325154, -1.570784364348059, -1.5708083554327295, -1.570784233634222, -1.5708084824839852, -1.570784110165444, -1.5708086024610635, -1.570783993901863, -1.5708087152248842, -1.5707838847595341, -1.5708088206888595, -1.5707837828283555, -1.5708089189943921, -1.5707836881554909, -1.5708090099679055, -1.570783600814968, -1.570809093709217, -1.5707835206998357, -1.5708091702613427, -1.5707834479381002, -1.5708092394171875, -1.570783382281396, -1.5708093012128745, -1.570783324154145, -1.570809355819229, -1.570783273195986, -1.5708094030878086, -1.570783229467448, -1.570809443195918, -1.5707831932100005, -1.5708094758967774, -1.5707831640010905, -1.5708095014108294, -1.5707831421059446, -1.5708095196971195, -1.5707831274291517, -1.5708095307748824, -1.5707831199844517, -1.5708095347042235, -1.5707831194961932, -1.570809531595619, -1.5707831262474563, -1.570809521253135, -1.57078314020654, -1.5708095039427201, -1.5707831608817924, -1.5708094796832188, -1.5707831887741222, -1.57080944836056, -1.5707832234446937, -1.5708094101606906, -1.5707832650783433, -1.570809365238476, -1.5707833134108335, -1.5708093133438257, -1.5707833685141543, -1.570809255031472, -1.5707834304123762, -1.570809189856589, -1.5707834987868556, -1.570809118183373, -1.5707835736484121, -1.5708090400547474, -1.5707836549749432, -1.570808955542581, -1.570783742754526, -1.5708088647052119, -1.5707838366313072, -1.570808767642305, -1.5707839367300729, -1.5708086645217632, -1.5707840428895863, -1.5708085553858222, -1.570784155118537, -1.570808440204222, -1.5707842731455566, -1.5708083192404483, -1.570784396935651, -1.5708081926488526, -1.570784526345382, -1.5708080603331236, -1.570784661418787, -1.5708079226692442, -1.5707848017874628, -1.5708077796296551, -1.5707849474804627, -1.5708076313096657, -1.570785098363746, -1.5708074778487655, -1.5707852542588094, -1.570807319577025, -1.570785415197078, -1.5708071562056793, -1.5707855807703819, -1.5708069882877256, -1.5707857509704337, -1.570806815723388, -1.5707859258424215, -1.570806638691675, -1.5707861049295222, -1.5708064575472183, -1.5707862882803254, -1.5708062720043903, -1.5707864756395944, -1.5708060827141064, -1.5707866670318098, -1.570805889428547, -1.5707868622241599, -1.5708056924624842, -1.5707870609881083, -1.570805491981592, -1.570787263111211, -1.570805288149573, -1.570787468604368, -1.5708050810948206, -1.5707876771035165, -1.5708048710834104, -1.5707878886882625, -1.5708046579380417, -1.570788103124698, -1.5708044422958054, -1.570788320143412, -1.5708042240451108, -1.5707885395762176, -1.570804003388877, -1.5707887614151448, -1.5708037804794368, -1.5707889851804813, -1.5708035556670883, -1.5707892110029433, -1.5708033289530867, -1.570789438663786, -1.5708031005020633, -1.5707896677744382])

    # print(f'origin phi: {phi_seq}')

    # It's the adjoint of A, not A itself!
    X = np.conj(A.T)


    # Check the source code of "qsppack".
    if set_kappa == True:
        kappa = np.linalg.cond(X)
        # b = int(kappa ** 2 * np.log2(kappa / eps))
        # deg = int(np.sqrt(b * np.log2(4 * b / eps))) + 1


        # Heuristically adjust deg
        interval = 5
        if kappa <= 10 + interval:
            phi_seq = phi_seq_121
            # deg = 121
            # npts = 500
        # elif kappa <= 25:
        #     phi_seq = phi_seq_500
        #     # deg = int(1.2 * int(np.sqrt(b * np.log2(4 * b / eps))) + 1)
        #     # deg = min(deg, 500)
        #     # npts = 5000
        else:
            Wx_seq = []
            # if kappa <= 30 + interval: file = './qsvt/inv901.txt'
            # else: file = './qsvt/inv2501.txt'
            file = './qsvt/inv601.txt'

            with open(file, 'r') as f:
                Wx_seq = f.read().split('\n')
                Wx_seq = [float(x) for x in Wx_seq]
        
            _, phi_seq = convert_convention(Wx_seq)

        # print(f'degree (cvx): {deg}')
        print(f'deg: {len(phi_seq)-1}')

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

        # # print(f'Wx: {Wx_seq}')
        # _, phi_seq = convert_convention(Wx_seq)

        # print(f'phi_seq (deg = {deg}): {list(phi_seq)}')

    import time
    st = time.time()
    qc = QSVT(phi_seq, X, real_only=real_only)
    ed = time.time()
    print(f'QSVT spends: {ed - st} sec')

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