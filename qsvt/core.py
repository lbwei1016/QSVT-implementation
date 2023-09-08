from qiskit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.extensions import UnitaryGate

import numpy as np

# tolorence for the norm of the block-encoded matrix
EPS = 1e-6


def QSP(
        phi_seq: np.ndarray, 
        # d: int, 
        a: float
) -> QuantumCircuit:
    """
        Description:
            Given a sequence of angles and a target real number, perform QSP (Quantum Signal Processing) 
            and return the circuit.
        Args:
            phi_seq: The sequence of angles for "signal processing operators".
            a: The real value to be transformed.

            [Deprecated]
                d: The degree of the target polynomial.
        Return:
            qc: The quantum circuit implementing QSP.
        Note:
            Please note that we use the "Wx" convention in this implementation.
            For convention converting, see "qsvt.core.convert_convention()".
        Example:
            See per application.
    """
    # precaution
    phi_seq = np.array(phi_seq)
    # In order to match Rz's definition.
    phi_seq *= -2 
    d = phi_seq.shape[0] - 1

    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)

    for k in range(1, d + 1):
        qc.rz(phi=phi_seq[k], qubit=0)
        qc.rx(theta=-2 * np.arccos(a), qubit=0)

    qc.rz(phi=phi_seq[0], qubit=0)

    return qc


def block_encode(A: np.ndarray) -> np.ndarray:
    """
        Description:
            Given a target matrix A, return a unitary block-encoding U of A, in the form:
                
                U = [ A                      sqrt(I - AA^\dagger) ]
                    [ sqrt(I - A^\dagger A)  -A^\dagger                    ].

            If the input matrix A has norm greater than 1, an exception would be raised! (Normalize it first!)
            (Why not normalize the input automatically? The author suppose that the caller should "completely know"
            what she/he is doing, which is better for programming and debugging (and understanding block-encoding itself).)
        Args:
            A: The matrix to block encode. (numpy.array())
        Return:
            U: A block-encoding of "A", in the form given above.
        Note:
            We have A = (<0| \otimes I)U(|0> \otimes I).

            This is not a efficient block-encoding. To efficiently block-encode a sparse matrix,
            see https://www.youtube.com/watch?v=d3f3JRo0WUo&ab_channel=InstituteforPure%26AppliedMathematics%28IPAM%29 
            (Chao Yang - Practical Quantum Circuits for Block Encodings of Sparse Matrices - IPAM at UCLA) for more information.
        Example:
            script:
                A = np.array([
                    [1, -1/3],
                    [-1/3, 1]
                ])
                
                U = block_encode(A)
                print(U)
            output:
                [[ 0.67082039+0.j -0.2236068 +0.j  0.67082039+0.j  0.2236068 +0.j]
                [-0.2236068 +0.j  0.67082039+0.j  0.2236068 +0.j  0.67082039+0.j]
                [ 0.67082039+0.j  0.2236068 +0.j -0.67082039+0.j  0.2236068 +0.j]
                [ 0.2236068 +0.j  0.67082039+0.j  0.2236068 +0.j -0.67082039+0.j]]
    """
    A = np.array(A) # precaution
    norm = np.linalg.norm(A)
    if abs(norm - 1.0) > EPS:
        raise Exception("The input matrix should be normalized first! E.g. divide it by \"np.linalg.norm(A)\".")

    n = A.shape[0]

    A_dag = np.conjugate(np.transpose(A))
    
    # this should Hermitian
    I_AAd = np.identity(n) - A @ A_dag 
    eigval, eigvec = np.linalg.eig(I_AAd)

    sq_I_AAd = eigvec * np.sqrt(eigval) @ np.linalg.inv(eigvec)
    

    I_AdA = np.identity(n) - A_dag @ A
    eigval, eigvec = np.linalg.eig(I_AdA)

    sq_I_AdA = eigvec * np.sqrt(eigval) @ np.linalg.inv(eigvec)


    U = np.zeros(shape=(2 * n, 2 * n), dtype=complex)

    U[0:n, 0:n] = A
    U[0:n, n:] = sq_I_AAd
    U[n:, 0:n] = sq_I_AdA
    U[n:, n:] = -A_dag

    return U

def QSVT(
        # n: int, 
        phi_seq: np.ndarray, 
        # d: int, 
        A: np.ndarray,
        convention = "R",
        real_only = True,
        # high_precision_block_encoding = True
) -> QuantumCircuit:
    r"""
        Description:
            Given a sequence of angles and the target matrix, perform QSVT (Quantum Singular Value Transformation) 
            and return the circuit.
        Args:
            phi_seq: The sequence of angles for projector controlled phase shift.
            A: The matrix to be transformed.
            convention: Angle sequence convention. By default, "R" is used. Another possibility is "Wx".
            real_only: The given "phi_seq" is assumed to implement a complex polynomial. However, if 
                       "real_only = True", only the real part of the polynomial is implemented, which
                       costs 1 more ancilla qubit.

            [Deprecated]
                n: The dimension of the input matrix. (e.g. A 8*8 matrix has dimension 3 (= lg8).)
                d: The degree of the target polynomial.
                high_precision_block_encoding: If this is "True", then "pennylane.BlockEncode()" is used to 
                                               block encode the input matrix, since my own implementation lose
                                               precision easily on large-condition-number matrices. If you 
                                               don't want to depend on pennylane, set this to "False".
                (block_encode() has been fixed!)
        Return:
            qc: The quantum circuit implementing QSVT. This is an (n+1)-qubit circuit, where
                "A" is a "2^n by 2^n" matrix. Let "qr" be "qc's" quantum register, then
                    qr[0:n]: The register we are in interest.
                    qr[-1]: Ancilla qubit for block-encoding and phase shift.
                The user should "post-select 0" for "qr[-1]" to obtain the desired outcome.

                If "real_only = True", then "qc" is an (n+2)-qubit circuit, where qr[0:n] is still
                the register in interest, and the remaining are ancilla qubits.
        Note:
            In the literature, there are two "projectors" \tilde{\Pi} and \Pi that 
            specify the encoding of "A" (i.e., projector unitary encoding). However, 
            to simplify things, we let the projectors be 

                \tilde{\Pi} = \Pi = (|0><0| \otimes I),

            which means that "A" is block-encoded at the "top-left" block of a unitary "U".
            See "qsvt.core.block_encode()" for more details.

            In this implementation, the returned circuit is a (n+1)-qubit register, where
            the first "n" qubits (numbered: 0 ~ n-1) are for our target matrix "A", the "n-th"
            qubit is for block-encoding and phase shift.

            Also note that we use the "reflection" convention ("R" convention) in this implementation.
            For convention converting, see "qsvt.core.convert_convention()".

            Finally, the qubit for block encoding ("work" qubit) also acts as a signal qubit, for 
            projector phase shifts. This way, the number of ancilla qubit is reduced from 2 to 1.
        Example:
            See per application.
    """
    if convention == 'Wx':
        d, phi_seq = convert_convention(phi_seq)
    else:
        # precaution
        phi_seq = np.array(phi_seq)
    # In order to match Rz's definition. (rotation Z)
    phi_seq *= -2 
    d = phi_seq.shape[0] - 1

    n = int(np.log2(A.shape[0]))

    if real_only:
        qr = QuantumRegister(n + 2)
        block, work, aux = qr[0:n], qr[-2], qr[-1]
    else:
        qr = QuantumRegister(n + 1)
        block, work = qr[0:n], qr[-1]

    qc = QuantumCircuit(qr)

    # if high_precision_block_encoding:
    #     import pennylane as qml
    #     U = qml.matrix(qml.BlockEncode(A, wires=range(n + 1)))
    # else:
    #    U = block_encode(A)

    U = block_encode(A)

    U_gate = UnitaryGate(data=U, label='$U$')
    Ud_gate = U_gate.adjoint()

    # For some kind of implementing real polynomial. See [Quantum Algorithms for Scientific Computation] p.107.
    if real_only:
        qc.h(aux)

    for k in range(1, d // 2 + 1):
        if n > 1:
            qc.append(U_gate, qargs=[*block, work])
        else:
            qc.append(U_gate, qargs=[block, work])

        ##################################################

        if real_only:
            qc.cx(control_qubit=work, target_qubit=aux, ctrl_state=0)

            if d % 2 == 1:
                qc.rz(phi=phi_seq[2 * k + 1], qubit=aux)
            else:
                qc.rz(phi=phi_seq[2 * k], qubit=aux)

            qc.cx(control_qubit=work, target_qubit=aux, ctrl_state=0)
        else:
            if d % 2 == 1:
                qc.rz(phi=-phi_seq[2 * k + 1], qubit=work)
            else:
                qc.rz(phi=-phi_seq[2 * k], qubit=work)


        #################################################
        
        if n > 1:
            qc.append(Ud_gate, qargs=[*block, work])
        else:
            qc.append(Ud_gate, qargs=[block, work])
        

        #################################################

        if real_only:
            qc.cx(control_qubit=work, target_qubit=aux, ctrl_state=0)

            if d % 2 == 1:
                qc.rz(phi=phi_seq[2 * k], qubit=aux)
            else:
                qc.rz(phi=phi_seq[2 * k - 1], qubit=aux)

            qc.cx(control_qubit=work, target_qubit=aux, ctrl_state=0)
        else:
            if d % 2 == 1:
                qc.rz(phi=-phi_seq[2 * k], qubit=work)
            else:
                qc.rz(phi=-phi_seq[2 * k - 1], qubit=work)
            
        #################################################


        # Since we want to convert this circuit to a gate, no barrier is allowed.
        # qc.barrier()

    if d % 2 == 1:
        if n > 1:
            qc.append(U_gate, qargs=[*block, work])
        else:
            qc.append(U_gate, qargs=[block, work])

    if real_only:
        qc.cx(control_qubit=work, target_qubit=aux, ctrl_state=0)
        qc.rz(phi=phi_seq[1], qubit=aux)
        qc.cx(control_qubit=work, target_qubit=aux, ctrl_state=0)
    else:
        qc.rz(phi=-phi_seq[1], qubit=work)
    
    # Implement only the real part of the given polynomial. See [Quantum Algorithms for Scientific Computation] p.107.
    if real_only:
        qc.h(aux)
        
    return qc


def convert_convention(Wx_seq: list) -> tuple:
    """
        Description:
            Convert "Wx" convention phases to "R" convention ones.
        Args:
            Wx_seq: A sequence (list or numpy.ndarray) of angles in "Wx" convention.
        Return:
            d: The degree of the polynomial represented by "R_seq". (Can be passed to qsvt.core.QSVT().) 
            R_seq: A sequence (numpy.ndarray) of angles in "R" convention.
    """
    R_seq = np.zeros(len(Wx_seq))
    d = len(Wx_seq) - 1
    R_seq[1] = Wx_seq[0] + Wx_seq[d] + (d - 1) * np.pi / 2
    for j in range(2, d + 1):
        R_seq[j] = Wx_seq[j - 1] - np.pi / 2
    return (d, R_seq)


def implement_real(
        circ: QuantumCircuit,
        circ_s: QuantumCircuit, 
        # n: int
) -> QuantumCircuit:
    """
        [Deprecation]
            This function is deprecated. The functionality of implementing only the 
            real part of a polynomial is integrated into QSVT().

            The main reason for deprecation is due to the inefficiency of `qc.to_gate()`,
            which leads to long circuit preparation time and deep transpiled circuit.
        Description:
            In many cases, we only want the "real" part of our approximated 
            polynomial "Poly()" in QSVT. Given two input quantum circuits that 
            compute "Poly()" and "Poly*()" (complex conjugate) respectively, 
            return a circuit that computes 
            
                Poly_R() = (Poly() + Poly*()) / 2,

            upon post-selecting "0" at the first (numbered: 0) qubit.
        Args:
            circ: The quantum circuit that computes "Poly()".
            circ_s: The quantum circuit that computes "Poly*()". (star)

            [Deprecated]
                n: The input size (number of qubits) of "circ" (and "circ_s" alike).
                   (If you don't know what this value should be, try (m + 2), 
                   when our input matrix for QSVT is of size (2^m * 2^m).)
        Return:
            qc: A quantum circuit that computes "Poly_R()". Let "qr" be "qc's" quantum 
                register and let "n" be the number of qubits of "circ", then
                    qr[0:-1]: The register we are in interest;
                    qr[-1]: Ancilla qubit for LCU (Linear Combination of Unitaries),
                which shows that "qc" is a (n+1)-qubit circuit. 

                The user should "post-select 0" for "qr[-1]" to obtain the desired
                outcome.
    """
    n = circ.num_qubits

    qr = QuantumRegister(n + 1)
    qc = QuantumCircuit(qr)

    # aux, work = qr[0], qr[1:]
    work, aux = qr[:-1], qr[-1]

    qc.h(aux)

    import time 

    st = time.time()
    U_gate = circ.to_gate(label='$U_\Phi$').control(num_ctrl_qubits=1, ctrl_state=0)
    Ud_gate = circ_s.to_gate(label='$U_{-\Phi}$').control(num_ctrl_qubits=1, ctrl_state=1)
    ed = time.time()
    print(f'circ to gate spends: {ed - st} sec')

    qc.append(U_gate, qargs=[aux, *work])
    qc.append(Ud_gate, qargs=[aux, *work])

    qc.h(aux)

    return qc