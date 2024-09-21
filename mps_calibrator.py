# Calibrating M from empirical time series - BFGS method

import tensorflow as tf
from tensorflow.python.keras import Model
#from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
from scipy.linalg import eig
import tensorflow_probability as tfp
import pandas as pd
from scipy.linalg import eigvals
from scipy.sparse.linalg import eigs

import mps_comparator




def augment_combine(A, B, small):
    num_states = len(A)
    nA, mA = A[0].shape
    nB, mB = B[0].shape
    AB = np.empty(shape=(num_states, nA+nB, mA+mB))
    for s in range(num_states): # s = up, down
        hor1 = np.hstack([A[s], small*np.random.randn(nA, mB)])
        hor2 = np.hstack([small*np.random.randn(nB, mA), B[s]])
        AB[s] = np.vstack([hor1, hor2])
    return AB


def analyze_MPS(A, axs, color):
    Au, Ad = A
    E = np.kron(Au, Au) + np.kron(Ad, Ad)
    evals = eigvals(E, check_finite=True)
    eval_absmax = np.max(np.abs(evals))

    evals = evals/eval_absmax
    #print('eval_absmax:', eval_absmax)
    r = 1 / np.sqrt(eval_absmax)
    Au, Ad = r * A

    # plotting spectrum of E:
    circle1 = plt.Circle((0., 0.), radius=1., color='green', fill=False)
    axs[1].plot(evals.real, evals.imag, color+'.')
    axs[1].add_patch(circle1)
    axs[1].axis('equal')
    axs[1].set_xlabel('real part')
    axs[1].set_ylabel('imag part')
    axs[1].title.set_text('Spectrum of the transfer matrix E')
    #print(evals)

    # plotting spectrum of Au:
    evals_a = eigvals(Au, check_finite=True)
    circle2 = plt.Circle((0., 0.), radius=1., color='green', fill=False)
    axs[2].plot(evals_a.real, evals_a.imag, color + '.')
    axs[2].add_patch(circle2)
    axs[2].axis('equal')
    axs[2].set_xlabel('real part')
    axs[2].set_ylabel('imag part')
    axs[2].title.set_text('Spectrum of Au')

    # # plotting spectrum of Ad:
    # evals_a = eigvals(Ad, check_finite=True)
    # circle3 = plt.Circle((0., 0.), radius=1., color='green', fill=False)
    # axs[3].plot(evals_a.real, evals_a.imag, color + '.')
    # axs[3].add_patch(circle3)
    # axs[3].axis('equal')
    # axs[3].set_xlabel('real part')
    # axs[3].set_ylabel('imag part')
    # axs[3].title.set_text('Spectrum of Ad')

    # plotting spectrum of kron(Au, Au):
    evals_a = eigvals(np.kron(Au, Au), check_finite=True)
    circle3 = plt.Circle((0., 0.), radius=1., color='green', fill=False)
    axs[3].plot(evals_a.real, evals_a.imag, color + '.')
    axs[3].add_patch(circle3)
    axs[3].axis('equal')
    axs[3].set_xlabel('real part')
    axs[3].set_ylabel('imag part')
    axs[3].title.set_text('Spectrum of kron(Au,Au)')

    # Transfer matrix:
    E = np.kron(Au, Au) + np.kron(Ad, Ad)

    # Right eigenvector of E:
    evals_k, evecs_k = eigs(E, k=8, which='LM')
    #ind = np.argmax(np.abs(evals_k))
    indices = np.argsort(np.abs(evals_k))
    ind = indices[-1]
    #print('---Right:\nevals_k:', evals_k, np.abs(evals_k))
    #print('ind:', ind, 'evals_k[ind]:', evals_k[ind])

    v_right = evecs_k[:,ind] # real
    #print('v_right', v_right)

    # Left eigenvector of E:
    evals_k, evecs_k = eigs(np.transpose(E), k=8, which='LM')
    #ind = np.argmax(np.abs(evals_k))
    indices = np.argsort(np.abs(evals_k))
    ind = indices[-1]
    #print('---Left:\nevals_k:', evals_k, np.abs(evals_k))
    #print('ind:', ind, 'evals_k[ind]:', evals_k[ind])
    v_left = np.conjugate(evecs_k[:,ind]) # real but different
    #print('v_left', v_left)
    #print(np.dot(v_left, v_right))

    # two-point functions:
    MSzM = 0.5* (np.kron(Au, Au) - np.kron(Ad, Ad))
    MSpM = np.kron(Au, Ad)
    MSmM = np.kron(Ad, Au)
    Er = np.eye(len(E))
    r_max = 50
    corr = np.zeros([r_max, 3])
    for l in range(1, r_max):
        Ozzr = np.matmul( np.matmul(MSzM, Er) , MSzM)
        Opmr = np.matmul(np.matmul(MSpM, Er), MSmM)
        Ompr = np.matmul(np.matmul(MSmM, Er), MSpM)

        Dr = np.matmul(np.matmul(E, Er), E)
        denominator = np.dot(np.dot(v_left, Dr), v_right)

        # zz:
        corr[l, 0] = np.real(np.dot( np.dot(v_left, Ozzr) , v_right) /denominator)
        # pm:
        corr[l, 1] = np.real(np.dot(np.dot(v_left, Opmr), v_right) / denominator)
        # mp:
        corr[l, 2] = np.real(np.dot(np.dot(v_left, Ompr), v_right) / denominator)

        Er = np.matmul(Er, E)
    #print('corr:', corr)
    #eigenvalue method with L and R doesnt seem to work, maybe due to gapless spectrum and E spectral singularity

    # # Lukyanov-Terras:
    # corrzz_exact = np.zeros(r_max)
    # slope_0p5 = np.zeros(r_max)
    # for l in range(1, r_max):
    #     corrzz_exact[l] = - 1/(2*np.pi**2 * l**2) + (-1)**l * 1/(2*np.pi**2 * l**2)
    #     slope_0p5[l] = 0.4 / l**0.5



    axs[4].clear()
    axs[4].title.set_text('correlations')
    axs[4].plot(corr[:,0], 'r.-', label='$<S^z_0S^z_r>$')
    axs[4].plot(corr[:, 1], 'b.-', label='$<S^+_0S^-_r>$')
    axs[4].legend(loc="upper right")
    axs[4].set_xlabel('r')
    axs[4].set_ylabel('corr(r)')

    axs[5].clear()
    axs[5].title.set_text('correlations')
    axs[5].set_xscale('log')
    axs[5].set_yscale('log')
    axs[5].plot(np.abs(corr[:, 0]), 'r.-', label='$<S^z_0S^z_r>$')
    #axs[5].plot(np.abs(corrzz_exact), 'ko', label='$<S^z_0S^z_r>_{exact}$')
    axs[5].plot(np.abs(corr[:, 1]), 'b.-', label='$<S^+_0S^-_r>$')
    #axs[5].plot(np.abs(slope_0p5), 'k-', label='0.4* r^{-0.5}')
    axs[5].legend(loc="lower left")
    axs[5].set_xlabel('r')
    axs[5].set_ylabel('corr(r)')

    # Heatmap of matrix elements
    #axs[6].clear()
    #axs[6].title.set_text('Heat map of Au')
    #im6 = axs[6].imshow(Au, cmap='cividis')

    axs[7].clear()
    axs[7].title.set_text('Heat map of Ad')
    im7 = axs[7].imshow(Ad, cmap='cividis')
    #plt.colorbar(im7, ax=axs[7])


class MPSCalibrator(Model):

    def __init__(self, train_data, k, epochs, loss_version, A0, learning_rate, axs, max_iter, grad_tolerance):
        super().__init__()

        self.samples = train_data['samples']
        self.sample_probs = train_data['sample_probs']
        self.decision_prob_matrix = train_data['decision_prob_matrix']

        print('datafile:', datafile,
              'samples.shape:', self.samples.shape, 'sample_probs.shape:', self.sample_probs.shape, '\n')
        self.k = k
        self.N = 2**k +2
        self.chi = A0[0].shape[0]
        self.en_inf_fo = -1/np.pi + np.pi**2/12/self.N**2 # first order
        self.epochs = epochs
        self.loss_version = loss_version
        self.counter = 0
        self.max_iter = max_iter
        self.grad_tolerance = grad_tolerance
        self.result_log = pd.DataFrame(columns=['step', 'runtime', 'chi', 'loss', 'min_loss', 'grad_norm', 'tolerance'])

        self.log2N = int(np.log2(self.N-2)) # fastest for N = 2^k +2 = 4, 6, 10, 18, ...
        print('Working with N=', 2**self.log2N+2, ' chi=', self.chi)

        self.A0 = self.eig_normalize(A0)
        self.A = tf.Variable(A0, trainable=True, dtype='float64')

        self.outfile_currentbest = 'MPS_calibrator_M_currentbest'

        self.min_loss = float('inf')

        self.axs = axs
        self.ax1 = axs[0]
        self.ax1.set_xlabel('runtime (sec)')
        self.ax1.set_ylabel('loss')
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax1.grid()

        self.ax1b = self.ax1.twinx()  # Create a second y-axis
        self.ax1b.set_ylabel('gradnorm', color='tab:green')
        self.ax1b.set_yscale('log')
        self.ax1b.tick_params(axis='y', labelcolor='tab:green')

    def eig_normalize(self, A):
        # For numerical stability we normalize the initial random guess

        Au = A[0]
        Ad = A[1]
        E = np.kron(Au, Au) + np.kron(Ad, Ad)
        print('E:', E.shape)

        # eig norm:
        if 1:
            evals_k, _ = eigs(E, k=8, which='LM')
            # print('eig_k:\n', evals_k)
            # print('|eig_k|:\n', np.abs(evals_k))
            eval_absmax = np.max(np.abs(evals_k))
            r = 1 / np.sqrt(eval_absmax)
            # print('r:', r)

        Au = r * Au
        Ad = r * Ad
        A_normalized = np.array([Au, Ad])

        return A_normalized

    def int2bits(self, n, num_bits):
        # n: integer to convert to binary array
        # num_bits: min length of the resulting numpy array
        return np.array([int(i) for i in bin(n)[2:].zfill(num_bits)])

    def print_result_log(self, value, gradnorm):
        toc = datetime.datetime.now()

        self.ax1.plot((toc - tic).total_seconds(), value, self.color + '.')
        self.ax1.title.set_text('min_loss=' + str(round(self.min_loss, 6)) +
                                '\nN=' + str(self.N) + ' chi=' + str(self.chi) )

        self.ax1b.plot((toc - tic).total_seconds(), gradnorm, 'tab:green', marker='.')

        row = {'step': self.counter, 'runtime': round((toc - tic).total_seconds(), 2),
               'chi': self.chi, 'loss': round(value.numpy(), 8), 'min_loss': round(self.min_loss, 10),
               'grad_norm': round(gradnorm.numpy(), 8), 'tolerance': self.grad_tolerance}
        print(row)
        new_row = pd.DataFrame([row])
        self.result_log = pd.concat([self.result_log, new_row], axis=0, ignore_index=True)
        plt.pause(.1)

    def lossfunction(self, A):
        if self.loss_version == 'prob_mse':
            # sample_probs based MSE loss (on MC sample)
            return self.lossfunction_prob_mse(A)
        elif self.loss_version == 'decision_prob_matrix':
            # decision_prob_matrix based MSE loss (on theoretical or empirical conditional decision prob matrix)
            return self.lossfunction_decision_prob_matrix(A)
        elif self.loss_version == 'likelihood':
            # negative loglikelihood loss (on Monte Carlo sample)
            return self.lossfunction_likelihood(A)

    def record_currentbest(self, loss, A):
        self.min_loss = loss

        np.savez(self.outfile_currentbest, M=A)
        print('Current best result dumped:', self.outfile_currentbest)

        # To compare with known target M_target on the fly:
        if 1:
            M_target_file = 'M_tensor'  # 'M_tensor_eps01' 'M_tensor_seed104' 'M_tensor'
            npzfile = np.load(M_target_file + '.npz')
            M_target = npzfile['M']
            print('M_target:\n', M_target)

            npzfile = np.load(self.outfile_currentbest + '.npz')
            M_model = npzfile['M']
            print('M_model:\n', M_model)

            comparator = mps_comparator.MPS_comparator(M_target, M_model, self.samples)
            pus_1, pus_2 = comparator.compare()
            comparator.visualize(self.axs[6], pus_1, pus_2, noise=0.005)

        if 1:
            analyze_MPS(A.numpy(), self.axs, color='k')

    def lossfunction_decision_prob_matrix(self, A):
        # Given A, calculate the MSE based on the target and model decision prob matrices

        self.counter += 1

        # Create EN for an almost infinite long chain of unstructured sites: ##########################################

        A = tf.reshape(A, [2, self.chi, self.chi])
        Au = A[0]
        Ad = A[1]

        AAu = tf.experimental.numpy.kron(Au, Au)
        AAd = tf.experimental.numpy.kron(Ad, Ad)

        E = AAu + AAd
        E2 = tf.matmul(E, E)

        E2k = E2
        for k in range(self.log2N - 1):
            E2k = tf.matmul(E2k, E2k)
            normalizer = 1 / tf.linalg.trace(E2k)
            E2k = normalizer * E2k  # safe and enough to renormalize here, as normalizer will be canceled by the same factor in the denominator term

        EN = tf.matmul(E2k, E2)  # N = 2^k +2

        # Iterate through all possible configs: ########################################################################

        N_spins = self.samples.shape[1]
        K_paths = 2 ** N_spins

        config_bits = np.empty([K_paths, N_spins])
        config_bits.fill(np.nan)

        for k in range(K_paths):
            config_bits[k, :] = self.int2bits(k, N_spins)

        mse = 0

        for k in range(K_paths):

            String = EN

            current_int_config = 0

            for n in range(N_spins):
                String_AAu = tf.matmul(String, AAu)
                String_AAd = tf.matmul(String, AAd)
                tr_u = tf.linalg.trace(String_AAu)
                tr_d = tf.linalg.trace(String_AAd)
                pu =  tr_u / (tr_u + tr_d)

                pu_exact = self.decision_prob_matrix[n, current_int_config, 1]               #self.sample_probs[k, n]
                mse += tf.square(pu - pu_exact) #* 2**n
                #print('k, n:', k, n, 'pu:', pu, 'pu_exact:', pu_exact)

                if config_bits[k, n] == 1:  # spin-up sampled
                    current_int_config = 2 * current_int_config + 1
                    String = tf.matmul(String, AAu)
                    norm = tf.linalg.trace(String)
                    String = String / norm
                else:  # spin-down sampled
                    current_int_config = 2 * current_int_config
                    String = tf.matmul(String, AAd)
                    norm = tf.linalg.trace(String)
                    String = String / norm

        mse = mse / (K_paths * N_spins)

        if mse.numpy() < self.min_loss:
            self.record_currentbest(mse.numpy(), A)

        return mse  # mse to minimize


    def lossfunction_prob_mse(self, A):
        #print('Running targetfunction', A.shape)

        self.counter += 1

        # Given A, calculate the MSE of the sample based on predicted and exact sampling probabilities.
        # Only works if exact (or approximate) sampling probabilities are known:

        # Create EN for an almost infinite long chain of unstructured sites: ##########################################

        A = tf.reshape(A, [2, self.chi, self.chi])
        Au = A[0]
        Ad = A[1]

        AAu = tf.experimental.numpy.kron(Au, Au)
        AAd = tf.experimental.numpy.kron(Ad, Ad)

        E = AAu + AAd
        E2 = tf.matmul(E, E)

        E2k = E2
        for k in range(self.log2N - 1):
            E2k = tf.matmul(E2k, E2k)
            normalizer = 1 / tf.linalg.trace(E2k)
            E2k = normalizer * E2k  # safe and enough to renormalize here, as normalizer will be canceled by the same factor in the denominator term

        EN = tf.matmul(E2k, E2)  # N = 2^k +2

        # Iterate through samples: ####################################################################################

        K_paths, N_spins = self.samples.shape

        mse = 0

        for k in range(K_paths):
            if (0 < k < 1000 and k % 1000 == 0) or (k >= 1000 and k % 1000 == 0):
                print('\nPath:', k, K_paths)

            String = EN

            for n in range(N_spins):
                String_AAu = tf.matmul(String, AAu)
                String_AAd = tf.matmul(String, AAd)
                tr_u = tf.linalg.trace(String_AAu)
                tr_d = tf.linalg.trace(String_AAd)
                pu =  tr_u / (tr_u + tr_d)
                #pd = tr_d / (tr_u + tr_d)

                pu_exact = self.sample_probs[k, n]
                mse += tf.square(pu - pu_exact)
                if (0 < k and k % 1000 == 0):
                    print('k, n:', k, n, mse.numpy())

                if self.samples[k, n] == 1:  # spin-up sampled
                    String = tf.matmul(String, AAu)
                    norm = tf.linalg.trace(String)
                    String = String / norm
                else:  # spin-down sampled
                    String = tf.matmul(String, AAd)
                    norm = tf.linalg.trace(String)
                    String = String / norm

        mse = mse / (K_paths * N_spins)

        if mse.numpy() < self.min_loss:
            self.record_currentbest(mse.numpy(), A)

        return mse  # mse to minimize


    #@tf.function
    def lossfunction_likelihood(self, A):

        self.counter += 1

        # Create the transfer matrix E and other tensors:  #############################################################

        A = tf.reshape(A, [2, self.chi, self.chi])
        Au = A[0]
        Ad = A[1]
        AA = [tf.experimental.numpy.kron(Au, Au), tf.experimental.numpy.kron(Ad, Ad)]
        AAu = tf.experimental.numpy.kron(Au, Au)
        AAd = tf.experimental.numpy.kron(Ad, Ad)

        #E = AAu + AAd
        E = AA[0] + AA[1]
        E2 = tf.matmul(E, E)

        E2k = E2
        for k in range(self.log2N - 1):
            E2k = tf.matmul(E2k, E2k)
            normalizer = 1 / tf.linalg.trace(E2k)
            E2k = normalizer * E2k  # safe and enough to renormalize here, as normalizer will be canceled by the same factor in the denominator term

        EN = tf.matmul(E2k, E2)  # N = 2^k +2

        # Iterate through samples: #####################################################################################

        K_paths, N_spins = self.samples.shape

        loglikelihood = 0

        for k in range(K_paths):
            if (0 < k < 1000 and k % 1000 == 0) or (k >= 1000 and k % 1000 == 0):
                print('\nPath:', k, K_paths)

            String = EN

            for n in range(N_spins):
                String_AAu = tf.matmul(String, AAu)
                String_AAd = tf.matmul(String, AAd)
                tr_u = tf.linalg.trace(String_AAu)
                tr_d = tf.linalg.trace(String_AAd)
                pu =  tr_u / (tr_u + tr_d)
                pd = tr_d / (tr_u + tr_d)
                if (0 < k and k % 2000 == 0):
                    print('n:', n, 'p:', pu.numpy(), pd.numpy(), tf.math.log(pu).numpy(), tf.math.log(pd).numpy())

                if self.samples[k, n] == 1:  # spin-up sampled
                    loglikelihood += tf.math.log(pu)
                    #if (k % 2000 == 0):
                    #    print('s: 1', loglikelihood.numpy())
                    String = tf.matmul(String, AAu)
                    norm = tf.linalg.trace(String)
                    String = String / norm
                else:  # spin-down sampled
                    loglikelihood += tf.math.log(pd)
                    #if (k % 2000 == 0):
                    #    print('s: 0', loglikelihood.numpy())
                    String = tf.matmul(String, AAd)
                    norm = tf.linalg.trace(String)
                    String = String / norm
                #print('loglikelihood', loglikelihood)

        loglikelihood = loglikelihood / (K_paths * N_spins)

        if -loglikelihood.numpy() < self.min_loss:
            self.record_currentbest(-loglikelihood.numpy(), A)

        return -loglikelihood  # negative likelihood to minimize = likelihood to maximize


    # The objective function and the gradient.
    #@tf.function
    def lossfunction_and_gradient(self, A):
        value, grad = tfp.math.value_and_gradient(self.lossfunction, A)
        if 0:
            print('Value:', value)
            print('Gradient:', grad)
            print('Grad norm:', tf.norm(grad, axis=-1).numpy(), 'Tolerance:', self.grad_tolerance)
        if (self.counter < 1000 and self.counter % 1 == 0) or (self.counter >= 1000 and self.counter % 10 == 0):
            self.print_result_log(value, tf.norm(grad, axis=-1))

        return value, grad


    def multi_valuator(self, A0_np, A1_np, lambdas):
        # Calculate the loss along a one-dimensional path interpolating between two M tensors. No calibration.

        sh = np.shape(A0_np)
        values = np.zeros(len(lambdas))
        grad_norms = np.zeros(len(lambdas))

        for i, lam in enumerate(lambdas):
            if i % 100 == 0:
                print('i', i)
            A_lam = (1- lam) * A0_np + lam * A1_np

            A = tf.convert_to_tensor(np.reshape(A_lam, sh[0] * sh[1] * sh[2]))

            value, grad = tfp.math.value_and_gradient(self.lossfunction, A)
            values[i] = value.numpy()
            grad_norms[i] = tf.norm(grad, axis=-1).numpy()
            if 1:
                print('Value:', value.numpy())
                #print('Gradient:', grad)
                print('Grad norm:', tf.norm(grad, axis=-1).numpy())

        plt.figure(num=401)
        plt.plot(lambdas, values, '.')
        plt.figure(num=402)
        plt.plot(lambdas, grad_norms, '.')

        return

    #@tf.function
    def run_calibrator(self, color, tic):

        self.color = color
        sh = np.shape(self.A0)
        start = tf.convert_to_tensor(np.reshape(self.A0, sh[0] * sh[1] * sh[2]))  # Starting point for the search.
        if 0:
            print('start:\n', start)
            initial_value, initial_grad = self.lossfunction_and_gradient(start)
            print('Initial value:', initial_value)
            print('Initial gradient:', initial_grad, 'norm:', tf.norm(initial_grad).numpy())

        optim_results = tfp.optimizer.bfgs_minimize(
            self.lossfunction_and_gradient, initial_position=start, parallel_iterations=1,
            tolerance=self.grad_tolerance, max_iterations=self.max_iter)

        print("num_objective_evaluations: %d" % optim_results.num_objective_evaluations)
        print('converged: %d' % optim_results.converged)
        print('failed: %d' % optim_results.failed)
        print('num_iterations: %d' % optim_results.num_iterations)
        print('objective_gradient:', optim_results.objective_gradient.numpy())
        #print('position:', optim_results.position)
        # print('optim_results:', optim_results)

        self.A = tf.reshape(optim_results.position, [2, self.chi, self.chi])



if __name__ == '__main__':

    print('Input parameters  #########################################################################################')

    # loss function version to use:
    if 0:
        loss_version = 'prob_mse'
    elif 1:
        loss_version = 'decision_prob_matrix'
    else:
        loss_version = 'likelihood'

    # Calibration parameters:
    chi = 2  # starting value 4
    grad_tolerance = 1.e-9 #-6
    learning_rate = 0.01  # 0.001 default - no impact for BFGS

    # Samples to calibrate to:
    datafile = 'samples.npz'

    # Write result M tensor to:
    outfile = 'MPS_calibrator_M_final'




    print('Seeting up TF environment  ################################################################################')

    train_data = np.load(datafile)

    k = 10  # -> N = 2 + 2**k # only works with these N values
    N = 2 + 2 ** k
    colors = ['r', 'g', 'b', 'k']


    result_log = pd.DataFrame(columns=['step', 'runtime', 'chi', 'loss', 'min_loss', 'grad_norm', 'tolerance'])
    tic = datetime.datetime.now()
    fig1 = plt.figure(1, figsize=(12, 6))
    axs = []
    for i in range(8):
        axs.append(fig1.add_subplot(2, 4, i + 1))


    print('Initial value for M  ######################################################################################')
    np.random.seed(102)
    if 1:
        np.random.seed(113)
        M_target_file = 'M_tensor'
        A0 = np.random.randn(2, chi, chi)  # 2 => up, down
    elif 0:
        A0 = [np.ones([chi, chi]), np.ones([chi, chi])]  # gets stuck immediately in a local minimum
    elif 1:
        M_target_file = 'M_tensor_seed104'  # 'M_tensor_eps01' 'M_tensor_seed104' 'M_tensor'
        npzfile = np.load(M_target_file + '.npz')
        A0 = npzfile['M'] + 1.2 * np.random.randn(2, chi, chi) #should be good??
        print('A0', A0)
    else:
        M_target_file = 'MPS_calibrator_M_currentbest'  # 'M_tensor_eps01'
        npzfile = np.load(M_target_file + '.npz')
        A0 = npzfile['M'] + 0.0 * np.random.randn(2, chi, chi)

    if 1:
        calibrator = MPSCalibrator(train_data, k, None, loss_version, A0, learning_rate, axs, None,
                                   grad_tolerance)
        lambdas = np.linspace(0,1,1000)

        M_target_file = 'M_tensor_seed104'  # 'M_tensor_eps01' 'M_tensor_seed104' 'M_tensor'
        npzfile = np.load(M_target_file + '.npz')
        A1 = npzfile['M']

        calibrator.multi_valuator(A0, A1, lambdas)
        plt.show()
        exit()

    epochs_list = 8 * [300]  # [2000, 1500, 1000, 800, 600, 400, 400, 400]
    for i_chi in range(1):

        max_iter = 200
        print('learning_rate:', learning_rate, 'max_iter:', max_iter)

        calibrator = MPSCalibrator(train_data, k, epochs_list[i_chi], loss_version, A0, learning_rate, axs, max_iter, grad_tolerance)
        calibrator.run_calibrator(color=colors[i_chi % len(colors)], tic=tic)

        result_log = pd.concat([result_log, calibrator.result_log], axis=0, ignore_index=True)
        print('result_log:\n', result_log)

        A_opt = calibrator.A.numpy()
        analyze_MPS(A_opt, axs, color=colors[i_chi % len(colors)])

        A_zero = np.zeros([2, 4, 4])  # always adding +4 extra dims [2, 4, 4]
        A0 = augment_combine(A_opt, A_zero, small=0.001)  # small=0.00001  0.001

    outfile = 'MPS_calibrator_M_dump'
    np.savez(outfile, M=A_opt)

    plt.show()
