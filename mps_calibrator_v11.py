# Calibrating M from empirical time series - Gradient method

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
    print('eval_absmax:', eval_absmax)
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
    print('---Right:\nevals_k:', evals_k, np.abs(evals_k))
    print('ind:', ind, 'evals_k[ind]:', evals_k[ind])

    v_right = evecs_k[:,ind] # real
    #print('v_right', v_right)

    # Left eigenvector of E:
    evals_k, evecs_k = eigs(np.transpose(E), k=8, which='LM')
    #ind = np.argmax(np.abs(evals_k))
    indices = np.argsort(np.abs(evals_k))
    ind = indices[-1]
    print('---Left:\nevals_k:', evals_k, np.abs(evals_k))
    print('ind:', ind, 'evals_k[ind]:', evals_k[ind])
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

    # Lukyanov-Terras:
    corrzz_exact = np.zeros(r_max)
    slope_0p5 = np.zeros(r_max)
    for l in range(1, r_max):
        corrzz_exact[l] = - 1/(2*np.pi**2 * l**2) + (-1)**l * 1/(2*np.pi**2 * l**2)
        slope_0p5[l] = 0.4 / l**0.5



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
    axs[5].plot(np.abs(corrzz_exact), 'ko', label='$<S^z_0S^z_r>_{exact}$')
    axs[5].plot(np.abs(corr[:, 1]), 'b.-', label='$<S^+_0S^-_r>$')
    axs[5].plot(np.abs(slope_0p5), 'k-', label='0.4* r^{-0.5}')
    axs[5].legend(loc="lower left")
    axs[5].set_xlabel('r')
    axs[5].set_ylabel('corr(r)')

    # Heatmap of matrix elements
    axs[6].clear()
    axs[6].title.set_text('Heat map of Au')
    im6 = axs[6].imshow(Au, cmap='cividis')
    #axs[6].colorbar(im6, ax=axs[6])

    axs[7].clear()
    axs[7].title.set_text('Heat map of Ad')
    im7 = axs[7].imshow(Ad, cmap='cividis')
    #plt.colorbar(im7, ax=axs[7])


class MPSCalibrator(Model):

    def __init__(self, k, epochs, A0, learning_rate, axs, max_iter):
        super().__init__()

        self.k = k
        self.N = 2**k +2
        self.chi = A0[0].shape[0]
        self.en_inf_fo = -1/np.pi + np.pi**2/12/self.N**2 # first order
        self.epochs = epochs
        self.counter = 0
        self.max_iter = max_iter
        #self.result_log = result_log
        self.result_log = pd.DataFrame(columns=['step', 'runtime', 'chi', 'max_loglikelihood', 'delta_energy'])

        self.log2N = int(np.log2(self.N-2)) # fastest for N = 2^k +2 = 4, 6, 10, 18, ...
        print('Working with N=', 2**self.log2N+2, ' chi=', self.chi)

        self.A0 = self.eig_normalize(A0)
        self.A = tf.Variable(A0, trainable=True, dtype='float64')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)  # RMSprop Adam learning_rate

        self.max_loglikelihood = -float('inf')

        self.ax1 = axs[0]
        self.ax1.set_xlabel('runtime (sec)')
        self.ax1.set_ylabel('bond energy')
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax1.grid()

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
            print('r:', r)

        Au = r * Au
        Ad = r * Ad
        A_normalized = np.array([Au, Ad])

        return A_normalized

    # def bond_energy(self):
    #     Au = self.A[0]
    #     Ad = self.A[1]
    #     E = tf.experimental.numpy.kron(Au, Au) + tf.experimental.numpy.kron(Ad, Ad)
    #     E2 = tf.matmul(E, E)
    #
    #     E2k = E2
    #     for k in range(self.log2N - 1):
    #         E2k = tf.matmul(E2k, E2k)
    #     EN = tf.matmul(E2k, E2) # N = 2^k +2
    #     trEN = tf.linalg.trace(EN)
    #
    #     #AuAu = tf.matmul(Au, Au)
    #     AuAd = tf.matmul(Au, Ad)
    #     AdAu = tf.matmul(Ad, Au)
    #     #AdAd = tf.matmul(Ad, Ad)
    #
    #     #XX model:
    #     G2 = 0.5 * (tf.experimental.numpy.kron(AuAd, AdAu) + tf.experimental.numpy.kron(AdAu, AuAd))
    #     GN = tf.matmul(E2k, G2) # N = 2^k +2
    #     trGN = tf.linalg.trace(GN)
    #
    #     energy = trGN / trEN
    #     #tf.print('trGN, trEN:', trGN, trEN)
    #
    #     nodes = [trGN, trEN] # nodes = [trGN, trEN, E, EN]
    #     return energy, nodes


    # # Use tf.GradientTape to train the model:
    # @tf.function
    # def my_train_step(self):
    #     with tf.GradientTape() as tape:
    #         bond_energy, nodes = self.bond_energy()
    #         gradients = tape.gradient(bond_energy, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #
    #     return bond_energy, nodes


    # function:
    def targetfunction_old(self, A):
        self.counter += 1

        A = tf.reshape(A, [2, self.chi, self.chi])
        Au = A[0]
        Ad = A[1]
        E = tf.experimental.numpy.kron(Au, Au) + tf.experimental.numpy.kron(Ad, Ad)
        E2 = tf.matmul(E, E)
        #print('E', E)

        E2k = E2
        for k in range(self.log2N - 1):
            E2k = tf.matmul(E2k, E2k)
            normalizer = 1 / tf.linalg.trace(E2k)
            E2k = normalizer * E2k  # safe and enough to renormalize here, as normalizer will be canceled by the same factor in the denominator term

        EN = tf.matmul(E2k, E2)  # N = 2^k +2
        trEN = tf.linalg.trace(EN)

        # AuAu = tf.matmul(Au, Au)
        AuAd = tf.matmul(Au, Ad)
        AdAu = tf.matmul(Ad, Au)
        # AdAd = tf.matmul(Ad, Ad)

        # XX model:
        G2 = 0.5 * (tf.experimental.numpy.kron(AuAd, AdAu) + tf.experimental.numpy.kron(AdAu, AuAd))
        GN = tf.matmul(E2k, G2)  # N = 2^k +2
        trGN = tf.linalg.trace(GN)

        energy = trGN / trEN
        #print('energy', energy.numpy(), trGN.numpy(), trEN.numpy())
        if energy.numpy() < self.lowest_energy:
            self.lowest_energy = energy.numpy()

        if (self.counter < 100 and self.counter % 10 == 0) or (self.counter >= 100 and self.counter % 50 == 0):
            toc = datetime.datetime.now()
            #tf.print(self.counter, 'energy, trGN, trEN:', energy, trGN, trEN, (toc-tic).total_seconds())
            self.ax1.plot((toc - tic).total_seconds(), energy - self.en_inf_fo, self.color + '.')  # inf: 0.31831
            self.ax1.title.set_text('lowest_energy:' + str(self.lowest_energy) +
                               '\nN=' + str(self.N) + ' chi_max='+str(self.chi)+' epochs=' + str(self.epochs))

            # columns=['step', 'runtime', 'chi', 'bond_energy']
            row = {'step': self.counter, 'runtime': round((toc - tic).total_seconds(),2),
                   'chi': self.chi, 'lowest_energy': round(self.lowest_energy, 10),
                   'delta_energy': round(energy.numpy() - self.en_inf_fo, 10)}
            print(row)
            new_row = pd.DataFrame([row])
            self.result_log = pd.concat([self.result_log, new_row], axis=0, ignore_index=True)
            #print('result_log:', self.result_log)
            plt.pause(.1)

        #nodes = [trGN, trEN]  # nodes = [trGN, trEN, E, EN]

        return energy


    def print_result_log(self, target_value):
        toc = datetime.datetime.now()
        # tf.print(self.counter, 'energy, trGN, trEN:', energy, trGN, trEN, (toc-tic).total_seconds())
        self.ax1.plot((toc - tic).total_seconds(), target_value - self.en_inf_fo, self.color + '.')  # inf: 0.31831
        self.ax1.title.set_text('max_loglikelihood' + str(self.max_loglikelihood) +
                                '\nN=' + str(self.N) + ' chi_max=' + str(self.chi) + ' epochs=' + str(self.epochs))

        # columns=['step', 'runtime', 'chi', 'bond_energy']
        row = {'step': self.counter, 'runtime': round((toc - tic).total_seconds(), 2),
               'chi': self.chi, 'max_loglikelihood': round(self.max_loglikelihood, 10),
               'delta_energy': round(target_value.numpy() - self.en_inf_fo, 10)}
        print(row)
        new_row = pd.DataFrame([row])
        self.result_log = pd.concat([self.result_log, new_row], axis=0, ignore_index=True)
        # print('result_log:', self.result_log)
        plt.pause(.1)


    def targetfunction(self, A):
        print('Running targetfunction')
        # using samples from outside scope

        self.counter += 1

        # given A, calculate the likelihood of the sample:

        # Create EN for an almost infinite long chain of unstructured sites: ##########################################

        A = tf.reshape(A, [2, self.chi, self.chi])
        Au = A[0]
        Ad = A[1]
        AA = [tf.experimental.numpy.kron(Au, Au), tf.experimental.numpy.kron(Ad, Ad)]
        AAu = tf.experimental.numpy.kron(Au, Au)
        AAd = tf.experimental.numpy.kron(Ad, Ad)

        #E = AAu + AAd
        E = AA[0] + AA[1]
        E2 = tf.matmul(E, E)
        #print('E', E)

        E2k = E2
        for k in range(self.log2N - 1):
            E2k = tf.matmul(E2k, E2k)
            normalizer = 1 / tf.linalg.trace(E2k)
            E2k = normalizer * E2k  # safe and enough to renormalize here, as normalizer will be canceled by the same factor in the denominator term

        EN = tf.matmul(E2k, E2)  # N = 2^k +2
        #trEN = tf.linalg.trace(EN)

        # Iterate through samples: ####################################################################################

        K_paths, N_spins = samples.shape

        for k in range(K_paths):
            #if (k < 1000 and k % 100 == 0) or (k >= 1000 and k % 200 == 0):
            print('Path:', k)

            String = EN
            loglikelihood = 0
            for n in range(N_spins):
                String_AAu = np.matmul(String, AAu)
                String_AAd = np.matmul(String, AAd)
                tr_u = tf.linalg.trace(String_AAu)
                tr_d = tf.linalg.trace(String_AAd)
                #print('hah:', k, 'n:', n, String_AuAuR, String_AdAdR)
                pu =  tr_u / (tr_u + tr_d)
                pd = tr_d / (tr_u + tr_d)

                if samples[k, n] == 1:  # spin-up sampled
                    loglikelihood += tf.math.log(pu)
                    String = np.matmul(String, AAu)
                else:  # spin-down sampled
                    loglikelihood += tf.math.log(pd)
                    String = np.matmul(String, AAd)
                print('loglikelihood', loglikelihood)


        if loglikelihood.numpy() > self.max_loglikelihood:
            self.max_loglikelihood = loglikelihood.numpy()

        if (self.counter < 100 and self.counter % 1 == 0) or (self.counter >= 100 and self.counter % 50 == 0):
            self.print_result_log(loglikelihood)

        return -loglikelihood  # negative likelihood to minimize = likelihood to maximize


    # The objective function and the gradient.
    #@tf.function
    def targetfunction_and_gradient(self, A):
        print('haha1:', tfp.math.value_and_gradient(self.targetfunction, A))
        return tfp.math.value_and_gradient(self.targetfunction, A)

    #@tf.function
    def run_calibrator(self, color, tic):

        self.color = color
        sh = np.shape(self.A0)
        #start = tf.constant(np.reshape(self.A0, sh[0]*sh[1]*sh[2]))  # Starting point for the search.
        start = tf.convert_to_tensor(np.reshape(self.A0, sh[0] * sh[1] * sh[2]))  # Starting point for the search.
        print('start:\n', start)
        optim_results = tfp.optimizer.bfgs_minimize(
            self.targetfunction_and_gradient, initial_position=start, parallel_iterations=1,
            tolerance=1e-8, max_iterations=self.max_iter)

        print("num_objective_evaluations: %d" % optim_results.num_objective_evaluations)
        print('converged: %d' % optim_results.converged)
        print('failed: %d' % optim_results.failed)
        print('num_iterations: %d' % optim_results.num_iterations)
        print('objective_gradient:', optim_results.objective_gradient.numpy())
        #print('position:', optim_results.position)
        # print('optim_results:', optim_results)

        self.A = tf.reshape(optim_results.position, [2, self.chi, self.chi])
        #tf.print('Best A:', self.A)




if __name__ == '__main__':

    # load samples
    samples = np.load('samples.npy')

    k = 4  # -> N = 2 + 2**k # only works with these N values
    N = 2 + 2 ** k
    colors = ['r', 'g', 'b', 'k']
    np.random.seed(100)

    result_log = pd.DataFrame(columns=['step', 'runtime', 'chi', 'max_loglikelihood', 'delta_energy'])
    tic = datetime.datetime.now()
    fig1 = plt.figure(1, figsize=(15, 9))
    axs = []
    for i in range(8):
        axs.append(fig1.add_subplot(2, 4, i + 1))

    chi = 2  # starting value 4
    A0 = np.random.randn(2, chi, chi)  # 2 => up, down

    epochs_list = 8 * [300]  # [2000, 1500, 1000, 800, 600, 400, 400, 400]
    for i_chi in range(1):
        learning_rate = 0.001  # *(i_chi+1)**2 #0.001 default
        print('learning_rate', learning_rate)
        if i_chi < 7:
            max_iter = 1000
        else:
            max_iter = 400

        calibrator = MPSCalibrator(k, epochs_list[i_chi], A0, learning_rate, axs, max_iter)
        calibrator.run_calibrator(color=colors[i_chi % len(colors)], tic=tic)

        result_log = pd.concat([result_log, calibrator.result_log], axis=0, ignore_index=True)
        print('result_log:\n', result_log)

        A_opt = calibrator.A.numpy()
        analyze_MPS(A_opt, axs, color=colors[i_chi % len(colors)])

        A_zero = np.zeros([2, 4, 4])  # always adding +4 extra dims [2, 4, 4]
        A0 = augment_combine(A_opt, A_zero, small=0.001)  # small=0.00001  0.001

    outfile = 'MPS_calibrator_M_dump'
    np.savez(outfile, M=A_opt)