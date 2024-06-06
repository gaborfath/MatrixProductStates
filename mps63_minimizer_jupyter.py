# Adding new degrees of freedom as empty rows/columns according to a +4 scheme.
# Using tensorflow_probability minimizers: BFGS

import tensorflow as tf
from tensorflow.python.keras import Model
from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import datetime
from scipy.linalg import eigh as largest_eigh
# from scipy.linalg import eigvals as all_eigvals
from scipy.sparse.linalg import eigs
from scipy.linalg import eigvals
from scipy.linalg import eig
import tensorflow_probability as tfp
import pandas as pd


def eig_normalize(A):
    # For numerical stability we normalize the initial random guess

    Au = A[0]
    Ad = A[1]
    E = np.kron(Au, Au) + np.kron(Ad, Ad)
    print('E:', E.shape)

    # eig norm:
    if 1:
        evals_k, _ = eigs(E, k=8, which='LM')
        #print('eig_k:\n', evals_k)
        #print('|eig_k|:\n', np.abs(evals_k))
        eval_absmax = np.max(np.abs(evals_k))
        r = 1 / np.sqrt(eval_absmax)
        print('r:', r)

    Au = r * Au
    Ad = r * Ad
    A_normalized = np.array([Au, Ad])

    return A_normalized

def augment(A, k): # addition does not learn
    Au, Ad = A
    small = 0.0001
    n, m = Au.shape
    Au_aug = np.vstack([np.hstack([Au, np.zeros([n, k])]), np.hstack([np.zeros([k, m]), small*np.ones([k, k])])])
    n, m = Ad.shape
    Ad_aug = np.vstack([np.hstack([Ad, np.zeros([n, k])]), np.hstack([np.zeros([k, m]), small*np.ones([k, k])])])

    return np.array([Au_aug, Ad_aug])

def augment_double(A, small):
    Au, Ad = A
    n, m = Au.shape
    hor1 = np.hstack([Au, small*np.random.randn(n, m)])
    hor2 = np.hstack([small*np.random.randn(n, m), Au])
    Au_aug = np.vstack([hor1, hor2])

    n, m = Ad.shape
    hor1 = np.hstack([Ad, small * np.random.randn(n, m)])
    hor2 = np.hstack([small * np.random.randn(n, m), Ad])
    Ad_aug = np.vstack([hor1, hor2])

    return np.array([Au_aug, Ad_aug])

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

def augment_offdiag(A, k):
    Au, Ad = A
    small = 0.001
    n, m = Au.shape
    Au_aug = np.vstack([np.hstack([Au, small*np.random.randn(n, k)]), np.zeros([k, m + k])])
    n, m = Ad.shape
    Ad_aug = np.vstack([np.hstack([Ad, small*np.random.randn(n, k)]), np.zeros([k, m + k])])

    return np.array([Au_aug, Ad_aug])

def augment_old(A, k): # addition does not learn
    Au, Ad = A
    n, m = Au.shape
    Au_aug = np.vstack([np.hstack([Au, np.zeros([n, k])]), np.zeros([k, m + k])])
    n, m = Ad.shape
    Ad_aug = np.vstack([np.hstack([Ad, np.zeros([n, k])]), np.zeros([k, m + k])])

    return np.array([Au_aug, Ad_aug])

def analyze_MPS(A, axs, color):
    Au, Ad = A

    # boost
    Au = np.kron(Au, A_boost[0].numpy())
    Ad = np.kron(Ad, A_boost[1].numpy())

    #


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
    
class MPS_Sampler():
    def __init__(self, M_tensor):
        self.M = M_tensor
        self.Au = self.M[0]
        self.Ad = self.M[1]
        self.chi = self.Au.shape[0]

        self.AuAu = np.kron(self.Au, self.Au)
        self.AdAd = np.kron(self.Ad, self.Ad)
        self.E = self.AuAu + self.AdAd

        if 0:
            # Right eigenvector of E:
            evals_k, evecs_k = eigs(self.E, k=8, which='LM')
            # ind = np.argmax(np.abs(evals_k))
            indices = np.argsort(np.abs(evals_k))
            ind = indices[-1]
            print('---Right:\nevals_k:', evals_k, np.abs(evals_k))
            print('ind:', ind, 'evals_k[ind]:', evals_k[ind])

            self.lam = evals_k[ind]
            self.R = evecs_k[:, ind]  # real
            print('lam:', self.lam)
            print('R:', self.R)

            # Left eigenvector of E:
            evals_k, evecs_k = eigs(np.transpose(self.E), k=8, which='LM')
            # ind = np.argmax(np.abs(evals_k))
            indices = np.argsort(np.abs(evals_k))
            ind = indices[-1]
            print('---Left:\nevals_k:', evals_k, np.abs(evals_k))
            print('ind:', ind, 'evals_k[ind]:', evals_k[ind])
            self.L = np.conjugate(evecs_k[:, ind])  # real but different
            print('L:', self.L)

        else:
            # Alternative:
            w, vl, vr = eig(self.E, left=True, right=True)
            indices = np.argsort(np.abs(w))
            self.lam = np.real(w[indices[-1]]) # is it justified?
            self.R = vr[:, indices[-1]]
            self.L = vl[:, indices[-1]]
            self.LR = np.dot(self.L,self.R)
            print('self.LR', self.LR)

            print('leading eig:\n', self.lam, self.L, self.R)



    def __call__(self, N_spins, K_paths):
        samples = np.empty((K_paths, N_spins))


        for k in range(K_paths):
            if (k < 1000 and k % 100 == 0) or (k >= 1000 and k % 100 == 0):
                print(k, '-----------\n')
            String = np.matmul(self.L, np.eye(self.chi**2)) # product of fixed spin operators
            AuAuR = np.matmul(self.AuAu, self.R) / self.LR
            p_cond = 1
            sample = np.empty(N_spins)
            lamb = np.array([self.lam ** (n+1) for n in range(N_spins)])
            for n in range(N_spins):
                String_AuAu = np.matmul(String, AuAuR)
                #String_AdAd = np.matmul(String, self.AdAd)
                pu_cond = String_AuAu / lamb[n]
                #pd_cond = np.matmul(np.matmul(self.L, String_AdAd), self.R) / self.lam ** (n + 1) / self.LR
                pu = pu_cond / p_cond
                #print('pu:', pu, 'p_cond:', p_cond, 'pu_cond:', pu_cond, 'pd_cond:', pd_cond)
                if np.random.rand() < pu:
                    #print('up')
                    # spin-up sampled
                    p_cond = pu_cond #pu
                    String = np.matmul(String, self.AuAu)
                    sample[n] = 1
                else:
                    #print('down')
                    # spin-down sampled
                    p_cond = p_cond-pu_cond #1-pu
                    String = np.matmul(String, self.AdAd)
                    sample[n] = 0

            samples[k, :] = sample

        return samples


    def generate_vec(self, N_spins, K_paths):
        samples = np.empty((K_paths, N_spins))

        chi2 = self.chi**2
        String = np.matmul(self.L, np.eye(chi2)) # product of fixed spin operators
        String_vec = np.reshape(np.tile(String, (K_paths, 1)), (K_paths, chi2))
        AuAuR = np.matmul(self.AuAu, self.R) / self.LR
        p_cond = 1
        p_cond_vec = np.ones(K_paths)
        samples = np.empty((K_paths, N_spins))
        lamb = np.array([self.lam ** (n+1) for n in range(N_spins)])

        for n in range(N_spins):
            print(String_vec.shape)
            print(AuAuR.shape)
            String_AuAu_vec = np.matmul(String_vec, AuAuR)

            pu_cond_vec = String_AuAu_vec / lamb[n]
            pu_vec = pu_cond_vec / p_cond_vec
            print(pu_vec.shape)

            String_vec = np.where(np.random.rand(K_paths) < pu_vec,
                                  np.matmul(String_vec, self.AuAu),
                                  np.matmul(String_vec, self.AdAd))


            if np.random.rand() < pu:
                # spin-up sampled
                p_cond = pu_cond #pu
                String = np.matmul(String, self.AuAu)
                samples[n] = 1
            else:
                # spin-down sampled
                p_cond = p_cond-pu_cond #1-pu
                String = np.matmul(String, self.AdAd)
                samples[n] = 0

            samples[k, :] = samples

        return samples


    def correlations(self, samples):
        samples = samples - 0.5
        K_paths, N_spins = samples.shape
        corr = np.zeros(N_spins)
        for shift in range(N_spins):
            overlap = samples[:, :N_spins-shift] * samples[:, shift:]
            corr[shift] = np.mean(overlap)
        print(corr)
        return corr


class MPSModel(Model):
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
        self.result_log = pd.DataFrame(columns=['step', 'runtime', 'chi', 'lowest_energy', 'delta_energy'])

        self.log2N = int(np.log2(self.N-2)) # fastest for N = 2^k +2 = 4, 6, 10, 18, ...
        print('Working with N=', 2**self.log2N+2, ' chi=', self.chi)

        self.A0 = A0
        self.A = tf.Variable(A0, trainable=True, dtype='float64')

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)  # RMSprop Adam learning_rate

        self.lowest_energy = float('inf')

        self.ax1 = axs[0]
        self.ax1.set_xlabel('runtime (sec)')
        self.ax1.set_ylabel('bond energy')
        self.ax1.set_xscale('log')
        self.ax1.set_yscale('log')
        self.ax1.grid()

    # def bond_energy(self):
    #
    #     print('NOT EXECUTED')
    #     Au = self.A[0]
    #     Ad = self.A[1]
    #
    #     # boost
    #     Au = tf.experimental.numpy.kron(Au, Au)
    #     Ad = tf.experimental.numpy.kron(Ad, Ad)
    #
    #     #
    #
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


    # Use tf.GradientTape to train the model:
    # @tf.function
    # def my_train_step(self):
    #     with tf.GradientTape() as tape:
    #         bond_energy, nodes = self.bond_energy()
    #         gradients = tape.gradient(bond_energy, self.trainable_variables)
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #
    #     return bond_energy, nodes


    # function:
    def n_bond_energy(self, A):
        self.counter += 1

        A = tf.reshape(A, [2, self.chi, self.chi])
        Au = A[0]
        Ad = A[1]

        # boost
        Au = tf.experimental.numpy.kron(Au, A_boost[0])
        Ad = tf.experimental.numpy.kron(Ad, A_boost[1])

        #

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

    # The objective function and the gradient.
    #@tf.function
    def bond_energy_and_gradient(self, A):
        return tfp.math.value_and_gradient(self.n_bond_energy, A)

    #@tf.function
    def run_model(self, color, tic):

        self.color = color
        sh = np.shape(A0)
        start = tf.constant(np.reshape(A0,sh[0]*sh[1]*sh[2]))  # Starting point for the search.
        optim_results = tfp.optimizer.bfgs_minimize(
            self.bond_energy_and_gradient, initial_position=start, parallel_iterations=2,
            tolerance=1e-7, max_iterations=self.max_iter)  # tolerance=1e-8

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

    if 1: # MPS calculation
        k = 10 # -> N = 2 + 2**k # only works with these N values
        N = 2 + 2 ** k
        en_est_fo = -1/np.pi + np.pi**2/12/N**2
        print('en_est_fo:', en_est_fo)

        epochs = 5000
        colors = ['r', 'g', 'b', 'k']

        np.random.seed(100)

        result_log = pd.DataFrame(columns=['step', 'runtime', 'chi', 'lowest_energy', 'delta_energy'])
        tic = datetime.datetime.now()
        fig1 = plt.figure(1, figsize=(15, 9))
        axs = []
        for i in range(8):
            axs.append(fig1.add_subplot(2, 4, i+1))

        chi = 4 # starting value 4
        A0 = np.random.randn(2, chi, chi)  # 2 => up, down
        A_opt_prev = np.empty(shape=(2, chi, chi))

        epochs_list = 8*[3000] # [2000, 1500, 1000, 800, 600, 400, 400, 400] #[2000, 2000, 1600, 1000, 1000]
        for i_chi in range(2):
            A0 = eig_normalize(A0)
            A_boost = tf.constant(np.random.randn(2, chi, chi))
            learning_rate = 0.001 #*(i_chi+1)**2 #0.001 default
            print('learning_rate', learning_rate)
            if i_chi < 7:
                max_iter = 2000
            else:
                max_iter = 4000
            model = MPSModel(k, epochs_list[i_chi], A0, learning_rate, axs, max_iter)
            model.run_model(color=colors[i_chi % len(colors)], tic=tic)
            result_log = pd.concat([result_log, model.result_log], axis=0, ignore_index=True)
            #print('result_log:\n', result_log)
            A_opt = model.A.numpy()
            analyze_MPS(A_opt, axs, color=colors[i_chi % len(colors)])
            if i_chi == 0:
                A_opt_prev = A_opt
            #A_pretrained = augment_combine(A_opt_prev, A_opt, small=0.001) #0.01
            ##A_pretrained = augment_double(A_opt, small=0.00001)  # 0.01

            #A_zero = np.zeros(A_opt_prev.shape)
            A_zero = np.zeros([2, 2, 2]) # always adding +4 extra dims [2, 4, 4]
            A0 = augment_combine(A_opt, A_zero, small=0.001) # small=0.00001  0.001

            A_opt_prev = A_opt

        outfile = 'MPS_M_dump'
        np.savez(outfile, M=A_opt)

        fig2 = plt.figure(2, figsize=(15, 9))
        axs = []
        for i in range(8):
            axs.append(fig2.add_subplot(2, 4, i + 1))
        analyze_MPS(A_opt, axs, color=colors[i_chi % len(colors)])

        plt.loglog(result_log['runtime'], result_log['delta_energy'], '.')
        plt.xlabel('runtime (s)')
        plt.ylabel('bond energy abs error')
        plt.title(r'BFGS optimizer $\chi=4,19,34$ scheme')
        axs[0].set_xlabel('runtime (sec)')
        axs[0].set_ylabel('bond energy')
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].plot(result_log['runtime'], result_log['delta_energy'], '.')
        axs[0].title.set_text('lowest_energy:' + str(model.lowest_energy) +
                                '\nN=' + str(model.N) + ' chi_max=' + str(model.chi) + ' epochs=' + str(model.epochs))

    if 0: # Sampling
        outfile = 'MPS_M_dump'
        npzfile = np.load(outfile+'.npz')
        M_tensor = npzfile['M']

        # test case:
        if 0:
            eps = 0.9
            Au = np.array([[1, 0], [eps, 0]])
            Ad = np.array([[0, eps], [0, 1]])
            M_tensor = np.array([Au, Ad])

        np.random.seed(100)
        sampler = MPS_Sampler(M_tensor)

        N_spins = 20
        K_paths = 12
        samples = sampler.generate_vec(N_spins, K_paths)
        print('samples', samples)
        sample_corr = sampler.correlations(samples)
        print('sample_corr:', sample_corr)

        samples2 = sampler(N_spins, K_paths)
        sample_corr2 = sampler.correlations(samples2)

        samples3 = sampler(N_spins, K_paths)
        sample_corr3 = sampler.correlations(samples3)

        plt.figure(2)
        #plt.plot(np.array([3,4,8,7]))
        plt.loglog(np.abs(sample_corr), '-k.')
        plt.loglog(np.abs(sample_corr2), 'b.')
        plt.loglog(np.abs(sample_corr3), 'r.')


print('--- The End ---')
plt.show()
