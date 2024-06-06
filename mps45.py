# Adding new degrees of freedom as empty rows/columns according to a +4 scheme.
# Adding a random binary mask on A to make it sparse during training and cut back the degrees of freedom.
# Randomness is quenched.


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


def eig_normalize(A):
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


def augment(A, k):  # addition does not learn
    Au, Ad = A
    small = 0.0001
    n, m = Au.shape
    Au_aug = np.vstack([np.hstack([Au, np.zeros([n, k])]), np.hstack([np.zeros([k, m]), small * np.ones([k, k])])])
    n, m = Ad.shape
    Ad_aug = np.vstack([np.hstack([Ad, np.zeros([n, k])]), np.hstack([np.zeros([k, m]), small * np.ones([k, k])])])

    return np.array([Au_aug, Ad_aug])


def augment_double(A, small):
    Au, Ad = A
    n, m = Au.shape
    hor1 = np.hstack([Au, small * np.random.randn(n, m)])
    hor2 = np.hstack([small * np.random.randn(n, m), Au])
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
    AB = np.empty(shape=(num_states, nA + nB, mA + mB))
    for s in range(num_states):  # s = up, down
        hor1 = np.hstack([A[s], small * np.random.randn(nA, mB)])
        hor2 = np.hstack([small * np.random.randn(nB, mA), B[s]])
        AB[s] = np.vstack([hor1, hor2])
    return AB


def augment_offdiag(A, k):
    Au, Ad = A
    small = 0.001
    n, m = Au.shape
    Au_aug = np.vstack([np.hstack([Au, small * np.random.randn(n, k)]), np.zeros([k, m + k])])
    n, m = Ad.shape
    Ad_aug = np.vstack([np.hstack([Ad, small * np.random.randn(n, k)]), np.zeros([k, m + k])])

    return np.array([Au_aug, Ad_aug])


def augment_old(A, k):  # addition does not learn
    Au, Ad = A
    n, m = Au.shape
    Au_aug = np.vstack([np.hstack([Au, np.zeros([n, k])]), np.zeros([k, m + k])])
    n, m = Ad.shape
    Ad_aug = np.vstack([np.hstack([Ad, np.zeros([n, k])]), np.zeros([k, m + k])])

    return np.array([Au_aug, Ad_aug])


def analyze_MPS(A, axs, color):
    Au, Ad = A
    Au = Au * model.mask_u
    Ad = Ad * model.mask_d

    E = np.kron(Au, Au) + np.kron(Ad, Ad)
    evals = eigvals(E, check_finite=True)
    eval_absmax = np.max(np.abs(evals))

    evals = evals / eval_absmax
    print('eval_absmax:', eval_absmax)
    r = 1 / np.sqrt(eval_absmax)
    Au, Ad = r * A
    Au = Au * model.mask_u
    Ad = Ad * model.mask_d

    # plotting spectrum of E:
    circle1 = plt.Circle((0., 0.), radius=1., color='green', fill=False)
    axs[1].plot(evals.real, evals.imag, color + '.')
    axs[1].add_patch(circle1)
    axs[1].axis('equal')
    axs[1].set_xlabel('real part')
    axs[1].set_ylabel('imag part')
    axs[1].title.set_text('Spectrum of the transfer matrix E')
    # print(evals)

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
    # ind = np.argmax(np.abs(evals_k))
    indices = np.argsort(np.abs(evals_k))
    ind = indices[-1]
    print('---Right:\nevals_k:', evals_k, np.abs(evals_k))
    print('ind:', ind, 'evals_k[ind]:', evals_k[ind])

    v_right = evecs_k[:, ind]  # real
    # print('v_right', v_right)

    # Left eigenvector of E:
    evals_k, evecs_k = eigs(np.transpose(E), k=8, which='LM')
    # ind = np.argmax(np.abs(evals_k))
    indices = np.argsort(np.abs(evals_k))
    ind = indices[-1]
    print('---Left:\nevals_k:', evals_k, np.abs(evals_k))
    print('ind:', ind, 'evals_k[ind]:', evals_k[ind])
    v_left = np.conjugate(evecs_k[:, ind])  # real but different
    # print('v_left', v_left)
    # print(np.dot(v_left, v_right))

    # two-point functions:
    MSzM = 0.5 * (np.kron(Au, Au) - np.kron(Ad, Ad))
    MSpM = np.kron(Au, Ad)
    MSmM = np.kron(Ad, Au)
    Er = np.eye(len(E))
    r_max = 50
    corr = np.zeros([r_max, 3])
    for l in range(1, r_max):
        Ozzr = np.matmul(np.matmul(MSzM, Er), MSzM)
        Opmr = np.matmul(np.matmul(MSpM, Er), MSmM)
        Ompr = np.matmul(np.matmul(MSmM, Er), MSpM)

        Dr = np.matmul(np.matmul(E, Er), E)
        denominator = np.dot(np.dot(v_left, Dr), v_right)

        # zz:
        corr[l, 0] = np.dot(np.dot(v_left, Ozzr), v_right) / denominator
        # pm:
        corr[l, 1] = np.dot(np.dot(v_left, Opmr), v_right) / denominator
        # mp:
        corr[l, 2] = np.dot(np.dot(v_left, Ompr), v_right) / denominator

        Er = np.matmul(Er, E)
    # print('corr:', corr)
    # eigenvalue method with L and R doesnt seem to work, maybe due to gapless spectrum and E spectral singularity

    # Lukyanov-Terras:
    corrzz_exact = np.zeros(r_max)
    slope_0p5 = np.zeros(r_max)
    for l in range(1, r_max):
        corrzz_exact[l] = - 1 / (2 * np.pi ** 2 * l ** 2) + (-1) ** l * 1 / (2 * np.pi ** 2 * l ** 2)
        slope_0p5[l] = 0.4 / l ** 0.5

    axs[4].clear()
    axs[4].title.set_text('correlations')
    axs[4].plot(corr[:, 0], 'r.-', label='$<S^z_0S^z_r>$')
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
    # axs[6].colorbar(im6, ax=axs[6])

    axs[7].clear()
    axs[7].title.set_text('Heat map of Ad')
    im7 = axs[7].imshow(Ad, cmap='cividis')
    # plt.colorbar(im7, ax=axs[7])


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
            self.lam = np.real(w[indices[-1]])  # is it justified?
            self.R = vr[:, indices[-1]]
            self.L = vl[:, indices[-1]]
            self.LR = np.dot(self.L, self.R)
            print('self.LR', self.LR)

            print('leading eig:\n', self.lam, self.L, self.R)

    def __call__(self, N_spins, K_paths):
        samples = np.empty((K_paths, N_spins))

        for k in range(K_paths):
            if (k < 1000 and k % 100 == 0) or (k >= 1000 and k % 100 == 0):
                print(k, '-----------\n')
            String = np.matmul(self.L, np.eye(self.chi ** 2))  # product of fixed spin operators
            AuAuR = np.matmul(self.AuAu, self.R) / self.LR
            p_cond = 1
            sample = np.empty(N_spins)
            lamb = np.array([self.lam ** (n + 1) for n in range(N_spins)])
            for n in range(N_spins):
                String_AuAu = np.matmul(String, AuAuR)
                # String_AdAd = np.matmul(String, self.AdAd)
                pu_cond = String_AuAu / lamb[n]
                # pd_cond = np.matmul(np.matmul(self.L, String_AdAd), self.R) / self.lam ** (n + 1) / self.LR
                pu = pu_cond / p_cond
                # print('pu:', pu, 'p_cond:', p_cond, 'pu_cond:', pu_cond, 'pd_cond:', pd_cond)
                if np.random.rand() < pu:
                    # print('up')
                    # spin-up sampled
                    p_cond = pu_cond  # pu
                    String = np.matmul(String, self.AuAu)
                    sample[n] = 1
                else:
                    # print('down')
                    # spin-down sampled
                    p_cond = p_cond - pu_cond  # 1-pu
                    String = np.matmul(String, self.AdAd)
                    sample[n] = 0

            samples[k, :] = sample

        return samples

    def generate_vec(self, N_spins, K_paths):
        samples = np.empty((K_paths, N_spins))

        chi2 = self.chi ** 2
        String = np.matmul(self.L, np.eye(chi2))  # product of fixed spin operators
        String_vec = np.reshape(np.tile(String, (K_paths, 1)), (K_paths, chi2))
        AuAuR = np.matmul(self.AuAu, self.R) / self.LR
        p_cond = 1
        p_cond_vec = np.ones(K_paths)
        samples = np.empty((K_paths, N_spins))
        lamb = np.array([self.lam ** (n + 1) for n in range(N_spins)])

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
                p_cond = pu_cond  # pu
                String = np.matmul(String, self.AuAu)
                samples[n] = 1
            else:
                # spin-down sampled
                p_cond = p_cond - pu_cond  # 1-pu
                String = np.matmul(String, self.AdAd)
                samples[n] = 0

            samples[k, :] = samples

        return samples

    def correlations(self, samples):
        samples = samples - 0.5
        K_paths, N_spins = samples.shape
        corr = np.zeros(N_spins)
        for shift in range(N_spins):
            overlap = samples[:, :N_spins - shift] * samples[:, shift:]
            corr[shift] = np.mean(overlap)
        print(corr)
        return corr


class MPSModel(Model):
    def __init__(self, k, epochs, A0, learning_rate, axs):
        super().__init__()

        self.k = k
        self.N = 2 ** k + 2
        self.chi = A0[0].shape[0]
        self.en_inf_fo = -1 / np.pi + np.pi ** 2 / 12 / self.N ** 2  # first order
        self.epochs = epochs

        self.log2N = int(np.log2(self.N - 2))  # fastest for N = 2^k +2 = 4, 6, 10, 18, ...
        print('Working with N=', 2 ** self.log2N + 2)

        self.A = tf.Variable(A0, trainable=True, dtype='float64')
        #p0 = 0.4
        #self.mask_u = np.random.choice([0, 1], size=(self.chi, self.chi), p=[p0, 1 - p0])
        #self.mask_d = np.random.choice([0, 1], size=(self.chi, self.chi), p=[p0, 1 - p0])
        self.mask_u = mask_u[:self.chi, :self.chi]
        self.mask_d = mask_d[:self.chi, :self.chi]
        print('mask_u:', self.mask_u)
        print('mask_d:', self.mask_d)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate)  # RMSprop Adam learning_rate

        self.lowest_energy = float('inf')

        self.ax1 = axs[0]
        self.ax1.set_xlabel('runtime (sec)')
        self.ax1.set_ylabel('bond energy')
        self.ax1.set_yscale('log')

    def bond_energy(self):
        Au = self.A[0] * self.mask_u
        Ad = self.A[1] * self.mask_d
        E = tf.experimental.numpy.kron(Au, Au) + tf.experimental.numpy.kron(Ad, Ad)
        E2 = tf.matmul(E, E)

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
        # tf.print('trGN, trEN:', trGN, trEN)

        nodes = [trGN, trEN]  # nodes = [trGN, trEN, E, EN]
        return energy, nodes

    # Use tf.GradientTape to train the model:
    @tf.function
    def my_train_step(self):
        with tf.GradientTape() as tape:
            bond_energy, nodes = self.bond_energy()
            if 0:
                loss = bond_energy + 0.01 * tf.square(tf.math.log(nodes[1]))  # if regularization needed
                gradients = tape.gradient(loss, self.trainable_variables)
                # tf.print('grad:', tf.norm(gradients))
                # co = self.optimizer.get_config()
            else:
                gradients = tape.gradient(bond_energy, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return bond_energy, nodes

    def run_model(self, color, tic):

        patience = 200
        wait = 0
        epoch_best_energy = float('inf')

        for i in range(1, self.epochs + 1):

            bond_energy, nodes = self.my_train_step()

            if bond_energy.numpy() < self.lowest_energy:
                self.lowest_energy = bond_energy.numpy()

            if (i < 100 and i % 10 == 0) or (i >= 100 and i % 50 == 0):
                toc = datetime.datetime.now()
                print('chi:', self.A.shape[1], 'Epoch:', i, 'time:', toc, toc - tic, '-----------------------')
                print('    bond_energy:', bond_energy.numpy())
                for j in range(2):
                    print('    nodes:', nodes[j].numpy())

                self.ax1.plot((toc - tic).total_seconds(), bond_energy - self.en_inf_fo, color + '.')  # inf: 0.31831
                self.ax1.title.set_text('lowest_energy:' + str(self.lowest_energy) +
                                        '\nN=' + str(self.N) + ' chi_max=' + str(self.chi) + ' epochs=' + str(
                    self.epochs))
                plt.pause(.1)

            # The early stopping strategy: stop the training if `val_loss` does not
            # decrease over a certain number of epochs.
            wait += 1
            if bond_energy.numpy() < epoch_best_energy - 1e-8:
                wait = 0
            else:
                print('Insufficient improvement: i, wait:', i, wait, 'en:', bond_energy.numpy(), epoch_best_energy)
            if bond_energy.numpy() < epoch_best_energy:
                epoch_best_energy = bond_energy.numpy()
            if wait >= patience:
                print('Early stopping:', bond_energy.numpy(), epoch_best_energy)
                break


if __name__ == '__main__':

    if 1:  # MPS calculation
        k = 8  # -> N = 2 + 2**k # only works with these N values
        N = 2 + 2 ** k
        en_est_fo = -1 / np.pi + np.pi ** 2 / 12 / N ** 2
        print('en_est_fo:', en_est_fo)

        epochs = 5000
        colors = ['r', 'g', 'b', 'k']

        np.random.seed(100)
        p0 = 0.75
        mask_u = np.random.choice([0, 1], size=(100, 100), p=[p0, 1 - p0])
        mask_d = np.random.choice([0, 1], size=(100, 100), p=[p0, 1 - p0])

        tic = datetime.datetime.now()
        fig1 = plt.figure(1, figsize=(12, 6))
        axs = []
        for i in range(8):
            axs.append(fig1.add_subplot(2, 4, i + 1))

        chi = 4  # starting value 4
        A0 = np.random.randn(2, chi, chi)  # 2 => up, down
        A_opt_prev = np.empty(shape=(2, chi, chi))

        epochs_list = 8 * [3000]  # [2000, 1500, 1000, 800, 600, 400, 400, 400] #[2000, 2000, 1600, 1000, 1000]
        for i_chi in range(6):
            A0 = eig_normalize(A0)
            learning_rate = 0.001  # *(i_chi+1)**2 #0.001 default
            print('learning_rate', learning_rate)
            model = MPSModel(k, epochs_list[i_chi], A0, learning_rate, axs)
            model.run_model(color=colors[i_chi % len(colors)], tic=tic)
            A_opt = model.A.numpy()
            analyze_MPS(A_opt, axs, color=colors[i_chi % len(colors)])
            if i_chi == 0:
                A_opt_prev = A_opt
            # A_pretrained = augment_combine(A_opt_prev, A_opt, small=0.001) #0.01
            ##A_pretrained = augment_double(A_opt, small=0.00001)  # 0.01

            # A_zero = np.zeros(A_opt_prev.shape)
            A_zero = np.zeros([2, 8, 8])  # always adding +4 extra dims [2, 4, 4]
            A0 = augment_combine(A_opt, A_zero, small=0.00001)

            A_opt_prev = A_opt

        outfile = 'MPS_M_dump'
        np.savez(outfile, M=A_opt)

    if 0:  # Sampling
        outfile = 'MPS_M_dump'
        npzfile = np.load(outfile + '.npz')
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
        # plt.plot(np.array([3,4,8,7]))
        plt.loglog(np.abs(sample_corr), '-k.')
        plt.loglog(np.abs(sample_corr2), 'b.')
        plt.loglog(np.abs(sample_corr3), 'r.')

print('--- The End ---')
plt.show()
