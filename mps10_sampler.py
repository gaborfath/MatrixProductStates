# Sampling from M tensor
# Calibrating M from empirical time series

#import tensorflow as tf
#from tensorflow.python.keras import Model
#from tensorflow.python.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#import datetime
from scipy.linalg import eig
#import tensorflow_probability as tfp
#import pandas as pd



class MPS_Sampler():
    def __init__(self, M_tensor):
        self.M = M_tensor
        self.Au = self.M[0]
        self.Ad = self.M[1]
        self.chi = self.Au.shape[0]

        self.AuAu = np.kron(self.Au, self.Au)
        self.AdAd = np.kron(self.Ad, self.Ad)
        self.E = self.AuAu + self.AdAd

        w, vl, vr = eig(self.E, left=True, right=True)
        print('w:\n', w)

        indices = np.argsort(np.abs(w))
        ind = indices[-1]
        self.lam = w[ind]
        if abs(np.imag(self.lam)) > 1e-12:
            print('Complex lam:', self.lam)
            exit('Complex lam')
        else:
            self.lam = np.real(self.lam)
        self.R = np.real(vr[:, ind])  # is it valid?
        self.L = np.real(vl[:, ind])  # is it valid?
        self.LR = np.dot(self.L, self.R)

        self.xi = abs(1/np.log(w[indices[-1]] / w[indices[-2]]))

        print('self.LR', self.LR)
        print('leading eig:\n', self.lam, self.L, self.R)
        print('correlation length: ', abs(self.xi))


    def __call__(self, N_spins, K_paths):
        print('Running sampling')
        samples = np.empty((K_paths, N_spins))
        rands = np.random.rand(K_paths, N_spins)
        AuAuR = np.matmul(self.AuAu, self.R) / self.LR
        AdAdR = np.matmul(self.AdAd, self.R) / self.LR

        for k in range(K_paths):
            if (k < 1000 and k % 100 == 0) or (k >= 1000 and k % 200 == 0):
                print('Path:', k)
            String = self.L
            for n in range(N_spins):
                String_AuAuR = np.matmul(String, AuAuR)
                String_AdAdR = np.matmul(String, AdAdR)  # decreasing exponentially!
                #print('hah:', k, 'n:', n, String_AuAuR, String_AdAdR)
                pu =  String_AuAuR / (String_AuAuR + String_AdAdR)

                if rands[k, n] < pu:  # spin-up sampled
                    samples[k, n] = 1
                    String = np.matmul(String, self.AuAu)
                else:  # spin-down sampled
                    samples[k, n] = 0
                    String = np.matmul(String, self.AdAd)

        return samples

    def __call__old(self, N_spins, K_paths):
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


    # def generate_vec(self, N_spins, K_paths):
    #     samples = np.empty((K_paths, N_spins))
    #
    #     chi2 = self.chi**2
    #     String = np.matmul(self.L, np.eye(chi2)) # product of fixed spin operators
    #     String_vec = np.reshape(np.tile(String, (K_paths, 1)), (K_paths, chi2))
    #     AuAuR = np.matmul(self.AuAu, self.R) / self.LR
    #     p_cond = 1
    #     p_cond_vec = np.ones(K_paths)
    #     samples = np.empty((K_paths, N_spins))
    #     lamb = np.array([self.lam ** (n+1) for n in range(N_spins)])
    #
    #     for n in range(N_spins):
    #         print(String_vec.shape)
    #         print(AuAuR.shape)
    #         String_AuAu_vec = np.matmul(String_vec, AuAuR)
    #
    #         pu_cond_vec = String_AuAu_vec / lamb[n]
    #         pu_vec = pu_cond_vec / p_cond_vec
    #         print(pu_vec.shape)
    #
    #         String_vec = np.where(np.random.rand(K_paths) < pu_vec,
    #                               np.matmul(String_vec, self.AuAu),
    #                               np.matmul(String_vec, self.AdAd))
    #
    #
    #         if np.random.rand() < pu:
    #             # spin-up sampled
    #             p_cond = pu_cond #pu
    #             String = np.matmul(String, self.AuAu)
    #             samples[n] = 1
    #         else:
    #             # spin-down sampled
    #             p_cond = p_cond-pu_cond #1-pu
    #             String = np.matmul(String, self.AdAd)
    #             samples[n] = 0
    #
    #         samples[k, :] = samples
    #
    #     return samples


    def corr_by_time_avr(self, samples):
        print('Running corr_by_time_avr')
        sample_mean = samples.mean(axis=1)
        print('sample_mean over paths:', samples.mean(axis=0))
        print('sample_mean over nodes:', samples.mean(axis=1))


        #samples = samples - sample_mean[:, np.newaxis]  # remove the mean
        #sample_mean = samples.mean()
        print('sample_mean:', sample_mean)
        samples = samples - 0.5 #sample_mean  # remove the mean

        K_paths, N_spins = samples.shape
        max_r = N_spins
        sample_corr = np.zeros([K_paths, max_r])

        for r in range(max_r):
            overlap = samples[:, :N_spins - r] * samples[:, r:]
            sample_corr[:, r] = np.mean(overlap, axis=1)  # over time dimension

        corr = np.mean(sample_corr, axis=0)
        error = np.std(sample_corr, axis=0) / np.sqrt(K_paths)
        return corr, error


    def corr_by_ensemble_avr(self, samples):

        return

    def corr_exp_decay(self):
        x = np.linspace(0, 10, 20)
        y = np.exp(-x / self.xi)
        return x, y


    def correlations(self, samples):
        samples = samples - 0.5
        K_paths, N_spins = samples.shape
        corr = np.zeros(N_spins)
        for shift in range(N_spins):
            overlap = samples[:, :N_spins-shift] * samples[:, shift:]
            corr[shift] = np.mean(overlap)
        print(corr)
        return corr


class MPS_Clibrator():
    def __init__(self, series):
        return


if __name__ == '__main__':

    if 1:  # Sampling
        outfile = 'MPS_M_dump'
        npzfile = np.load(outfile + '.npz')
        M_tensor = npzfile['M']
        print('M shape:', M_tensor.shape)

    else:  # test case:
        eps = 0.2
        Au = np.array([[1, 0], [eps, 0]])
        Ad = np.array([[0, eps], [0, 1]])
        M_tensor = np.array([Au, Ad])

    np.random.seed(100)
    sampler = MPS_Sampler(M_tensor)

    N_spins = 10
    K_paths = 500
    # samples = sampler.generate_vec(N_spins, K_paths)
    samples = sampler(N_spins, K_paths)
    print('samples', samples)

    # save to disk:
    np.save('samples.npy', samples)  # save
    ### samples = np.load('samples.npy')  # load

    corr, error = sampler.corr_by_time_avr(samples)
    #print('corr:', corr)

    plt.figure(1)
    # plt.plot(np.array([3,4,8,7]))
    plt.loglog(abs(corr), '-k.')
    plt.loglog(error, '-r.')
    x, y = sampler.corr_exp_decay()
    plt.loglog(x, corr[0]*y, '-g.')
    plt.show()

    exit()

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