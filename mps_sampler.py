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
from scipy.stats import norm
#import tensorflow_probability as tfp
#import pandas as pd



class MPS_Sampler():

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, M_tensor):
        self.M = M_tensor
        self.Au = self.M[0]
        self.Ad = self.M[1]
        self.chi = self.Au.shape[0]

        self.AuAu = np.kron(self.Au, self.Au)
        self.AdAd = np.kron(self.Ad, self.Ad)
        self.E = self.AuAu + self.AdAd

        w, vl, vr = eig(self.E, left=True, right=True)
        #w_abs = np.flip(np.sort(np.abs(w)))
        #xi = 1 / np.log(w_abs[0]/w_abs[1])
        #print('w:\n', w, w_abs, xi)

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
        self.xi = 1/np.log(np.abs(w[indices[-1]] / w[indices[-2]]))

        self.loglikelihood = 0

        #print('self.LR', self.LR)
        #print('leading eig:\n', self.lam, self.L, self.R)
        #print('correlation length: ', self.xi)
        print('Eigs:', w)
        print('Lam1:', self.lam, 'xi:', self.xi)
        print('')


    #-------------------------------------------------------------------------------------------------------------------
    def __call__(self, N_spins, K_paths):

        # print('Running sampling')
        samples = -np.ones((K_paths, N_spins))
        rands = np.random.rand(K_paths, N_spins)
        AuAuR = np.matmul(self.AuAu, self.R) / self.LR
        AdAdR = np.matmul(self.AdAd, self.R) / self.LR

        pus = -np.ones(N_spins)
        b_len = 8
        b_pu = -np.ones([N_spins, 2])
        loglikelihood = 0
        naive_model_loglikelihood = 0  # naive_model: all probs are 1/2

        for k in range(K_paths):
            if (k < 1000 and k % 100 == 0) or (k >= 1000 and k % 200 == 0):
                print('Path:', k)
            String = self.L

            for n in range(N_spins):
                String_AuAuR = np.matmul(String, AuAuR)
                String_AdAdR = np.matmul(String, AdAdR)  # decreasing exponentially!
                pu =  String_AuAuR / (String_AuAuR + String_AdAdR)
                #print('hah:', k, 'n:', n, String_AuAuR, String_AdAdR, 'pu:', pu)

                # Check empirically if series is Markov:
                if k==0:
                    pus[n] = pu
                    #print('n:', n, 'formers:', samples[k, n-3:n], 'pu:', pu)

                    if n >= b_len:
                        b = samples[k, n - b_len : n]
                        b_int = b.dot(2 ** np.arange(b.size)[::-1]) + 0.005 * 2**b_len * np.random.randn()
                        b_pu[n] = [b_int, pu + 0.005 * np.random.randn()]


                if rands[k, n] < pu:  # spin-up sampled
                    samples[k, n] = 1
                    String = np.matmul(String, self.AuAu)
                    loglikelihood += np.log(pu)
                    naive_model_loglikelihood += np.log(0.5)
                else:  # spin-down sampled
                    samples[k, n] = 0
                    String = np.matmul(String, self.AdAd)
                    loglikelihood += np.log(1 - pu)
                    naive_model_loglikelihood += np.log(0.5)
                if n % 10 == 0:
                    String = String / np.linalg.norm(String)  # safe and enough to renormalize here

            if k==0:
                fig, (ax1, ax2) = plt.subplots(2, 1, num=101, figsize=(4, 8))
                ax1.hist(pus, bins=100, density=False)
                ax1.set_title('Distribution of pu probabilities in 1st path')
                ax1.set_xlabel('p_u')
                ax1.set_ylabel('Density')

                b_pu = b_pu[b_len:]
                ax2.plot(b_pu[:,0], b_pu[:,1], 'b.')
                ax2.set_title('Markov property check for 1st path')
                plt.pause(0.1)


        self.loglikelihood = loglikelihood / (K_paths * N_spins)
        self.naive_model_loglikelihood = naive_model_loglikelihood / (K_paths * N_spins)

        return samples

    def corr_by_time_avr(self, samples):
        print('Running corr_by_time_avr')
        sample_mean = samples.mean(axis=1)

        #print('sample_mean over paths:', samples.mean(axis=0))
        #samples = samples - sample_mean[:, np.newaxis]  # remove the mean
        #sample_mean = samples.mean()

        #print('sample_mean:', sample_mean)
        samples = samples - 0.5 #sample_mean  # remove the mean

        K_paths, N_spins = samples.shape
        max_r = N_spins
        sample_corr = np.zeros([K_paths, max_r])

        for r in range(max_r):
            if (r < 1000 and r % 10 == 0) or (r >= 1000 and r % 100 == 0):
                print('r:', r)
            overlap = samples[:, :N_spins - r] * samples[:, r:]
            sample_corr[:, r] = np.mean(overlap, axis=1)  # over time dimension

        corr = np.mean(sample_corr, axis=0)
        error = np.std(sample_corr, axis=0) / np.sqrt(K_paths)
        return corr, error

    def magnetization_by_time_avr(self, samples):
        tot_magn = samples.mean(axis=1)

        plt.figure(10)
        plt.title('Distribution of total magnetization (after transient), xi:'+ str(round(self.xi, 4)))

        M_mean = np.mean(tot_magn)
        M_std = np.std(tot_magn)
        xx = np.linspace(0,1,1000)
        yy = norm.pdf(xx, M_mean, M_std)

        #data = np.random.normal(M_mean, M_std, 1000)
        #print(tot_magn[:10])
        #print(data[: 10])

        tot_magn += M_std/50 * np.random.randn(len(tot_magn)) # adding some noise for stability
        plt.hist(tot_magn, bins=100, density=True)

        # Plotting the result to visualize
        plt.plot(xx, yy)
        plt.xlabel('tot magn')
        plt.ylabel('PDF')
        plt.grid(True)

        #print('sample_mean total magnetization:', samples.mean(axis=1))
        return

    def corr_by_ensemble_avr(self, samples):

        return

    def corr_exp_decay(self, r_max):
        x = np.linspace(0, r_max, 50)
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


#=======================================================================================================================
if __name__ == '__main__':

    print('\nInputs:  ################################################################################################')

    N_spins = 1000
    K_paths = 10
    transient_multiplier = 0  # transient_size = transient_multiplier * sampler.xi
    outfile = 'samples.npy'

    # Define MPS tensor to sample from:
    np.random.seed(104) #104: xi=5.27
    if 0:
        outfile = 'MPS_calibrator_M_dump_seed104' #'MPS_M_dump'
        npzfile = np.load(outfile + '.npz')
        M_tensor = npzfile['M']
        print('M shape:', M_tensor.shape)
    elif 0:  # test case:
        eps = 0.1
        Au = np.array([[1, 0], [eps, 0]])
        Ad = np.array([[0, eps], [0, 1]])
        M_tensor = np.array([Au, Ad])
        dumpM_file = 'M_tensor_eps01'
    elif 0:
        eps = 0.8
        Au = np.array([[1, 0], [eps, 0]])
        Ad = np.array([[0, eps], [0, 0]])
        M_tensor = np.array([Au, Ad])
    else:
        Au = np.random.randn(2,2)
        Ad = np.random.randn(2,2)
        print('M:', Au, Ad)
        M_tensor = np.array([Au, Ad])
        dumpM_file = 'M_tensor_seed104'
    np.savez(dumpM_file, M=M_tensor)

    print('N_spins', N_spins, 'K_paths', K_paths, 'transient_multiplier:', transient_multiplier)
    print('M_tensor.shape:', M_tensor.shape)
    print('M_tensor was dumped in:', dumpM_file)


    print('\nSampling: ###############################################################################################')

    # sampling:
    sampler = MPS_Sampler(M_tensor)
    transient_size = transient_multiplier * round(5 * sampler.xi)
    samples = sampler(transient_size + N_spins, K_paths)
    samples = samples[:, transient_size:]  # remove transient
    print('After removing', transient_multiplier,'*xi transients remaining samples.shape:', samples.shape)
    print('Average sampled loglikelihood:', sampler.loglikelihood)
    print('Naive model loglikelihood:', sampler.naive_model_loglikelihood)

    # saving to disk:
    np.save(outfile, samples)
    print('Samples dumped in to', outfile)


    print('\nAnalysis:  ##############################################################################################')



    sampler.magnetization_by_time_avr(samples)
    if 0:
        plt.show()
        exit()

    corr, error = sampler.corr_by_time_avr(samples)
    #print('corr:', corr)

    plt.figure(1)
    plt.semilogy(abs(corr), '-k.')
    plt.semilogy(error, '-r.')
    x, y = sampler.corr_exp_decay(len(corr))
    plt.semilogy(x, corr[0]*y, '-g.')

    if 1:
        print('--- The End #34 ---')
        plt.show()

        exit()

    sample_corr = sampler.correlations(samples)
    print('sample_corr:', sample_corr)

    #samples2 = sampler(N_spins, K_paths)
    #sample_corr2 = sampler.correlations(samples2)

    #samples3 = sampler(N_spins, K_paths)
    #sample_corr3 = sampler.correlations(samples3)

    plt.figure(2)
    if 1:
        plt.semilogy(np.abs(sample_corr), '-k.')
        #plt.semilogy(np.abs(sample_corr2), 'b.')
        #plt.semilogy(np.abs(sample_corr3), 'r.')
    else:
        plt.loglog(np.abs(sample_corr), '-k.')
        #plt.loglog(np.abs(sample_corr2), 'b.')
        #plt.loglog(np.abs(sample_corr3), 'r.')

print('--- The End ---')
plt.show()