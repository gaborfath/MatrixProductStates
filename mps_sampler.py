# Sampling from M tensor
# Calibrating M from empirical time series

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.stats import norm

from mps_comparator import *
from mps_comparator import MPS_comparator


class MPS_Sampler():

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, input_params):

        np.set_printoptions(nanstr='')

        self.N_spins = input_params['N_spins']
        self.K_paths = input_params['K_paths']
        self.M = input_params['M_tensor']
        self.transient_multiplier = input_params['transient_multiplier']
        self.visualization_noise = input_params['visualization_noise']
        self.max_markov_length = input_params['max_markov_length']

        self.Au = self.M[0]
        self.Ad = self.M[1]
        self.chi = self.Au.shape[0]

        self.AuAu = np.kron(self.Au, self.Au)
        self.AdAd = np.kron(self.Ad, self.Ad)
        self.E = self.AuAu + self.AdAd

        w, vl, vr = eig(self.E, left=True, right=True)

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

        self.transient_size = round(self.transient_multiplier * self.xi)

        self.loglikelihood_tot = 0

        # to record node probailities in decision tree:
        if self.N_spins <= 20:
            self.decision_prob_theoretical = np.empty([self.N_spins, 2**(self.N_spins-1), 2]) # site, node, spin
            self.decision_prob_theoretical.fill(np.nan)

            self.decision_count = np.zeros([self.N_spins, 2 ** (self.N_spins - 1), 2])  # site, node, spin

            self.decision_prob_empirical = np.empty([self.N_spins, 2**(self.N_spins-1), 2]) # site, node, spin
            self.decision_prob_empirical.fill(np.nan)
        else:
            exit('N_spins > 20, stopping to avoid creating a large matrix')


        print('Eigs:', w)
        print('Lam1:', self.lam, 'xi:', self.xi)
        print('')


    #-------------------------------------------------------------------------------------------------------------------
    def __call__(self):

        # print('Running sampling')
        samples = -np.ones((self.K_paths, self.N_spins))
        sample_probs = -np.ones((self.K_paths, self.N_spins))
        rands = np.random.rand(self.K_paths, self.N_spins)
        AuAuR = np.matmul(self.AuAu, self.R) / self.LR
        AdAdR = np.matmul(self.AdAd, self.R) / self.LR

        self.loglikelihood = np.zeros(self.K_paths)
        naive_model_loglikelihood = 0  # naive_model: all probs are 1/2

        for k in range(self.K_paths):
            if (k < 1000 and k % 1000 == 0) or (k >= 1000 and k % 2000 == 0):
                print('Path:', k)
            String = self.L


            current_int_config = 0

            for n in range(self.N_spins):
                String_AuAuR = np.matmul(String, AuAuR)
                String_AdAdR = np.matmul(String, AdAdR)  # decreasing exponentially!
                pu =  String_AuAuR / (String_AuAuR + String_AdAdR)
                sample_probs[k, n] = pu
                self.decision_prob_theoretical[n, current_int_config, 1] = pu # [site, node, decision] -> prob
                self.decision_prob_theoretical[n, current_int_config, 0] = 1 - pu  # [site, node, decision] -> prob
                #print('hah:', k, 'n:', n, String_AuAuR, String_AdAdR, 'pu:', pu)

                if rands[k, n] < pu:  # spin-up sampled
                    samples[k, n] = 1
                    self.decision_count[n, current_int_config, 1] += 1
                    current_int_config = 2 * current_int_config +1
                    String = np.matmul(String, self.AuAu)
                    self.loglikelihood[k] += np.log(pu)
                    naive_model_loglikelihood += np.log(0.5)
                else:  # spin-down sampled
                    samples[k, n] = 0
                    self.decision_count[n, current_int_config, 0] += 1
                    current_int_config = 2 * current_int_config
                    String = np.matmul(String, self.AdAd)
                    self.loglikelihood[k] += np.log(1 - pu)
                    naive_model_loglikelihood += np.log(0.5)
                if n % 10 == 0:
                    String = String / np.linalg.norm(String)  # safe and enough to renormalize here

        self.loglikelihood_tot = np.sum(self.loglikelihood) / (self.K_paths * self.N_spins)
        self.naive_model_loglikelihood = naive_model_loglikelihood / (self.K_paths * self.N_spins)

        self.decision_prob_empirical = self.decision_count[:, :, 1] / (np.sum(self.decision_count, axis=2))

        print('self.decision_prob_theoretical[7, :, 1]', self.decision_prob_theoretical[7, :, 1])
        return samples, sample_probs

    # Create every possible configuration and calculate decition_prob_theoretical
    def populate_decision_prob_theoretical(self):
        print('Not good yet at all!')
        exit()

        AuAuR = np.matmul(self.AuAu, self.R) / self.LR
        AdAdR = np.matmul(self.AdAd, self.R) / self.LR

        for k in range(2**(self.N_spins - 1)):
            print('k:', k)

            String = self.L

            bit_list = self.int2bits(k, N_spins-1)
            print('bit_list', bit_list)

            for n in bit_list:
                String_AuAuR = np.matmul(String, AuAuR)
                String_AdAdR = np.matmul(String, AdAdR)  # decreasing exponentially!
                pu = String_AuAuR / (String_AuAuR + String_AdAdR)
                self.decision_prob_theoretical[n, k, 1] = pu  # [site, node, decision] -> prob
                self.decision_prob_theoretical[n, k, 0] = 1 - pu  # [site, node, decision] -> prob
                # print('hah:', k, 'n:', n, String_AuAuR, String_AdAdR, 'pu:', pu)

                if n==1:  # spin-up sampled
                    String = np.matmul(String, self.AuAu)
                else:  # spin-down sampled
                    String = np.matmul(String, self.AdAd)
                if n % 10 == 0:
                    String = String / np.linalg.norm(String)  # safe and enough to renormalize here

                print('pu', self.decision_prob_theoretical[n, k, 1])

        print('self.decision_prob_theoretical[6:, :, 1]', self.decision_prob_theoretical[6:, :, 1])

        return


    # Transform bit configuration array into integer
    def bits2int(self, bits):
        return bits.dot(2 ** np.arange(len(bits))[::-1])

    def int2bits(self, n, num_bits):
        # n: integer to convert to binary array
        # num_bits: min length of the resulting numpy array
        return np.array([int(i) for i in bin(n)[2:].zfill(num_bits)])

    def markov_check(self, samples, sample_probs):
        # Check empirically if series is Markov:

        precursor_length = min(self.max_markov_length, self.N_spins -1)
        print('precursor_length', precursor_length)

        precursor = -np.ones([self.K_paths, self.N_spins]) # m-length historical precursor in integer form

        # calculate:
        for k in range(self.K_paths):
            for n in range(self.N_spins):

                if n >= precursor_length:
                    bits = samples[k, n - precursor_length: n]
                    #bits2int = bits.dot(2 ** np.arange(precursor_length)[::-1])
                    precursor[k, n] = self.bits2int(bits)

        # filter out transient:
        precursor = precursor[:, precursor_length:]
        sample_probs = sample_probs[:, precursor_length:]

        # visualize:
        unique, counts = np.unique(np.round(sample_probs.flatten(), 10), return_counts=True)
        with np.printoptions(precision=6, suppress=True, threshold=np.inf):
            print('\nUp_probs used when generating samples (sample_probs):')
            print('    unique | counts:')
            print(np.asarray((unique, counts)).T)

        fig, (ax1, ax2) = plt.subplots(2, 1, num=101, figsize=(4, 8))
        fig.subplots_adjust(left=0.18, bottom=None, right=None, top=None, wspace=None, hspace=0.3)

        ax1.hist(sample_probs.flatten(), bins=100, density=True)
        ax1.set_title('Histogram of pu probabilities in samples', fontsize=11)
        ax1.set_xlabel('pu probability')
        ax1.set_ylabel('Density')

        ax2.plot(precursor.flatten(), sample_probs.flatten(), 'b.')
        ax2.set_title('Markov property check', fontsize=11)
        ax2.set_xlabel('int rep of precursor string of length '+ str(precursor_length))
        ax2.set_ylabel('pu probability')
        plt.pause(0.1)


    def all_configs_stats(self, samples):
        # Compare empirical config probs with theoretical probs:

        # decision tree probs:
        print('Theoretical decision tree up_probs dictated by the M tensor:')
        print(self.decision_prob_theoretical[:,:,1].T)

        # decision tree probs:
        print('Empirical decision tree up_probs dictated by the realized sample:')
        print(self.decision_prob_empirical[:,:].T)



        # configurations statistics:
        bits_int = np.zeros(self.K_paths)
        p_theor = np.zeros(self.K_paths)
        for k in range(self.K_paths):
            bits = samples[k, :]
            bits_int[k] = self.bits2int(bits) # bits.dot(2 ** np.arange(self.N_spins)[::-1])
            p_theor[k] = np.exp(self.loglikelihood[k])


        unique, counts = np.unique(bits_int, return_counts=True)

        plt.figure(num=301)
        plt.plot(bits_int, p_theor, 'rx')
        plt.plot(unique, counts/self.K_paths, 'b.')
        plt.xlabel('configuration (integer rep)')
        plt.ylabel('frequency')
        plt.title('Occurrence frequency in sample' + ' K=' + str(self.K_paths) + ' N=' + str(self.N_spins))
        plt.legend(['theoretical', 'empirical'])



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
            if (r < 1000 and r % 100 == 0) or (r >= 1000 and r % 100 == 0):
                print('r:', r)
            overlap = samples[:, :N_spins - r] * samples[:, r:]
            sample_corr[:, r] = np.mean(overlap, axis=1)  # over time dimension

        corr = np.mean(sample_corr, axis=0)
        error = np.std(sample_corr, axis=0) / np.sqrt(K_paths)
        return corr, error

    def magnetization_by_time_avr(self, samples):




        tot_magn = samples.mean(axis=1)

        # unique, counts = np.unique(np.round(b_pu[:, 1], 10), return_counts=True)
        # with np.printoptions(precision=6, suppress=True, threshold=np.inf):
        #     print('b_pu: uniques | counts:')
        #     print(np.asarray((unique, counts)).T)

        plt.figure(10)
        plt.title('Distribution of total magnetization (after transient), xi:'+ str(round(self.xi, 4)))

        M_mean = np.mean(tot_magn)
        M_std = np.std(tot_magn)
        xx = np.linspace(0, 1, 1000)
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

    chi0 = 4
    N_spins = 8
    K_paths = 10000

    samples_file = 'samples_' + str(chi0) + '_' +str(N_spins) + '_' + str(K_paths) + '.npz'
    M_tensor_file = 'M_tensor'
    transient_multiplier = 0.0  # transient_size = transient_multiplier * sampler.xi
    visualization_noise = 0.0
    max_markov_length = 5 # max lengths of history when doing the markovity check.
                          # Effective value: min(N_spins-1, max_markov_length), i.e., last generation step

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
    elif 0:
        Au = np.array([[1,   0,   0],
                       [1.7, 0,   0],
                       [2.3, 0, 0]])
        Ad = np.array([[0,  0,  2.3],
                       [0,  0,  1.7], # 4-sate, 4-leg Markov -NO ??
                       [0,  0,  1]])
        M_tensor = np.array([Au, Ad])
        dumpM_file = 'M_tensor_33'
    else:
        np.random.seed(104)  # 104: xi=5.27
        Au = np.random.randn(chi0, chi0)
        Ad = np.random.randn(chi0, chi0)
        print('M:', Au, '\n', Ad)
        M_tensor = np.array([Au, Ad])
        dumpM_file = 'M_tensor_seed104'


    print('\nInitialization: #########################################################################################')

    # Dump M_tensor used for sampling:
    np.savez(dumpM_file, M=M_tensor)     # specific file, always the same, and has specific name
    np.savez(M_tensor_file, M=M_tensor)  # general file, always different

    input_params = {'N_spins': N_spins,
                    'K_paths': K_paths,
                    'M_tensor': M_tensor,
                    'transient_multiplier': transient_multiplier,
                    'visualization_noise': visualization_noise,
                    'max_markov_length': max_markov_length}

    print('N_spins', input_params['N_spins'], 'K_paths', input_params['K_paths'],
          'transient_multiplier:', input_params['transient_multiplier'])
    print('M_tensor.shape:', M_tensor.shape)
    print('M_tensor was dumped in:', dumpM_file)


    print('\nSampling: ###############################################################################################')

    # sampling:
    sampler = MPS_Sampler(input_params)
    samples, sample_probs = sampler()
    samples = samples[:, sampler.transient_size:]  # remove transient
    print('After removing', sampler.transient_multiplier,'*xi transients remaining samples.shape:', samples.shape)
    if 0:
        print('Samples:\n', samples)

    print('\nAverage sampled loglikelihood:', sampler.loglikelihood_tot)
    print('Naive p=0.5 model loglikelihood:', sampler.naive_model_loglikelihood)

    #sampler.populate_decision_prob_theoretical()
    #exit()

    comparator = MPS_comparator(M_tensor, M_tensor, samples)
    pu_tree = comparator.A_to_pu_tree(M_tensor)

    exit()



    # saving to disk:
    np.savez(samples_file, samples=samples, sample_probs=sample_probs,
             decision_prob_matrix=sampler.decision_prob_theoretical, M_tensor=M_tensor)
    print('\nSamples dumped in to', samples_file)


    print('\nAnalysis:  ##############################################################################################')

    # Config occurrence in sample, theoretical vs empirical
    sampler.all_configs_stats(samples)

    # Check if model is Markov:
    sampler.markov_check(samples, sample_probs)

    # Magnetization distribution:
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

    if 1:
        plt.figure(2)
        plt.semilogy(np.abs(sample_corr), '-k.')
    else:
        plt.loglog(np.abs(sample_corr), '-k.')

print('--- The End ---')
plt.show()