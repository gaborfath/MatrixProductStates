# To compare two MPS matrices on a dataset in terms of their predictive capabilities

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from scipy.stats import norm


class MPS_comparator():

    #-------------------------------------------------------------------------------------------------------------------
    def __init__(self, M_tensor1, M_tensor2, samples):

        self.K_paths, self.N_spins = samples.shape
        self.samples = samples

        # Setting M1:
        self.M1 = M_tensor1
        self.A1u = self.M1[0]
        self.A1d = self.M1[1]
        self.chi1 = self.A1u.shape[0]

        self.A1uA1u = np.kron(self.A1u, self.A1u)
        self.A1dA1d = np.kron(self.A1d, self.A1d)
        self.E1 = self.A1uA1u + self.A1dA1d

        w1, vl1, vr1 = eig(self.E1, left=True, right=True)

        indices1 = np.argsort(np.abs(w1))
        ind1 = indices1[-1]
        self.lam1 = w1[ind1]
        if abs(np.imag(self.lam1)) > 1e-12:
            print('Complex lam1:', self.lam1)
            exit('Complex lam1')
        else:
            self.lam1 = np.real(self.lam1)
        self.R1 = np.real(vr1[:, ind1])  # is it valid?
        self.L1 = np.real(vl1[:, ind1])  # is it valid?
        self.LR1 = np.dot(self.L1, self.R1)
        self.xi1 = 1 / np.log(np.abs(w1[indices1[-1]] / w1[indices1[-2]]))

        # Setting M2:
        self.M2 = M_tensor2
        self.A2u = self.M2[0]
        self.A2d = self.M2[1]
        self.chi2 = self.A2u.shape[0]

        self.A2uA2u = np.kron(self.A2u, self.A2u)
        self.A2dA2d = np.kron(self.A2d, self.A2d)
        self.E2 = self.A2uA2u + self.A2dA2d

        w2, vl2, vr2 = eig(self.E2, left=True, right=True)

        indices2 = np.argsort(np.abs(w2))
        ind2 = indices2[-1]
        self.lam2 = w2[ind2]
        if abs(np.imag(self.lam2)) > 1e-12:
            print('Complex lam2:', self.lam2)
            exit('Complex lam2')
        else:
            self.lam2 = np.real(self.lam2)
        self.R2 = np.real(vr2[:, ind2])  # is it valid?
        self.L2 = np.real(vl2[:, ind2])  # is it valid?
        self.LR2 = np.dot(self.L2, self.R2)
        self.xi2 = 1 / np.log(np.abs(w2[indices2[-1]] / w2[indices2[-2]]))

        # to record node probabilities in decision tree:
        if self.N_spins <= 20:
            self.decision_prob_theoretical_1 = np.empty([self.N_spins, 2 ** (self.N_spins - 1), 2])  # site, node, spin
            self.decision_prob_theoretical_1.fill(np.nan)
            self.decision_prob_theoretical_2 = np.empty([self.N_spins, 2 ** (self.N_spins - 1), 2])  # site, node, spin
            self.decision_prob_theoretical_2.fill(np.nan)

            self.decision_count = np.zeros([self.N_spins, 2 ** (self.N_spins - 1), 2])  # site, node, spin

            self.decision_prob_empirical = np.empty([self.N_spins, 2 ** (self.N_spins - 1), 2])  # site, node, spin
            self.decision_prob_empirical.fill(np.nan)
        else:
            exit('N_spins > 20, stopping to avoid creating a large matrix')

        # Printing:
        if 0:
            print('Eigs_1:', w1)
            print('Lam1:', self.lam1, 'xi1:', self.xi1)
            print('Eigs_2:', w2)
            print('Lam2:', self.lam2, 'xi2:', self.xi2)
            print('')

        return

    #-------------------------------------------------------------------------------------------------------------------
    def compare(self):

        AuAuR_1 = np.matmul(self.A1uA1u, self.R1) / self.LR1
        AdAdR_1 = np.matmul(self.A1dA1d, self.R1) / self.LR1
        AuAuR_2 = np.matmul(self.A2uA2u, self.R2) / self.LR2
        AdAdR_2 = np.matmul(self.A2dA2d, self.R2) / self.LR2

        pus_1 = -np.ones([self.K_paths, self.N_spins])
        pus_2 = -np.ones([self.K_paths, self.N_spins])

        for k in range(self.K_paths):
            #if (k < 1000 and k % 100 == 0) or (k >= 1000 and k % 300 == 0):
            #    print('Processing path:', k)
            String_1 = self.L1
            String_2 = self.L2

            current_int_config = 0

            for n in range(self.N_spins):
                String_AuAuR_1 = np.matmul(String_1, AuAuR_1)
                String_AdAdR_1 = np.matmul(String_1, AdAdR_1)  # decreasing exponentially!? -> normalize
                #print('hah0:', k, 'n:', n, String_AuAuR_1, String_AdAdR_1, String_1)
                #print('hah1:', k, 'n:', n, String_AuAuR_1, String_AdAdR_1)
                #
                pu_1 = String_AuAuR_1 / (String_AuAuR_1 + String_AdAdR_1)
                self.decision_prob_theoretical_1[n, current_int_config, 1] = pu_1  # [site, node, decision] -> prob
                self.decision_prob_theoretical_1[n, current_int_config, 0] = 1 - pu_1  # [site, node, decision] -> prob

                String_AuAuR_2 = np.matmul(String_2, AuAuR_2)
                String_AdAdR_2 = np.matmul(String_2, AdAdR_2)  # decreasing exponentially!?
                #print('hah2:', k, 'n:', n, String_AuAuR_2, String_AdAdR_2)
                #
                pu_2 = String_AuAuR_2 / (String_AuAuR_2 + String_AdAdR_2)
                self.decision_prob_theoretical_2[n, current_int_config, 1] = pu_2  # [site, node, decision] -> prob
                self.decision_prob_theoretical_2[n, current_int_config, 0] = 1 - pu_2  # [site, node, decision] -> prob

                pus_1[k,n] = pu_1
                pus_2[k, n] = pu_2

                if self.samples[k, n] == 1:  # spin-up sampled
                    current_int_config = 2 * current_int_config + 1
                    String_1 = np.matmul(String_1, self.A1uA1u)
                    String_2 = np.matmul(String_2, self.A2uA2u)
                else:  # spin-down sampled
                    current_int_config = 2 * current_int_config
                    String_1 = np.matmul(String_1, self.A1dA1d)
                    String_2 = np.matmul(String_2, self.A2dA2d)
                if n % 10 == 0:
                    String_1 = String_1 / np.linalg.norm(String_1)  # safe and enough to renormalize here
                    String_2 = String_2 / np.linalg.norm(String_2)  # safe and enough to renormalize here

        #print('comparator: pus_1', pus_1)
        #print('comparator: pus_2', pus_2)

        return pus_1, pus_2

    #-------------------------------------------------------------------------------------------------------------------
    def visualize(self, pus_1, pus_2, noise):

        # decision tree probs:
        print('Comparator: Theoretical decision tree up_probs dictated by the M1 (target) tensor:')
        print(self.decision_prob_theoretical_1[:, :, 1].T)
        print('Comparator: Theoretical decision tree up_probs dictated by the M2 (model) tensor:')
        print(self.decision_prob_theoretical_2[:, :, 1].T)
        print('')

        # adding noise for visualization
        pus_1 = pus_1 + noise * np.random.randn(*pus_1.shape)
        pus_2 = pus_2 + noise * np.random.randn(*pus_2.shape)

        plt.figure(201, figsize=(4,4))
        plt.clf()
        colors = ['b', 'r', 'g', 'm']
        num_sample_to_visualize = 100
        for i in range(min(num_sample_to_visualize, pus_1.shape[0])):
            plt.plot(pus_1[i, :], pus_2[i, :], marker='.', linestyle='', color=colors[i % len(colors)])

        plt.plot([0,1], [0,1], 'r--')

        plt.xlabel('$p_{up}^{Target}$')
        plt.ylabel('$p_{up}^{Model}$')
        #plt.pause(0.1)

        return

#=======================================================================================================================
if __name__ == '__main__':

    print('\nInputs:  ##################################################################################################')
    # Input the dataset and the two MPS tensors to compare

    np.set_printoptions(precision=3)

    # Dataset:
    datafile = 'samples.npy'
    samples = np.load(datafile)
    print('samples.shape:', samples.shape, '\n')

    # MPS_1 (Target):
    if 0:
        file1 = 'MPS_calibrator_M_dump' #'MPS_M_dump' 'MPS_calibrator_M_dump_seed104'
        npzfile = np.load(file1 + '.npz')
        M_tensor1 = npzfile['M']
    elif 0:
        eps = 0.1
        Au = np.array([[1, 0], [eps, 0]])
        Ad = np.array([[0, eps], [0, 1]])
        M_tensor1 = np.array([Au, Ad])
    else:
        np.random.seed(104)
        Au = np.random.randn(2,2)
        Ad = np.random.randn(2,2)
        M_tensor1 = np.array([Au, Ad])
    print('M1 (Target):\n', M_tensor1[0], '\n', M_tensor1[1])
    print('M1.shape:', M_tensor1.shape)

    # MSP_2 (Model):
    if 1:
        file2 = 'MPS_calibrator_M_currentbest' #'MPS_M_dump' 'MPS_calibrator_M_dump_seed104'
        npzfile = np.load(file2 + '.npz')
        M_tensor2 = npzfile['M']
    elif 1:
        eps = 0.1
        Au = np.array([[1, 0], [eps, 0]])
        Ad = np.array([[0, eps], [0, 1]])
        M_tensor2 = np.array([Au, Ad])
    else:
        np.random.seed(104)
        Au = np.random.randn(2,2)
        Ad = np.random.randn(2,2)
        M_tensor2 = np.array([Au, Ad])
    print('M2 (Model):\n', M_tensor2[0], '\n', M_tensor2[1])
    print('M2.shape:', M_tensor2.shape)


    print('\nComparison:  ##############################################################################################')
    # Comparing two MPSs on the same sample to see how close they are in sampling probabilities

    comparator = MPS_comparator(M_tensor1, M_tensor2, samples)
    pus_1, pus_2 = comparator.compare()

    comparator.visualize(pus_1, pus_2, noise=0.05)

    plt.show()
    exit()
