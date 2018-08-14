"""
This script implements Lee et al.'s 2012 version of simultaneous NMF. Most
notably, this version alters the update steps described in their paper such
that a single W matrix is maintained for multiple H matrices (each H matrix
corresponds to one of the input matrices). Different keyword inputs result 
in different types of NMF
"""

"""
TODO

Implementation of sorth_H; aff_sorth_H strange (but update rule is fine)
    sorth_H is totally unstable (never goes below threshold)
    aff_sorth_H sometimes gets too small values

What is going on with residual? (math proof that is monotonically decreasing, but sometimes not in code)
    (like, what is going on with the residuals when I run aff_sorth_W multiple times?)

Implement cost term for orthogonal instead of being either orthogonanl or not

Create a "restart function"

Could implement affine sorth NMF w/o normalizing

Would be nice to find way to assess how similar or different results of given factorization are (specifically normalized
and unnormalized sorth, but all really)

Visualize a_0 vectors in affine cases

"""

import matplotlib.pyplot as plt
import numpy as np
# normalize imported in order to normalize the columns of the W matrix
from sklearn.preprocessing import normalize

class SNMF:
    """
    Implements update and API functions for simultaneous NMF
    """
    def __init__(self, As, n_comps, seed=None, n_iter=1, resid_thresh=1e-125, alg = 'base', start = 'rand',
                 debug=False):
        """
        Stores the input matrices and starts the algorithm

            As ([array-like]): List of input matrices to factor - are
                                converted to np.matrix for consistency
            n_comps (int): Number of components to factor
            seed (int|None): Seed for input to numpy.random.RandomState
                                construction. If None, no seed is used
            n_iter (int): Number of iterations to perform
            resid_thresh (float): Threshold value for the ratio of the 
                                  difference in recent errors to the original
                                  residual of the factored matrix.
            alg (string) : Which update to perform. 
                                options are: 
                                'base' : (simultaneous NMF) 
                                'aff' : (affine simultaneous NMF)
                                'sorth_W' : (simultaneous NMF with semi-orthogonal W)
                                'sorth_H : (simultaneous NMF with semi-orthogonal H)
                                'norm_sorth_W' : (simultaneous NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'norm_sorth_H' : (simultaneous NMF with semi-orthogonal H
                                                where rows of H normalized every iteration)
                                'aff_sorth_W' : (simultaneous affine NMF with semi-orthogonal W
                                                where columns of W normalized every iteration)
                                'aff_sorth_H' : (simultaneous affine NMF with semi-orthogonal H
                                                where rows of H normalized every iteration)
            start (string) : how to initialize matrices.
                                options are:
                                'rand' : random non-negative initialization
                                'sorth_W' : semi-orthogonal W
                                'sorth_H' : semi-orthogonal H
            debug (bool): If True, don't call self.run_snmf() on init

            returns: None 
        """
#         print('starting')
        ## Error message listing options for alg
        self.count = 0
        self.ALG_ERROR = 'Alg Error: options are base; aff; sorth_W; sorth_H; norm_sorth_W; norm_sorth_H; aff_sorth_W; aff_sorth_H'
        ## Error messgage listing options for start
        self.START_ERROR = 'Start Error: options are rand; sorth_W; sorth_H'
        
        ## make sure all inputs are numpy matrices
        self.As = list(map(np.matrix, As))
        self.d = len(self.As) #save lenth of As
        self.m1 = self.As[0] #save first entry of As matrix for betas
        self.bs = [self.beta(self.As[i], self.m1) for i in range(self.d)] #save betas for simultaneous NMF
        self.n_comps = n_comps  #save num components to factor
        self.seed = seed  #save random seed
        self.n_iter = n_iter  #save num iterations
        self.n_iter_done = 0
        self.resid_thresh = resid_thresh  #save resid_thresh
        self.debug = debug  #save debug flag
        self.alg = alg #save alg
        self.start = start #save start

        ## generate a random state based on seed
        self.state = np.random.RandomState(self.seed)

        ## generate init random matrix for W
        n_w_rows = self.As[0].shape[0]  #all As have same #rows
        if ((self.start == 'rand') or (self.start == 'sorth_H')):
            self.W = self.get_rand_nn_mat(n_w_rows, self.n_comps)
        elif ((self.start == 'sorth_W')):
            self.W = self.get_rand_nn_sorth_mat(n_w_rows, self.n_comps)
        else:
            print(self.START_ERROR)
        ## saves initial random W for later comparison with W generated by NMF
        self.W_init = self.W
        ## save  intial orthogonality measure for W for later comparison
        self.W_orth = []
        self.W_orth.append(self.orth_measure(self.W))

        ## generate init random matrices for H
        self.Hs = []
        if ((self.start == 'rand') or  (self.start == 'sorth_W')):
            for A in self.As:
                n_h_cols = A.shape[1]
                self.Hs.append(self.get_rand_nn_mat(self.n_comps, n_h_cols))
        elif (self.start == 'sorth_H'):
            for A in self.As:
                n_h_cols = A.shape[1]
                self.Hs.append(self.get_rand_nn_sorth_mat(n_h_cols, self.n_comps))
        else:
            print(self.START_ERROR)
        ## saves initial random Hs for later comparison with Hs generated by NMF
        self.Hs_init = []
        for h in self.Hs:
            self.Hs_init.append(h)
        ## saves  intial orthogonality measures for Hs for later comparison
        self.Hs_orth = []
        self.Hs_orth.append([self.orth_measure(h) for h in self.Hs])
        
        if ((alg == 'aff') or (alg == 'aff_sorth_W') or (alg == 'aff_sorth_H')):
        ## generate init random vector for a
            self.a_0s = []
            for A in self.As:
                n_h_rows = A.shape[0]
                self.a_0s.append(self.get_rand_nn_mat(n_h_rows, 1))
            ## saves initial random a_0s for later comparison
            self.a_0s_init = []
            for a in self.a_0s:
                self.a_0s_init.append(a)
            

            ## generate first approximation for A
            ##  Note that this step has the potential to be very slow
            self.As_approx = [self.W * self.Hs[i] + self.a_0s[i] * np.matrix(np.ones(self.As[i].shape[1])) for i in range(len(self.As))]
            ## saves initial "approximation" of A for later comparison 
            self.As_approx_init = []
            for a in self.As_approx:
                self.As_approx_init.append(a)
        elif ((alg == 'base') or (alg == 'sorth_W') or (alg == 'norm_sorth_W') or (alg == 'sorth_H') or (alg == 'norm_sorth_H')):
            self.a_0s = [np.matrix(np.zeros(A.shape[0])).T for A in self.As]
            self.As_approx_init = []
        else:
            print(self.ALG_ERROR)
        ## a list to save residuals
        self.resids = [self.get_resids()]
        
        ## lists to save norms of matrices
        self.W_norms = [np.linalg.norm(self.W)]
        self.H_norms = [np.linalg.norm(self.Hs[0])]

        ## Run update loop
        if not self.debug:
            self.n_iter_done = self.run_snmf()

    """
    Initialization Functions
    """

    def get_rand_nn_mat(self, n_rows, n_cols):
        """
        Builds a random matrix with n_rows by n_cols dimensions
        containing all non-negative values

            n_rows (int): Number of rows in the matrix
            n_cols (int): Number of columns in the matrix

            returns (np.matrix): New random non-negative matrix
        """
        return np.matrix(abs(self.state.randn(n_rows, n_cols)))
    
    def get_rand_nn_sorth_mat(self, long, k):
        """
        Depending on alg, builds random non-negative matrix W (long x k) s.t.
            W.T * W = I or H (k x long) s.t. H * H^T = I.

            n_rows (int): Number of rows in the matrix
            n_cols (int): Number of columns in the matrix

            returns (np.matrix): New random non-negative semi-orthogonal matrix
        """
        tst_ary = np.matrix(np.zeros((long,k)))
        for i in range(long):
            if i < k:
                # makes sure no orthogonal vectors have norm zero
                tst_ary[i,i] = abs(np.random.randn()) + 0.01
            else:
                key = np.random.randint(0,k)
                tst_ary[i,key] = abs(np.random.randn()) + 0.01
        np.random.shuffle(tst_ary) #makes order random
        tst_ary = tst_ary.T
        for i in range(len(tst_ary)):
            tst_ary[i] = tst_ary[i] / np.linalg.norm(tst_ary[i]) #normalizes each orthgonal vector
        if (self.start == 'sorth_H'):
            return tst_ary
        elif (self.start == 'sorth_W'):
            return tst_ary.T
        else:
            print(self.START_ERROR)
            
    """
    Helper Functions
    """
    def orth_measure(self, M):
        """
        Measures the semi-orthogonality of a matrix M according to
        the formula O = |M^T / |M^T| - M^pinv \ |M^\pinv| |
            
            M (np.matrix) : matrix to measure orthogonality of
            
            returns (float) : measurement of orthogonality, 0 is orthogonal
                and higher is less orthogonal
        """
        M_pinv = np.linalg.pinv(M)
        return np.linalg.norm((M.T / np.linalg.norm(M.T)) - (M_pinv / np.linalg.norm(M_pinv)))

    """
    Beta Weights
    """

    def beta(self, mi, m1):
        """
        Computes the beta weighting for the given matrix mi using
        the baseline matrix m1

            mi (np.matrix): Matrix for which to find the beta weight
            m1 (np.matrix): First/baseline matrix for calculating weights

            returns (float): The beta weighting coefficient for mi
        """
        return np.linalg.norm(m1, ord='fro')/np.linalg.norm(mi, ord='fro')

    """
    Update Functions
    """
    def mult_update(self, T, A, A_approx, a_0, H, numer_exp, denom_exp, normalizer):
        """
        The factored core of multiplicative update

            T (np.matrix) : matrix to update
            A (np.matrix): corresponding A matrix
            A_approx (np.matrix): Current approximation to A
            a_0 (np.matrix): Current offset
            H (np.matrix): current H matrix
            numer_exp ((np.matrix * np.matrix * np.matrix) -> np.matrix) : function for numerator in update
            denom_exp ((np.matrix * np.matrix * np.matrix * np.matrix) -> np.matrix) -> np.matrix) :
                                                                            function for denominator in update
            normalizer (np.matrix -> np.matrix) : normalizes rows or columns of matrix as appropriate
                    
            returns (np.matrix): updated T
        """
        if normalizer != None:
            self.count += 1
            try:
                T = normalizer(T)
            except ValueError:
                print("Algorithm is " + self.alg + " ; start is " + self.start)
                print("Values in matrix got too small; restarting")
                if ((self.alg == 'aff_sorth_H') and (self.start == 'sorth_W')):
#                     print(self.a_0s)
#                     print(T)
#                     print(self.W)
#                     print(self.As_approx)
#                     print()
                    (numb_rows, numb_cols) = T.shape
                    T = self.get_rand_nn_mat(numb_rows, numb_cols)
#                     print(T)
                    (numb_rows, numb_cols) = self.W.shape
                    self.W = self.get_rand_nn_sorth_mat(numb_rows, numb_cols)
#                     print(self.W)
                    self.a_0s = []
                    for A in self.As:
                        n_h_rows = A.shape[0]
                        self.a_0s.append(self.get_rand_nn_mat(n_h_rows, 1))
#                     print(self.a_0s)
                    self.As_approx = self.As_approx_init
                else:
                    print("An unexpected error occured")
            if ((self.alg == 'norm_sorth_W') or (self.alg == 'aff_sorth_W')):
                self.W = T
            elif ((self.alg == 'norm_sorth_H') or (self.alg == 'aff_sorth_H')):
                H = T
            else:
                print(self.ALG_ERROR)
        if numer_exp == None:
            return T
        numer = numer_exp(A, a_0, H)
        denom = denom_exp(A, A_approx, a_0, H)
        return np.multiply(T, np.divide(numer, denom))
    
    """
    W update functions
    """
    
    def numer_base_W(self, A, a_0, H):
        """
        Returns the numerator for the expression in the base W update
        """
        return sum([self.bs[i]*self.As[i]*self.Hs[i].T for i in range(self.d)])
    
    def numer_aff_sorth_W(self, A, a_0, H):
        """
        Returns the numerator for the expression for the sorth W in the affine NMF update
        """
        return sum([self.bs[i] * (self.As[i] - self.a_0s[i] * np.matrix(np.ones(self.As[i].shape[1]))) * self.Hs[i].T 
                    for i in range(self.d)])
    
    def denom_base_W(self, A, A_approx, a_0, H):
        """
        Returns the denominator for the expression in the base W update
        """
        return sum([self.bs[i] * self.W * (self.Hs[i] * self.Hs[i].T) for i in range(self.d)])
    
    def denom_aff_W(self, A, A_approx, a_0, H):
        """
        Returns the denominator for the expression in the affine NMF W update
        """
        return sum([self.bs[i] * self.As_approx[i] * self.Hs[i].T for i in range(self.d)])
    
    def denom_sorth_W(self, A, A_approx, a_0, H):
        """
        Returns the denominator for the expression in the sorth W update
        """
        return sum([self.bs[i] * self.W * (self.Hs[i] * (self.As[i].T * self.W)) for i in range(self.d)])
    
    def denom_aff_sorth_W(self, A, A_approx, a_0, H):
        """
        Returns the denominator for the expression in the affine NMF sorth W update
        """
        return sum([self.bs[i] * self.W * (self.Hs[i] * ((self.As[i] - self.a_0s[i] * np.matrix(np.ones(self.As[i].shape[1]))).T 
                                                         * self.W)) for i in range(self.d)])

    """
    H update functions
    """
    
    def numer_base_H(self, A, a_0, H):
        """
        Returns the numerator for the expression in the base H update
        """
        return self.W.T * A
    
    def numer_aff_sorth_H(self, A, a_0, H):
        """
        Returns the numerator for the expression for the sorth H in the affine NMF
        """
        return self.W.T * (A - a_0 * np.matrix(np.ones(A.shape[1])))
    
    def denom_base_H(self, A, A_approx, a_0, H):
        """
        Returns the denominator for the expression in the base H update
        """
        return (self.W.T * self.W) * H
    
    def denom_aff_H(self, A, A_approx, a_0, H):
        """
        Returns the denominator for the expression for H in the affine NMF update
        """
        return self.W.T * A_approx
    
    def denom_sorth_H(self, A, A_approx, a_0, H):
        """
        Returns the denominator for the expression in the sorth H update
        """
        return (H * (A.T * self.W)) * H
    
    def denom_aff_sorth_H(self, A, A_approx, a_0, H):
        """
        Returns the denominator for the expression for the sorth H in the affine NMF update
        """
        return H * ((A - a_0 * np.matrix(np.ones(A.shape[1]))).T * self.W) * H

    """
    a_0 update functions
    """
    
    def numer_a_0(self, A, a_0, H):
        """
        Returns updated numerator of a_0
        """
        return A * np.matrix(np.ones(A.shape[1])).T
    
    def denom_a_0(self, A, A_approx, a_0, H):
        """
        Returns updated denominator of a_0
        """
        return A_approx * np.matrix(np.ones(A.shape[1])).T
    
    """
    Residual/Convergence
    """
    
    def get_resids(self):
        """
        Returns a list of all residuals, one corresponding to each
        H matrix in self.Hs
        
            returns ([float]): List of residuals
        """
        ## Save space, define norm as lambda
        f = lambda m: np.linalg.norm(m, ord='fro')
        return [f(self.As[i] - self.W*self.Hs[i])
                for i in range(len(self.As))]
    
    def get_aff_resids(self):
        """
        Returns a list of all residuals, one corresponding to each
        H matrix in self.Hs, in the affine  setting
        
            returns ([float]): List of residuals
        """
        ## Save space, define norm as lambda
        f = lambda m: np.linalg.norm(m, ord='fro')
        return [f(self.As[i] - self.As_approx[i])
                for i in range(len(self.As))]
    
    def has_converged(self):
        """
        Checks if the NMF has converged according to the stopping 
        criteria imposed by self.resid_thresh
        
            returns (bool): True if stopping criteria has been reached
        """
        prev = np.mean(self.resids[-2])
        curr = np.mean(self.resids[-1])
        init = np.mean(self.resids[0])
        stop_ratio = np.abs((prev-curr)/init)
        return stop_ratio < self.resid_thresh

    """
    Loop
    """
    def wrap_up(self, iter_numb):
        """
        Desired behavior after convergence
        
            iter_numb (int) : number of iterations completed
            
            returns (int) iter_numb
        """
        self.W_norms.append(np.linalg.norm(self.W))
        self.H_norms.append(np.linalg.norm(self.Hs[0]))
#         self.residual_graph = plt.plot(self.resids)
#         plt.xlabel('iterations')
#         plt.ylabel('residual')
#         plt.title(str(self.alg) + ' residuals graph')
        self.W
#         print("ran {} iterations".format(iter_numb))
        return(iter_numb)
        
    def individual_run(self, W_norm, W_numer, W_denom, H_norm, H_numer, H_denom, a_0_norm, a_0_numer, a_0_denom):
        """
        The abstracted core of the NMF update.
        
            W_norm (np.matrix -> np.matrix) : desired normalization for W
            W_numer ((np.matrix * np.matrix * np.matrix) -> np.matrix) : function for numerator in desired update of W
            W_denom ((np.matrix * np.matrix * np.matrix * np.matrix ) -> np.matrix) : function for denominator in desired
                                                                                                            updated of W
            H_norm (np.matrix -> np.matrix) : desired normalization for H
            H_numer ((np.matrix * np.matrix * np.matrix) -> np.matrix) : function for numerator in desired update of H
            H_denom ((np.matrix * np.matrix * np.matrix * np.matrix ) -> np.matrix) : function for denominator in desired
                                                                                                            updated of H
            a_0_norm (np.matrix -> np.matrix) : desired normalization for a_0
            a_0_numer ((np.matrix * np.matrix * np.matrix) -> np.matrix) : function for numerator in desired update of a_0
            a_0_denom ((np.matrix * np.matrix * np.matrix * np.matrix ) -> np.matrix) : function for denominator in desired
                                                                                                            updated of a_0
        """
        for i in range(self.n_iter):
            self.As_approx = [self.W * self.Hs[i] + self.a_0s[i] * np.matrix(np.ones(self.As[i].shape[1])) 
                              for i in range(len(self.As))]
            self.W = self.mult_update(self.W, None, None, None, None,W_numer, W_denom, W_norm)
            self.W_orth.append(self.orth_measure(self.W))
            for j in range(len(self.Hs)):
                self.Hs[j] = self.mult_update(self.Hs[j], self.As[j], self.As_approx[j], self.a_0s[j], self.Hs[j], 
                                              H_numer, H_denom, H_norm)
                self.a_0s[j] = self.mult_update(self.a_0s[j], self.As[j], self.As_approx[j],self.a_0s[j],self.Hs[j], 
                                                a_0_numer, a_0_denom, a_0_norm)
            self.Hs_orth.append([self.orth_measure(h) for h in self.Hs])
            self.resids.append(self.get_aff_resids())
            if self.has_converged():
                self.wrap_up(i)
        self.wrap_up(i)
    
    def run_snmf(self):
        """
        Runs individual_run with functions needed for desired algorithm

        returns (int): Number of iterations performed
        """
#         print('starting update')
        w_norm = lambda W: np.matrix(normalize(W, norm='l2', axis=0))
        h_norm = lambda H: np.matrix(normalize(H, norm='l2', axis=1))
        if self.alg == 'base':
            self.individual_run(None, self.numer_base_W, self.denom_base_W, 
                                None, self.numer_base_H, self.denom_base_H, 
                                None, None, None)
        elif self.alg == 'aff':
            self.individual_run(None, self.numer_base_W, self.denom_aff_W,
                                None, self.numer_base_H, self.denom_aff_H, 
                                None, self.numer_a_0, self.denom_a_0)
        elif self.alg == 'sorth_W':
            self.individual_run(None, self.numer_base_W, self.denom_sorth_W, 
                                None, self.numer_base_H, self.denom_base_H, 
                                None, None, None)
        elif self.alg == 'norm_sorth_W':
            self.individual_run(w_norm, self.numer_base_W, self.denom_sorth_W, 
                                None, self.numer_base_H, self.denom_base_H,
                                None, None, None)
        elif self.alg == 'aff_sorth_W':
            self.individual_run(w_norm, self.numer_aff_sorth_W, self.denom_aff_sorth_W, 
                                None, self.numer_base_H, self.denom_aff_H,
                                None, self.numer_a_0, self.denom_a_0)
        elif self.alg == 'sorth_H':
            self.individual_run(None, self.numer_base_W, self.denom_base_W,
                                None, self.numer_base_H, self.denom_sorth_H, 
                                None, None, None)
        elif self.alg == 'norm_sorth_H':
            self.individual_run(None, self.numer_base_W, self.denom_base_W,
                                h_norm, self.numer_base_H, self.denom_sorth_H,
                                None, None, None)
        elif self.alg == 'aff_sorth_H':
            self.individual_run(None, self.numer_base_W, self.denom_aff_W,
                                h_norm, self.numer_aff_sorth_H, self.denom_aff_sorth_H,
                                None, self.numer_a_0, self.denom_a_0)
        else:
            print(self.ERROR_MESSAGE)

if __name__ == "__main__":
    print("SNMF.py")

    test_mat = np.matrix([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    
    updates = ['base', 'aff', 'sorth_W', 'norm_sorth_W', 'aff_sorth_W', 'sorth_H', 'norm_sorth_H', 'aff_sorth_H']
#     updates = ['norm_sorth_W', 'aff_sorth_W', 'sorth_H', 'norm_sorth_H', 'aff_sorth_H']
#     updates = ['sorth_H', 'norm_sorth_H']
#     updates = ['norm_sorth_W']
#     updates = ['aff', 'aff_sorth_W', 'aff_sorth_H']
    
    starts = ['rand', 'sorth_W', 'sorth_H']
#     starts = ['rand', 'sorth_H']
#     starts = ['sorth_W']
    
#     for i in range(1,50):
#         print()
#         print("***")
#         print("Iteration" + str(i))
#         snmf = snmf = SNMF([test_mat], n_comps=2, n_iter=2000, alg = 'aff_sorth_H', start = 'rand')
#         print("***")
#         print()
    for i in range(1,2):
        for u in updates:
            for s in starts:
                print()
                print("***")
                print("UPDATE IS " + str(u) + "; START IS " + str(s))
                print("***")
    #             print(len(snmf.resids))
                print()
                snmf = SNMF([test_mat], n_comps=2, n_iter=2000, alg = u, start = s)
#                 print(snmf.W * snmf.Hs[0])
#                 print()
#                 print(snmf.As_approx[0])
#                 print()
#     #             print(snmf.a_0s[0] * np.matrix(np.ones(snmf.As[0].shape[1])))
#     #             print()
#             #     print(snmf.a_0s[0])
#             #     print()
#             #     print(snmf.a_0s_init[0])
#             #     print()
#                 print(snmf.W)
#                 print()
            #         print(snmf.W.T * snmf.W)
            #         print()
            #         print(snmf.W_init)
            #         print()
            #         print(snmf.W_init.T * snmf.W_init)
            #         print()
#                 print(snmf.Hs[0])
#                 print()
            #         print(snmf.Hs[0] * snmf.Hs[0].T)
#                 print(snmf.Hs_init[0])
#                 print()
            #         print(snmf.Hs_init[0] * snmf.Hs_init[0].T)
            #         print()
#                 print(snmf.W_orth[0])
#                 print()
#                 print(snmf.W_orth[-1])
#                 print()
#                 print(snmf.Hs_orth[0])
#                 print()
#                 print(snmf.Hs_orth[-1])
#                 print()
                print('FINISHED')