"""
This script implements NMF with gradient descent
"""

"""
TODO

Make simultaneous

"""

import matplotlib.pyplot as plt
import numpy as np
# normalize imported in order to normalize the columns of the W matrix
from sklearn.preprocessing import normalize

class grad_NMF:
    """
    Implements update and API functions for simultaneous NMF
    """
    def __init__(self, As, n_comps, seed=None, n_iter=1, resid_thresh=1e-125, wO = 0.0, wS = 0.0, hO = 0.0, hS = 0.0,
                 alg = 'base', start = 'rand',
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
            wO (float) : how much to weight the W orthogonality term in the cost function
            wS (float) : how much to weight the W sparsity term in the cost function
            hO (float) : how much to weight the H orthogonality term in the cost function
            hS (float) : how much to weight the H sparsity term in the cost function
            alg (string) : Which update to perform. 
                                options are: 
                                'base' : (simultaneous NMF) 
                                'poisson' : (NMF using KL divergence)
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
        self.EPS = np.finfo(float).eps #save smallest non-zero float in python
        self.nonneg = self.EPS #value for reset if gradient descent goes negative
        
        #record weighting parameters
#         self.eta = eta
#         self.eta_hold = eta #an origional value for eta to remember for relaxation methods
        self.eta_decay_rate = 6.0
        self.eta_wO = wO
        self.eta_wS = wS
        self.eta_hO = hO
        self.eta_hS = hS
        self.test = 1 #dumb constant for testing updates

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
#         print(self.W_init)
#         print()
        ## save  intial orthogonality measure for W for later comparison
        self.W_orth = []
        self.W_orth.append(self.orth_measure(self.W))
        ## save intial sparsity measure for W for later comparison
        self.W_sparse = []
        self.W_sparse.append(np.linalg.norm(self.W))

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
#         print(self.Hs_init)
#         print()
        ## saves  intial orthogonality measures for Hs for later comparison
        self.Hs_orth = []
#         self.Hs_orth.append([self.orth_measure(h) for h in self.Hs])
        self.Hs_orth.append(self.orth_measure(self.Hs[0])) #only looks at Hs[0] for ease of coding
        ## save intial sparsity measure for Hs for later comparison
        self.Hs_sparse = []
#         self.Hs_sparse.append([np.linalg.norm(h) for h in self.Hs])
        self.Hs_sparse.append(np.linalg.norm(self.Hs[0]))
        
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
        elif ((alg == 'base') or (alg == 'poisson') or (alg == 'sorth_W') or (alg == 'norm_sorth_W') or (alg == 'sorth_H') or (alg == 'norm_sorth_H')):
            self.a_0s = [np.matrix(np.zeros(A.shape[0])).T for A in self.As]
            self.As_approx_init = []
        else:
            print(self.ALG_ERROR)
        ## a list to save residuals
        self.resids = [self.get_resids()]
        
        ## a list to save cost function
        self.cost_func_list = [self.get_cost_func()]
        
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
    
    def base_W_grad(self, A,W,H):
        """
        The gradient of the general cost function wrt W, multiplied by the appropriate factor of eta
        
            A (np.matrix) : A matrix factoring
            W (np.matrix) : current W matrix
            H (np.matrix) : current H matrix
            
            returns (np.matrix) : gradient of cost function wrt W
        """
        eta_factor = np.divide(self.W, self.W * (self.Hs[0] * self.Hs[0].T))
        return np.multiply(eta_factor, 
                           self.test * (W * (H * H.T) - A * H.T) + self.eta_wO * (W * (W.T * W) - W) + self.eta_wS * (W))
    
    def pois_W_grad(self, A,W,H):
        """
        The gradient of the KL divergence function wrt W
        
            A (np.matrix) : A matrix factoring
            W (np.matrix) : current W matrix
            H (np.matrix) : current H matrix
            
            returns (np.matrix) : gradient of cost function wrt W
        """
        eta_factor = np.divide(W, np.ones(A.shape) * H.T)
#         print("W eta factor is {}".format(eta_factor))
#         print("first part of W gradient is {}".format((np.divide(A, W * H) * H.T)))
#         print('second part of W gradient is {}'.format(np.ones(A.shape) * H.T))
#         print("W gradient is {}".format(self.test * (-np.divide(A, W * H) * H.T + np.ones(A.shape) * H.T)
#                 + self.eta_wO * (W * (W.T * W) - W) + self.eta_wS * (W)))
#         print()
        return np.multiply(eta_factor, self.test * (-np.divide(A, W * H) * H.T + np.ones(A.shape) * H.T)
                + self.eta_wO * (W * (W.T * W) - W) + self.eta_wS * (W))

    """
    H update functions
    """
    
    def base_H_grad(self, A,W,H):
        """
        The gradient of the general cost function wrt H
        
            A (np.matrix) : A matrix factoring
            W (np.matrix) : current W matrix
            H (np.matrix) : current H matrix
            
            returns (np.matrix) : gradient of cost function wrt H
        """
        eta_factor = np.divide(H, self.W.T * self.W * H)
        return np.multiply(eta_factor, 
                           self.test * (W.T * W * H - W.T * A) + self.eta_hO * (H * H.T * H - H) + self.eta_hS * (H))
    
    def pois_H_grad(self, A,W,H):
        """
        The gradient of the KL divergence wrt H
        
            A (np.matrix) : A matrix factoring
            W (np.matrix) : current W matrix
            H (np.matrix) : current H matrix
            
            returns (np.matrix) : gradient of cost function wrt H
        """
        eta_factor = np.divide(H, W.T * np.ones(A.shape))
#         print("numerator of H eta factor is {}".format(H))
#         print()
#         print("denominator of H eta factor is {}".format(W.T * np.ones(A.shape)))
#         print()
#         print("H eta factor is {}".format(eta_factor))
#         print("H gradient is {}".format(self.test * (W.T * np.divide(A, W * H) - W.T * np.ones(A.shape))
#                 + self.eta_hO * (H * H.T * H - H) + self.eta_hS * (H)))
#         print()
        return np.multiply(eta_factor, self.test * (-W.T * np.divide(A, W * H) + W.T * np.ones(A.shape))
                + self.eta_hO * (H * H.T * H - H) + self.eta_hS * (H))

    """
    a_0 update functions
    """

    
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
        return [0.5 * f(self.As[i] - self.W*self.Hs[i])
                for i in range(len(self.As))]
    
    def get_aff_resids(self):
        """
        Returns a list of all residuals, one corresponding to each
        H matrix in self.Hs, in the affine  setting
        
            returns ([float]): List of residuals
        """
        ## Save space, define norm as lambda
        f = lambda m: np.linalg.norm(m, ord='fro')
        return [0.5 * f(self.As[i] - self.As_approx[i])
                for i in range(len(self.As))]
    
    def get_cost_func(self):
        """
        Calculates the cost function
        """
        f = lambda m: np.linalg.norm(m, ord='fro')
        return [0.5 * (f(self.As[i] - self.W * self.Hs[i]) 
                + self.eta_wO * f(self.W.T * self.W - np.identity(self.n_comps)) + self.eta_wS * f(self.W * self.W.T)
                + self.eta_hO * f(self.Hs[i] * self.Hs[i].T - np.identity(self.n_comps)) 
                + self.eta_hS * f(self.Hs[i] * self.Hs[i].T))
                for i in range(len(self.As))]
    
    def has_converged(self):
        """
        Checks if the NMF has converged according to the stopping 
        criteria imposed by self.resid_thresh
        
            returns (bool): True if stopping criteria has been reached
        """
        prev = np.mean(self.cost_func_list[-2])
        curr = np.mean(self.cost_func_list[-1])
        init = np.mean(self.cost_func_list[0])
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
#         print("ran {} iterations".format(iter_numb))
        return(iter_numb)
        
    def individual_run(self, W_grad, H_grad):
        """
        The abstracted core of the NMF update.
        
            W_grad (np.matrix) : d Cost / d W
            H_grad (np.matrix) : d Cost / d H
        
        returns (int) : Number of iterations performed
        """
        for i in range(self.n_iter):
            self.count += 1
#             print("iteration number {}".format(self.count))
#             self.eta = self.eta_hold / np.sqrt(self.count)
#             self.eta = 1000 / (1000 + self.eta_decay_rate * self.count)
#             print(self.eta)
            self.As_approx = [self.W * self.Hs[i] + self.a_0s[i] * np.matrix(np.ones(self.As[i].shape[1])) 
                              for i in range(len(self.As))]
#             print(self.a_0s)
#             print(self.As_approx)
#             print()
#             print("W is {}".format(self.W))
#             print()
            self.W = self.W - (W_grad(self.As[0], self.W, self.Hs[0]))
            self.W[self.W < 0] = self.nonneg
            self.W_orth.append(self.orth_measure(self.W))
            self.W_sparse.append(np.linalg.norm(self.W))
#             print(self.orth_measure(self.W))
#             print()
            for j in range(len(self.Hs)):
#                 print("H is {}".format(self.Hs[j]))
#                 print()
                self.Hs[j] = self.Hs[j] - (H_grad(self.As[0], self.W, self.Hs[0]))
                self.Hs[j][self.Hs[j] < 0] = self.nonneg
#                 self.a_0s[j] = self.mult_update(self.a_0s[j], self.As[j], self.As_approx[j],self.a_0s[j],self.Hs[j], 
#                                                 a_0_numer, a_0_denom, a_0_norm)
#             self.Hs_orth.append([self.orth_measure(h) for h in self.Hs])
#             self.Hs_sparse.append([np.linalg.norm(h) for h in self.Hs])
                self.Hs_orth.append(self.orth_measure(self.Hs[0]))
                self.Hs_sparse.append(np.linalg.norm(self.Hs[0]))
#             print("residuals are")
#             print(self.get_aff_resids())
#             print()
#             print("cost function is")
#             print(self.get_cost_func())
#             print()
#             print(self.W)
#             print()
            self.resids.append(self.get_aff_resids())
            self.cost_func_list.append(self.get_cost_func())
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
            self.individual_run(self.base_W_grad, self.base_H_grad)
        elif self.alg == 'poisson':
            self.individual_run(self.pois_W_grad, self.pois_H_grad)
#         elif self.alg == 'aff':
#             self.individual_run(None, self.numer_base_W, self.denom_aff_W,
#                                 None, self.numer_base_H, self.denom_aff_H, 
#                                 None, self.numer_a_0, self.denom_a_0)
#         elif self.alg == 'sorth_W':
#             self.individual_run(None, self.numer_base_W, self.denom_sorth_W, 
#                                 None, self.numer_base_H, self.denom_base_H, 
#                                 None, None, None)
#         elif self.alg == 'norm_sorth_W':
#             self.individual_run(w_norm, self.numer_base_W, self.denom_sorth_W, 
#                                 None, self.numer_base_H, self.denom_base_H,
#                                 None, None, None)
#         elif self.alg == 'aff_sorth_W':
#             self.individual_run(w_norm, self.numer_aff_sorth_W, self.denom_aff_sorth_W, 
#                                 None, self.numer_base_H, self.denom_aff_H,
#                                 None, self.numer_a_0, self.denom_a_0)
#         elif self.alg == 'sorth_H':
#             self.individual_run(None, self.numer_base_W, self.denom_base_W,
#                                 None, self.numer_base_H, self.denom_sorth_H, 
#                                 None, None, None)
#         elif self.alg == 'norm_sorth_H':
#             self.individual_run(None, self.numer_base_W, self.denom_base_W,
#                                 h_norm, self.numer_base_H, self.denom_sorth_H,
#                                 None, None, None)
#         elif self.alg == 'aff_sorth_H':
#             self.individual_run(None, self.numer_base_W, self.denom_aff_W,
#                                 h_norm, self.numer_aff_sorth_H, self.denom_aff_sorth_H,
#                                 None, self.numer_a_0, self.denom_a_0)
        else:
            print(self.ERROR_MESSAGE)

if __name__ == "__main__":
    print("additive_NMF.py")

    test_mat = np.matrix([
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ])
    
#     updates = ['base', 'aff', 'sorth_W', 'norm_sorth_W', 'aff_sorth_W', 'sorth_H', 'norm_sorth_H', 'aff_sorth_H']
#     updates = ['norm_sorth_W', 'aff_sorth_W', 'sorth_H', 'norm_sorth_H', 'aff_sorth_H']
#     updates = ['sorth_H', 'norm_sorth_H']
#     updates = ['base']
#     updates = ['base','poisson']
    updates = ['poisson']
#     updates = ['aff', 'aff_sorth_W', 'aff_sorth_H']
    
#     starts = ['rand', 'sorth_W', 'sorth_H']
#     starts = ['rand', 'sorth_H']
    starts = ['rand']
    
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
                for WO in np.arange(0,1):
                    for WS in np.arange(0,1):
                        for HO in np.arange(0,1):
                            for HS in np.arange(0,1):
                                print()
                                print("***")
                                print("UPDATE IS " + str(u) + "; START IS " + str(s))
                                print("***")
                    #             print(len(snmf.resids))
                                print()
                                print("W orthogonality is {}".format(WO))
                                print("W sparsity is {}".format(WS))
                                print("H orthogonality is {}".format(HO))
                                print("H sparsity is {}".format(HS))
                                nmf = grad_NMF([test_mat], n_comps=2, n_iter=100, alg = u, start = s, wO = WO, wS = WS, 
                                               hO = HO, hS = HS)
#                                 print("step size is {}".format(nmf.eta))
            #                     print(nmf.count)
                                print()
                                print(nmf.W * nmf.Hs[0])
                                print()
            #         #                 print(snmf.As_approx[0])
            #         #                 print()
            #         #             print(snmf.a_0s[0] * np.matrix(np.ones(snmf.As[0].shape[1])))
            #         #             print()
            #                 #     print(snmf.a_0s[0])
            #                 #     print()
            #                 #     print(snmf.a_0s_init[0])
            #                 #     print()
                                print(nmf.W)
                                print()
                            #         print(snmf.W.T * snmf.W)
                            #         print()
                    #                 print(nmf.W_init)
                    #                 print()
                            #         print(snmf.W_init.T * snmf.W_init)
                            #         print()
                                print(nmf.Hs[0])
                                print()
                            #         print(snmf.Hs[0] * snmf.Hs[0].T)
                    #                 print(nmf.Hs_init[0])
                    #                 print()
                            #         print(snmf.Hs_init[0] * snmf.Hs_init[0].T)
                            #         print()
#                                 plt.plot(nmf.resids)
#                                 plt.xlabel('iterations')
#                                 plt.ylabel('residual')
#                                 plt.title(str(nmf.alg) + ' residuals graph')
#                                 plt.show()
#                                 print()
#                                 plt.plot(nmf.cost_func_list)
#                                 plt.xlabel('iterations')
#                                 plt.ylabel('cost function')
#                                 plt.title(str(nmf.alg) + ' cost function graph')
#                                 plt.show()
#                                 print()
#                                 plt.plot(nmf.W_orth)
#                                 plt.xlabel('iterations')
#                                 plt.ylabel('W orthogonality')
#                                 plt.title(str(nmf.alg) + ' W orthogonality graph')
#                                 plt.show()
#                                 print()
#                                 plt.plot(nmf.W_sparse)
#                                 plt.xlabel('iterations')
#                                 plt.ylabel('W sparsity')
#                                 plt.title(str(nmf.alg) + ' W sparsity graph')
#                                 plt.show()
#                                 print()
#                                 plt.plot(nmf.Hs_orth)
#                                 plt.xlabel('iterations')
#                                 plt.ylabel('H orthogonality')
#                                 plt.title(str(nmf.alg) + ' H orthogonality graph')
#                                 plt.show()
#                                 print()
#                                 plt.plot(nmf.Hs_sparse)
#                                 plt.xlabel('iterations')
#                                 plt.ylabel('H sparsity')
#                                 plt.title(str(nmf.alg) + ' H sparsity graph')
#                                 plt.show()
#                                 print()
#                                 print('FINISHED')