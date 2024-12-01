# Necessary libraries
import numpy as np
class amp:
  def __init__(self, L, B):
    np.random.seed(0)
    self.L = L
    self.B = B
    self.N = L * B
    self.x = self.gen_ss()

  def awgn_ch(self, snr):
    self.snr = snr
    self.M = int((self.N * np.log2(self.B)) / (self.R * self.B))
    std = 1 / np.sqrt(snr)
    x_gt = self.x.reshape(self.N)
    F = np.random.normal(0, 1/np.sqrt(self.L), (self.M, self.N))
    self.F = F
    noise = std * np.random.randn(self.M)
    y = (F @ x_gt) + noise
    return y

  def gen_ss(self):
    x = np.zeros((self.L, self.B))
    for arr in x:
      randy = np.random.randint(0, self.B)
      arr[randy] = 1
    return x

  def g(self, z, y, V, snr):
    '''
    A function to be described
    Args:
    z (numpy.array): Cavity mean
    y (numpy.array): Corrupted message (codeword)
    snr (float): signal to noise ratio
    kind (str) : kind of channel used for transmition
    Returns:
    B (numpy.array): ?
    '''
    var = 1 / snr
    B = (y - z) / (V + var)
    return B

  def dg(self, V, snr):
    '''
    A function to be described
    Args:
    y (numpy.array): Corrupted message (codeword)
    snr (float): signal to noise ratio
    kind (str) : kind of channel used for transmition
    Returns:
    A (numpy.array): ?
    '''
    var = 1 / snr
    A = 1 / (V + var)
    return A

  def denoising_a(self, x_hat, v, sigma, r, L, B, c=1):

    N = int(L * B)
    new_sigma = sigma.copy().reshape(L, B)
    new_r = r.copy().reshape(L, B)
    x_hat = x_hat.reshape(L, B)
    v = v.reshape(L, B)
    for l in range(L):
        x_hat[l, :] = c* np.exp((-c * (c-2*new_r[l, :]))/(2*new_sigma[l, :]**2)) / np.sum(np.exp((-c * (c-2*new_r[l, :]))/(2*new_sigma[l, :]**2)))
        v[l,:] = x_hat[l,:] * (1-x_hat[l, :])
    x_hat  = x_hat.reshape(N)
    v = v.reshape(N)
    return x_hat, v

  def amplify(self, arr, c=1):
    """
    Output the sparse superposition code corresponding to the AMP estimation
    Args:
        arr (numpy.array) : AMP estimation (N-d s.t N = L * B)
        L (int)           : Number of sections (Message length)
        B (int)           : Section size (Alphabet length)
    Returns:
        arr_new (numpy.array) : Sparse suprposition code (L * B)
        """
    row_idx = np.arange(self.L)
    col_idx = self.get_argmax(arr)
    arr = arr.reshape(self.L, self.B)
    arr_new = np.zeros((self.L, self.B))
    arr_new[row_idx, col_idx] = c
    return arr_new

  def section_error_rate(self, x_gt, x_hat, c=1):
    """
    Calculates Section Error Rate

    Args:
        x_gt  (numpy.array) : Vector corresponding to the ground truth signal
        x_hat (numpy.array) : Vector that corresponds to the estimation of x_gt
        L (int)           : Number of sections (Message length)
        B (int)           : Section size (Alphabet length)

    Returns:
        ser (float) : Section Error Rate (fraction of the wrongly re-constructed sections)
    """
    x_hat = self.amplify(x_hat, c=1)
    x_gt = x_gt.reshape(self.L, self.B)
    count = 0
    for row1, row2 in zip(x_hat, x_gt):
        if all(row1 == row2):
            count += 0
        else:
            count += 1
    ser = count / self.L
    return ser
  def get_argmax(self, arr):
    """
    Retrive the indices of the maximum argument of a L * B array

    Args:
        arr  (numpy.array) : (L * B)-d vector
        L (int)            : Number of sections (Message length)
        B (int)            : Section size (Alphabet length)
    Returns:
        idx (numpy.array) : array of indices

    """
    arr = arr.reshape(self.L, self.B)
    idx = np.argmax(arr, axis=1)
    return idx

  def decode(self, snr, R=1.3, t_max=25, ep=10**-8, c=1):
    """
    Approximate message decoder for sparse superposition codes as defined in https://arxiv.org/abs/1403.8024

    Args:

        x (numpy.array) : Ground truth N-Dimensional signal
        y (numpy.array) : Corrupted M-dimensional signal
        F (numpy.array) : Measurement Matrix of Dimensions M * N (coding operator)
        snr (int)       : Signal to noise ratio (1/sigma**2 fora power constraint of 1)
        B (int)         : Section (alphabet) size (power of 2)
        t_max (int)     : Maximum number of iterations
        ep (float)      : Tolerance for the difference between two succesive estimations of the signal x

    Returns:

        x_hat (numpy.array)         : AMP estimation of of the ground truth signal at convergence (i.e posterior means)
        v (numpy.array)             : AMP confidence in the estimations at convergence (i.e posterior variances)
        section_error (numpy.array) : Section-Error-Rate at each
        mse (numpy.array)           : Mean-Square-Error per section
    """
    self.R = R
    y = self.awgn_ch(snr)
    # Initialization
    delta = 1 + ep
    t = 0
    # AMP estimate of the signal
    x_hat = np.zeros(self.N)
    # AMP estimate of the variances per component
    v = np.ones(self.N)/(snr * self.B)
    #v = np.zeros(N)
    # Variances of the estimation of messages before taking the prior into account
    V = np.zeros(self.M)
    z = y.copy()
    ser = []
    ser.append(1)
    while t < t_max and delta > ep:
        # Here I just created a copy of he intilized arrays to use them as old instances (i.e values at t-1 when calculatin values at time step t)
        V_old = V.copy()
        z_old = z.copy()
        x_hat_old =x_hat.copy()

        # AMP starts here
        V = self.F **2 @ v
        z = (self.F @ x_hat) - V * self.g(z_old, y, V_old, snr)

        sigma = 1 / np.sqrt(self.dg(V, snr) @ self.F**2)

        # Estimation of messages before taking the prior into account
        r = x_hat_old + sigma ** 2 * (self.g(z, y, V, snr) @ self.F)
        # Creating copies of sigma and r arrays and reshaping them in order to use them in the denoisin function and update the estimates
        x_hat, v = self.denoising_a(x_hat, v, sigma, r, self.L , self.B, c)
        delta = (1/self.N)* np.sum(np.square(x_hat-x_hat_old))
        temp_err = self.section_error_rate(self.x, x_hat)
        ser.append(temp_err)
        #ber.append(bit_error)
        t = t + 1
        if t > 2 * t_max:
            return ser
            break
        #a_t.append(a)
    return ser
if __name__ == "__main__":
    print(__name__)