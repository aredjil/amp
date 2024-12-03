# Necessary libraries
import numpy as np
# import matplotlib.pyplot as plt 
# TODO: update the comments. 
# TODO: Update variable and function names. 
# TODO: Use numpy arrays instead of lists in the SER.  
class amp:
  def __init__(self, L:int, B:int):
    np.random.seed(0)
    self.L:int = L
    self.B:int = B
    self.N:int = L * B
    self.x:int = self.gen_ss()

  def awgn_ch(self, snr:int)->np.ndarray:
    self.snr:int = snr
    self.M:int = int((self.N * np.log2(self.B)) / (self.R * self.B))
    std:float = 1 / np.sqrt(snr)
    x_gt:np.ndarray = self.x.reshape(self.N)
    F:np.ndarray = np.random.normal(0, 1/np.sqrt(self.L), (self.M, self.N))
    self.F:np.ndarray = F
    noise:np.ndarray = std * np.random.randn(self.M)
    y:np.ndarray = (F @ x_gt) + noise
    return y

  def gen_ss(self)->np.ndarray:
    x:np.ndarray = np.zeros((self.L, self.B))
    for arr in x:
      randy:np.ndarray = np.random.randint(0, self.B)
      arr[randy] = 1
    return x

  def g(self, z:np.ndarray, y:np.ndarray, V:np.ndarray, snr:int)->np.ndarray:
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
    var:float = 1 / snr
    B:np.ndarray = (y - z) / (V + var)
    return B

  def dg(self, V:np.ndarray, snr:int)->np.ndarray:
    '''
    A function to be described
    Args:
    y (numpy.array): Corrupted message (codeword)
    snr (float): signal to noise ratio
    kind (str) : kind of channel used for transmition
    Returns:
    A (numpy.array): ?
    '''
    var:int = 1 / snr
    A:np.ndarray = 1 / (V + var)
    return A

  def denoising_a(self, x_hat, v, sigma, r, L, B, c=1)->tuple[np.ndarray, np.ndarray]:

    N:int = int(L * B)
    new_sigma:np.ndarray = sigma.copy().reshape(L, B)
    new_r:np.ndarray = r.copy().reshape(L, B)
    x_hat:np.ndarray = x_hat.reshape(L, B)
    v:np.ndarray = v.reshape(L, B)
    for l in range(L):
        x_hat[l, :] = c* np.exp((-c * (c-2*new_r[l, :]))/(2*new_sigma[l, :]**2)) / np.sum(np.exp((-c * (c-2*new_r[l, :]))/(2*new_sigma[l, :]**2)))
        v[l,:] = x_hat[l,:] * (1-x_hat[l, :])
    x_hat:np.ndarray  = x_hat.reshape(N)
    v:np.ndarray = v.reshape(N)
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
    row_idx:int = np.arange(self.L)
    col_idx:int = self.get_argmax(arr)
    arr:np.ndarray = arr.reshape(self.L, self.B)
    arr_new:np.ndarray = np.zeros((self.L, self.B))
    arr_new[row_idx, col_idx] = c
    return arr_new

  def section_error_rate(self, x_gt, x_hat, c=1)->list[float]:
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
    x_hat:np.ndarray = self.amplify(x_hat, c=1)
    x_gt:np.ndarray = x_gt.reshape(self.L, self.B)
    count:int = 0
    for row1, row2 in zip(x_hat, x_gt):
        if all(row1 == row2):
            count += 0
        else:
            count += 1
    ser = count / self.L
    return ser
  def get_argmax(self, arr:np.ndarray)->int:
    """
    Retrive the indices of the maximum argument of a L * B array

    Args:
        arr  (numpy.array) : (L * B)-d vector
        L (int)            : Number of sections (Message length)
        B (int)            : Section size (Alphabet length)
    Returns:
        idx (numpy.array) : array of indices

    """
    arr:np.ndarray = arr.reshape(self.L, self.B)
    idx:int = np.argmax(arr, axis=1)
    return idx

  def decode(self, snr:int, R:float=1.3, t_max:int=25, ep:int=10**-8, c:int=1)->list[float]:
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
    self.R:int = R
    y:np.ndarray = self.awgn_ch(snr)
    # Initialization
    delta:float = 1 + ep
    t:int = 0
    # AMP estimate of the signal
    x_hat:np.ndarray = np.zeros(self.N)
    # AMP estimate of the variances per component
    v:np.ndarray = np.ones(self.N)/(snr * self.B)
    #v = np.zeros(N)
    # Variances of the estimation of messages before taking the prior into account
    V:np.ndarray = np.zeros(self.M)
    z:np.ndarray = y.copy()
    ser:list[float] = []
    ser.append(1)
    while t < t_max and delta > ep:
        # Here I just created a copy of he intilized arrays to use them as old instances (i.e values at t-1 when calculatin values at time step t)
        V_old:np.ndarray = V.copy()
        z_old:np.ndarray = z.copy()
        x_hat_old:np.ndarray =x_hat.copy()

        # AMP starts here
        V:np.ndarray = self.F **2 @ v
        z:np.ndarray = (self.F @ x_hat) - V * self.g(z_old, y, V_old, snr)

        sigma:np.ndarray = 1 / np.sqrt(self.dg(V, snr) @ self.F**2)

        # Estimation of messages before taking the prior into account
        r:np.ndarray = x_hat_old + sigma ** 2 * (self.g(z, y, V, snr) @ self.F)
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
# def plot_dat(*data):
#    for ser in data: 
#       plt.plot(ser_13, label="R = 1.3", linestyle="dashed")
#       plt.plot(ser_14, label="R = 1.4", linestyle="dashed")
#       plt.plot(ser_145, label="R = 1.45", linestyle="dashed")
#       plt.plot(ser_16, label="R = 1.6", linestyle="dashed")
#       plt.xlabel("#iterations")
#       plt.ylabel("Section Error Rate (SER)")
#       plt.title(f"Section error rate vs number of iterations B={B}, L ={L}, and snr={snr}")
#       plt.xlim(0, )
#       plt.yscale("log")
#       plt.legend(loc="best")
#       plt.grid(True)
#       plt.savefig("./ser.png")
#     plt.show()
if __name__ == "__main__":
    print(__name__)