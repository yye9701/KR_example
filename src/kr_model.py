import numpy as np

def KR(C,
        B,
        ridge,
        inv_w,
        K
       ):

    '''
    - Get solution in KR model
    - Args: 
        - C (numpy array of dim (nt, Nmax)): cashflow matrix
        - B (numpy array of dim (nt,)): price vector corresponding to C
        - ridge (float): the ridge hyper-parameter. Require ridge>=0
        - inv_w (numpy array of dim (nt,)): inverse of weight vector w
        - K (numpy array of dim (Nmax, Nmax_y)): kernel matrix specific to kernel hyper-parameter alpha and delta.
            Nmax_y (in days) is the limit of extrapolation. 
    - Returns:
        - g_solved (numpy array of dim (Nmax,)): solved discount curve
        - y_solved (numpy array of dim (Nmax,)): solved log yield curve. annualized and not in %
    '''

    Nmax,Nmax_y=K.shape
    nt=B.shape[0]

    # get column indexes with nonzero cashflow
    arr_msk_col=np.where(C.sum(axis=0)!=0)[0]
    # max time to nonzero cashflow (in days) 
    tau_max_inday=arr_msk_col[-1]+1
    l_scaled=ridge/tau_max_inday

    # only keep rows and columns in K corresponding to nonzero cashflow days
    K_masked=K.take(arr_msk_col,axis=0).take(arr_msk_col,axis=1)

    # only keep columns of C with cashflow
    C_masked=C[:,arr_msk_col]
    Nt=C_masked.shape[1]
    # x: vector of time to cashflow dates (in year)
    x=(arr_msk_col+1)/365

    # get solution for (beta, r)
    CKC_inv=np.linalg.inv(C_masked@K_masked@C_masked.T+l_scaled*inv_w*np.identity(nt))
    # get coefficient vector beta. shape of beta is (nt,)
    beta=(C_masked.T@CKC_inv)@(B-C_masked@np.ones(Nt))
    # get discount vector with length Nmax
    g_solved=1+K.take(arr_msk_col,axis=1)@beta
    y_solved=-np.log(g_solved)/(np.arange(1,Nmax+1)/365)
    
    dict_out={'g_solved':g_solved,
        'y_solved':y_solved}

    return dict_out



def generate_kernel_matrix(alpha,
    delta,
    Nmax=3650,
    Nmax_y=None):
    '''
    - Generate a kernel matrix with parameters alpha and delta. No rows or columns correspond to infinite maturity.
    - Args:
        - alpha (float): kernel hyper-parameter alpha
        - delta (float): kernel hyper-parameter delta
        - Nmax (int): number of rows in the output kernel matrix
        - Nmax_y (int): number of columns in the output kernel matrix. If it is None, set Nmax_y=Nmax.
    - Returns:
        - K (numpy array of dim (Nmax,Nmax_y)): kernel matrix
    '''
    assert 0<=delta<=1

    if 0<delta<1:
        sqrt_D=np.sqrt(alpha**2 + 4*delta/(1-delta))
        l_1=(alpha-sqrt_D)/2
        l_2=(alpha+sqrt_D)/2
    else:
        sqrt_D=l_1=l_2=None

    if Nmax_y is None:
        Nmax_y=Nmax
    K=np.full((Nmax,Nmax_y),np.nan)
    for i in range(Nmax):
        x=(i+1)/365
        arr_y=np.arange(1,Nmax_y+1)/365
            
        min_xy=np.minimum(x,arr_y)
        max_xy=np.maximum(x,arr_y)
        
        if delta==0:
            K[i,:]=\
                -min_xy/alpha**2*np.exp(-alpha*min_xy)+\
                2/alpha**3*(1-np.exp(-alpha*min_xy))-\
                min_xy/alpha**2*np.exp(-alpha*max_xy)
        elif delta==1:
            K[i,:]=\
                1/alpha*(1-np.exp(-alpha*min_xy))
        else:
            K[i,:]=\
                -alpha/(delta*l_2**2)*(1-np.exp(-l_2*x)-np.exp(-l_2*arr_y))+\
                1/(alpha*delta)*(1-np.exp(-alpha*min_xy))+\
                1/(delta*sqrt_D)*(l_1**2/l_2**2 * np.exp(-l_2*(x+arr_y)) - np.exp(-l_1*min_xy - l_2* max_xy) )

    return K
