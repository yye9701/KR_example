import numpy as np
import scipy.optimize as opt 


def get_ytm_and_duration(cashflow,time_to_cashflow_inday, B_i, y_guess=0.0):
    '''
    - Calculate annualized YTM (not in %) and duration (in years) of a security
    - YTM is estimated using Newton's method
    - Assume (1) continuous compounding and (2) each year has 365 days
    - Args:
        - cashflow (numpy array): amount of cashflow
        - time_to_cashflow_inday (numpy array): time to cashflow in days
        - B_i (float): price of the security
        - y_guess (float): initial guess for YTM in Newton's method
    - Returns:
        - ytm_solved (float): estimated YTM
        - dur_solved (float): estimated duration in years
    '''

    assert time_to_cashflow_inday.shape==cashflow.shape
    ytm_func=lambda y: (sum(cashflow*np.exp(-time_to_cashflow_inday/365*y))-B_i)**2

    ytm_solved=opt.newton(ytm_func,y_guess)
    dur_solved=sum((time_to_cashflow_inday/365)*cashflow*np.exp(-time_to_cashflow_inday/365*ytm_solved))/B_i

    return ytm_solved, dur_solved