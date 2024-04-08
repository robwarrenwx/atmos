import numpy as np
import argparse
from atmos.constant import p_ref
from atmos.thermo import follow_moist_adiabat


def get_exact_temp(pres, wbpt, phase):
    
    # Convert pressure to Pa and WBPT to K
    p = pres * 100.
    thw = wbpt + 273.15

    # Note the array dimensions
    n_curve = len(thw)
    n_level = len(p)

    # Create array to store exact temperature data
    T = np.zeros((n_level, n_curve))

    # Set T at 1000 hPa equal to the WBPT
    k_ref = np.nonzero(p == p_ref)[0][0]
    T[k_ref] = thw

    # Loop downward from 1000 hPa to p_max
    for k in range(k_ref-1, -1, -1):

        # Follow a pseudoadiabat from p[k+1] to p[k]
        T[k] = follow_moist_adiabat(p[k+1], p[k], T[k+1], phase=phase,
                                    pseudo_method='iterative')
        
        #if k % 10 == 0:
        #    pk = p[k] / 100
        #    T_min = T[k].min() - 273.15
        #    T_max = T[k].max() - 273.15
        #    print(f"{pk:4.0f}, {T_min:6.1f}, {T_max:5.1f}")
        
    # Loop upward from 1000 hPa to p_min
    for k in range(k_ref+1, n_level):

        # Follow a pseudoadiabat from p[k-1] to p[k]
        T[k] = follow_moist_adiabat(p[k-1], p[k], T[k-1], phase=phase,
                                    pseudo_method='iterative')
        
        #if k % 10 == 0:
        #    pk = p[k] / 100
        #    T_min = T[k].min() - 273.15
        #    T_max = T[k].max() - 273.15
        #    print(f"{pk:4.0f}, {T_min:6.1f}, {T_max:5.1f}")

    return T - 273.15  # convert to degC


def get_exact_wbpt(pres, temp, phase):

    # Convert pressure to Pa and temperature to K
    p = pres * 100.
    T = temp + 273.15

    # Note the array dimensions
    n_curve = len(T)
    n_level = len(p)

    # Create array to store exact WBPT values
    thw = np.zeros((n_level, n_curve))

    # Set the WBPT at 1000 hPa equal to T
    k_ref = np.nonzero(p == p_ref)[0][0]
    thw[k_ref] = T

    # Loop downward from 1000 hPa to p_max
    for k in range(k_ref-1, -1, -1):

        # Follow a pseudoadiabat from p[k] to 1000 hPa
        thw[k] = follow_moist_adiabat(p[k], p_ref, T, phase=phase,
                                      pseudo_method='iterative')
        
        #if k % 10 == 0:
        #    pk = p[k] / 100
        #    thw_min = thw[k].min() - 273.15
        #    thw_max = thw[k].max() - 273.15
        #    print(f"{pk:4.0f}, {thw_min:6.1f}, {thw_max:5.1f}")
        
    # Loop upward from 1000 hPa to p_min
    for k in range(k_ref+1, n_level):

        # Follow a pseudoadiabat from p[k] to 1000 hPa
        thw[k] = follow_moist_adiabat(p[k], p_ref, T, phase=phase,
                                      pseudo_method='iterative')
        
        #if k % 10 == 0:
        #    pk = p[k] / 100
        #    thw_min = thw[k].min() - 273.15
        #    thw_max = thw[k].max() - 273.15
        #    print(f"{pk:4.0f}, {thw_min:6.1f}, {thw_max:5.1f}")

    return thw - 273.15  # convert to degC


def fit_temp_polynomials(p, thw, T, M=20, N=10):

    # Note the number of temperature curves
    n_curve = len(thw)

    # Select a reference WBPT curve
    i_ref = np.nonzero(thw == 20.)[0][0]  # optimal for liquid phase
    thw_ref = T[:, i_ref]

    # Fit M-degree polynomial to thw_ref(p) and store coefficients, b
    thw_ref_fit = np.poly1d(np.polyfit(p, thw_ref, M))
    b = thw_ref_fit.coeffs

    # Fit N-degree polynomial to T(thw_ref) and store coefficients, k
    k = np.zeros((N+1, n_curve))
    for i in range(n_curve):
        T_fit = np.poly1d(np.polyfit(thw_ref_fit(p), T[:, i], N))
        k[:, i] = T_fit.coeffs

    # Fit M-degree polynomial to k(thw) and store coefficients, a
    a = np.zeros((N+1, M+1))
    for i in range(N+1):
        k_fit = np.poly1d(np.polyfit(thw, k[i, :], M))
        a[i, :] = k_fit.coeffs

    return a, b


def fit_wbpt_polynomials(p, T, thw, M=20, N=10):

    # Note the number of WBPT curves
    n_curve = len(T)

    # Select a reference temperature curve
    i_ref = np.nonzero(T == -90.)[0][0]  # optimal for liquid phase
    T_ref = thw[:, i_ref]

    # Fit M-degree polynomial to T_ref(p) and store coefficients, beta
    T_ref_fit = np.poly1d(np.polyfit(p, T_ref, M))
    beta = T_ref_fit.coeffs

    # Fit N-degree polynomials to thw(T_ref) and store coefficients, kappa
    kappa = np.zeros((N+1, n_curve))
    for i in range(n_curve):
        thw_fit = np.poly1d(np.polyfit(T_ref_fit(p), thw[:, i], N))
        kappa[:, i] = thw_fit.coeffs

    # Fit M-degree polynomials to kappa(T) and store coefficients, alpha
    alpha = np.zeros((N+1, M+1))
    for i in range(N+1):
        kappa_fit = np.poly1d(np.polyfit(T, kappa[i, :], M))
        alpha[i, :] = kappa_fit.coeffs

    return alpha, beta


def get_approx_temp(p, thw, a, b, M=20, N=10):

    # Note array dimensions
    n_level = len(p)
    n_curve = len(thw)

    # Create arrays of polynomial exponents
    m = np.arange(M+1)
    n = np.arange(N+1)

    # Compute temperature using polynomial fits
    T = np.zeros((n_level, n_curve))
    for i in range(n_level):
        for j in range(n_curve):
            k = np.sum(a * np.power(thw[j], M-m), axis=1)
            thw_ref = np.sum(b * np.power(p[i], M-m), axis=0)
            T[i, j] = np.sum(k * np.power(thw_ref, N-n), axis=0)

    return T


def get_approx_wbpt(p, T, alpha, beta, M=20, N=10):

    # Note array dimensions
    n_level = len(p)
    n_curve = len(T)

    # Create arrays of polynomial exponents
    m = np.arange(M+1)
    n = np.arange(N+1)

    # Compute WBPT using polynomial fits
    thw = np.zeros((n_level, n_curve))
    for i in range(n_level):
        for j in range(n_curve):
            kappa = np.sum(alpha * np.power(T[j], M-m), axis=1)
            T_ref = np.sum(beta * np.power(p[i], M-m), axis=0)
            thw[i, j] = np.sum(kappa * np.power(T_ref, N-n), axis=0)

    return thw


def parse_args():

    parser = argparse.ArgumentParser(description="""Fits high-order 
                                     polynomials to pseudoadiabats following 
                                     Moisseeva and Stull (2017)""")
    
    #parser.add_argument("--phase", type=str, default="liquid", 
    #                    help="""condensate phase (valid options are 'liquid', 
    #                    'ice', or 'mixed')""")
    parser.add_argument("--pres_min", type=float, default=50.0,
                        help="minimum pressure (hPa)")
    parser.add_argument("--pres_max", type=float, default=1100.0,
                        help="maximum pressure (hPa)")
    parser.add_argument("--pres_inc", type=float, default=-5.0,
                        help="pressure increment (hPa)")
    parser.add_argument("--temp_min", type=float, default=-100.0,
                        help="minimum temperature (degC)")
    parser.add_argument("--temp_max", type=float, default=50.0,
                        help="maximum temperature (degC)")
    parser.add_argument("--temp_inc", type=float, default=0.5,
                        help="temperature increment (degC)")
    parser.add_argument("--wbpt_min", type=float, default=-70.0,
                        help="minimum WBPT (degC)")
    parser.add_argument("--wbpt_max", type=float, default=50.0,
                        help="maximum WBPT (degC)")
    parser.add_argument("--wbpt_inc", type=float, default=0.5,
                        help="WBPT increment (degC)")
    parser.add_argument("-M", type=int, default=20,
                        help="""order of polynomial fits for T_ref(p), 
                        kappa(T), thw_ref(p), and k(T)""")
    parser.add_argument("-N", type=int, default=10,
                        help="""order of polynomial fits for thw(T_ref) and
                        T(thw_ref)""")
    args = parser.parse_args()

    return args


def main():

    # Parse arguments
    args = parse_args()

    # Extract arguments
    #phase = args.phase
    pres_min = args.pres_min
    pres_max = args.pres_max
    pres_inc = args.pres_inc
    temp_min = args.temp_min
    temp_max = args.temp_max
    temp_inc = args.temp_inc
    wbpt_min = args.wbpt_min
    wbpt_max = args.wbpt_max
    wbpt_inc = args.wbpt_inc
    M = args.M
    N = args.N

    print(f"Minimum pressure is {pres_min} hPa")
    print(f"Maximum pressure is {pres_max} hPa")
    print(f"Pressure increment is {pres_inc} hPa")
    print(f"Minimum temperature is {temp_min} degC")
    print(f"Maximum temperature is {temp_max} degC")
    print(f"Temperature increment is {temp_inc} degC")
    print(f"Minimum WBPT is {wbpt_min} degC")
    print(f"Maximum WBPT is {wbpt_max} degC")
    print(f"WBPT increment is {wbpt_inc} degC")
    print(f"Order of inner polynomial is {M}")
    print(f"Order of outer polynomial is {N}")

    #print(f"\nPerforming fits for {phase}-phase pseudoadiabats")

    phase = 'liquid'  # fits for ice and mixed-phase are too inaccurate

    # Create pressure, temperature, and WBPT arrays
    pres = np.arange(pres_max, pres_min+pres_inc, pres_inc)  # max to min
    temp = np.arange(temp_min, temp_max+temp_inc, temp_inc)  # min to max
    wbpt = np.arange(wbpt_min, wbpt_max+wbpt_inc, wbpt_inc)  # min to max
    
    # Get exact temperature curves
    print("\nComputing exact temperature curves...")
    temp_exact = get_exact_temp(pres, wbpt, phase)

    # Fit polynomials to temperature curves
    print("\nFitting polynomials to temperature curves...")
    a, b = fit_temp_polynomials(pres, wbpt, temp_exact, M=M, N=N)

    # Compute approximate temperature using polynomial fits
    print("\nComputing approximate temperature curves...")
    temp_approx = get_approx_temp(pres, wbpt, a, b, M=M, N=N)
    
    # Compute minimum, maximum, and mean WBPT errors
    temp_error = temp_approx - temp_exact
    temp_mae = np.mean(np.abs(temp_error))
    temp_rmse = np.sqrt(np.mean(temp_error**2))
    print("\nTemperature Error Statistics:")
    print(f"Min  = {temp_error.min(): .5f} degC")
    print(f"Max  = {temp_error.max(): .5f} degC")
    print(f"Mean = {temp_error.mean(): .5f} degC")
    print(f"MAE  = {temp_mae: .5f} degC")
    print(f"RMSE = {temp_rmse: .5f} degC")

    # Get exact WBPT curves
    print("\nComputing exact WBPT curves...")
    wbpt_exact = get_exact_wbpt(pres, temp, phase)

    # Fit polynomials to WBPT curves
    print("\nFitting polynomials to WBPT curves...")
    alpha, beta = fit_wbpt_polynomials(pres, temp, wbpt_exact, M=M, N=N)
    
    # Compute approximate WBPT using polynomial fits
    print("\nComputing approximate WBPT curves...")
    wbpt_approx = get_approx_wbpt(pres, temp, alpha, beta, M=M, N=N)
    
    # Compute minimum, maximum, and mean WBPT errors
    wbpt_error = wbpt_approx - wbpt_exact
    wbpt_mae = np.mean(np.abs(wbpt_error))
    wbpt_rmse = np.sqrt(np.mean(wbpt_error**2))
    print("\nWBPT Error Statistics:")
    print(f"Min  = {wbpt_error.min(): .5f} degC")
    print(f"Max  = {wbpt_error.max(): .5f} degC")
    print(f"Mean = {wbpt_error.mean(): .5f} degC")
    print(f"MAE  = {wbpt_mae: .5f} degC")
    print(f"RMSE = {wbpt_rmse: .5f} degC")

    # Write polynomial coefficients to a file
    print("\nWriting polynomial coefficients to files...")
    #np.savetxt(f'a_coeff_{args.phase[:3]}.txt', a)
    #np.savetxt(f'b_coeff_{args.phase[:3]}.txt', b)
    #np.savetxt(f'alpha_coeff_{args.phase[:3]}.txt', alpha)
    #np.savetxt(f'beta_coeff_{args.phase[:3]}.txt', beta)
    np.savetxt(f'a_coeff.txt', a)
    np.savetxt(f'b_coeff.txt', b)
    np.savetxt(f'alpha_coeff.txt', alpha)
    np.savetxt(f'beta_coeff.txt', beta)

    print("\nDone")


if __name__ == "__main__":
    main()
