"""
Functions for calculating the following thermodynamic variables:
* effective gas constant, Rm
* effective specific heat, cpm
* latent heat of vaporisation, Lv
* latent heat of freezing, Lf
* latent heat of sublimation, Ls
* mixed-phase latent heat, Lx
* air density, rho
* dry air density, rhod
* virtual temperature, Tv
* mixing ratio, r
* vapour pressure, e
* saturation vapour pressure, es
* saturation specific humidity, qs
* saturation mixing ratio, rs
* relative humidity, RH
* dewpoint temperature, Td
* frost-point temperature, Tf
* saturation-point temperature, Ts
* lifting condensation level temperature, T_lcl, and pressure, p_lcl
* lifting deposition level temperature, T_ldl, and pressure, p_ldl
* lifting saturation level temperature, T_lsl, and pressure, T_lsl
* dry, pseudo, and saturated adiabatic pressure lapse rates
* parcel temperature following dry, pseudo, and saturated adiabats
* pseudo and isobaric wet-bulb temperatures, Tw
* dry potential temperature, thd
* moist potential temperature, thm
* virtual potential temperature, thv
* equivalent potential temperature, theq
* ice-liquid water potential temperature, thil
* wet-bulb potential temperature, thw
* saturated wet-bulb potential temperature, thws
* precipitable water, PW
* saturation fraction, SF
* integrated vapour transport, IVT
* geopotential height, Z
* hydrostatic pressure, P

References:
* Ambaum, M. H., 2020: Accurate, simple equation for saturated vapour
    pressure over water and ice. Quarterly Journal of the Royal Meteorological
    Society, 146, 4252-4258, https://doi.org/10.1002/qj.3899.
* Bryan, G. H., and J. M. Fristch, 2004: A reevaluation of ice-liquid water
    potential temperature. Monthly Weather Review, 132, 2421-2431,
    https://doi.org/10.1175/1520-0493(2004)132<2421:AROIWP>2.0.CO;2.
* Knox, J. A., D. S. Nevius, and P. N. Knox, 2017: Two simple and accurate
    approximations for wet-bulb temperature in moist conditions, with
    forecasting applications. Bulletin of the American Meteorological Society,
    98, 1897-1906, https://doi.org/10.1175/BAMS-D-16-0246.1.
* Romps, D. M., 2017: Exact expression for the lifting condensation level.
    Journal of the Atmospheric Sciences, 74, 3033-3057,
    https://doi.org/10.1175/JAS-D-17-0102.1.
* Romps, D. M., 2021: Accurate expressions for the dewpoint and frost point
    derived from the Rankine-Kirchoff approximations. Journal of the
    Atmospheric Sciences, 78, 2113-2116,
    https://doi.org/10.1175/JAS-D-20-0301.1.
* Romps, D. M., 2026: Wet-bulb temperature from pressure, relative humidity,
    and air temperature. Journal of Applied Meteorology and Climatology,
    https://doi.org/10.1175/JAMC-D-25-0130.1.
* Vazquez-Leal, H., Sandoval-Hernandez, M.A., Garcia-Gervacio, J.L.,
    Herrera-May, A.L., and Filobello-Nino, U.A., 2019. PSEM approximations
    for both branches of Lambert W function with applications. Discrete
    Dynamics in Nature and Society, 2019, 1-15,
    https://doi.org/10.1155/2019/8267951.
* Warren, R. A., 2025: A consistent treatment of mixed-phase saturation for
    atmospheric thermodynamics, Quarterly Journal of the Royal Meteorological
    Society, 151, e4866, https://doi.org/10.1002/qj.4866.

"""


import numpy as np
import warnings
#from scipy.special import lambertw
from atmos.constant import (g, Rd, Rv, eps, cpd, cpv, cpl, cpi, p_ref,
                            T0, es0, Lv0, Lf0, Ls0, T_liq, T_ice)
import atmos.pseudoadiabat as pseudoadiabat
from numba import vectorize


# Precision for iterative temperature calculations (K)
precision = 0.001

# Maximum number of iterations for iterative calculations
max_n_iter = 20


def effective_gas_constant(q, qt=None):
    """
    Computes effective gas constant for moist air.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        Rm (float or ndarray): effective gas constant (J/kg/K)

    """
    if qt is None:
        # (Eq. 12 from Warren 2025)
        Rm = (1 - q) * Rd + q * Rv  # = Rd * (1 - q + q / eps)
    else:
        # (Eq. 10 from Warren 2025)
        Rm = (1 - qt) * Rd + q * Rv  # = Rd * (1 - qt + q / eps)

    return Rm


def effective_specific_heat(q, qt=None, omega=0.0):
    """
    Computes effective isobaric specific heat for moist air.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        omega (float or ndarray, optional): ice fraction

    Returns:
        cpm (float or ndarray): effective isobaric specific heat (J/kg/K)

    """
    if qt is None:
        # (Eq. 17 from Warren 2025)
        cpm = (1 - q) * cpd + q * cpv  # = cpd * (1 - q + q / lambda),
                                       # where lambda = cpd / cpv
    else:
        # (Eq. 16 from Warren 2025)
        ql = (1 - omega) * (qt - q)  # liquid water mass fraction
        qi = omega * (qt - q)        # ice water mass fraction
        cpm = (1 - qt) * cpd + q * cpv + ql * cpl + qi * cpi

    return cpm


def latent_heat_of_vaporisation(T):
    """
    Computes latent heat of vaporisation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Lv (float or ndarray): latent heat of vaporisation (J/kg)

    """
    # (Eq. 23 from Warren 2025)
    Lv = Lv0 + (cpv - cpl) * (T - T0)

    return Lv


def latent_heat_of_sublimation(T):
    """
    Computes latent heat of sublimation for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Ls (float or ndarray): latent heat of sublimation (J/kg)

    """
    # (Eq. 24 from Warren 2025)
    Ls = Ls0 + (cpv - cpi) * (T - T0)

    return Ls


def latent_heat_of_freezing(T):
    """
    Computes latent heat of freezing for a given temperature.

    Args:
        T (float or ndarray): temperature (K)

    Returns:
        Lf (float or ndarray): latent heat of freezing (J/kg)

    """
    # (Eq. 25 from Warren 2025)
    Lf = Lf0 + (cpl - cpi) * (T - T0)

    return Lf


def mixed_phase_latent_heat(T, omega):
    """
    Computes mixed-phase latent heat for a given temperature and ice fraction
    using equations from Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        omega (float or ndarray): ice fraction

    Returns:
        Lx (float or ndarray): mixed-phase latent heat (J/kg)

    """

    # Compute mixed-phase specific heat
    # (Eq. 30 from Warren 2025)
    cpx = (1 - omega) * cpl + omega * cpi

    # Compute mixed-phase latent heat at the triple point
    # (Eq. 31 from Warren 2025)
    Lx0 = (1 - omega) * Lv0 + omega * Ls0  # = Lv0 + omega * Lf0

    # Compute mixed-phase latent heat
    # (Eq. 32 from Warren 2025)
    Lx = Lx0 + (cpv - cpx) * (T - T0)  # = (1 - omega) * Lv + omega * Ls
                                       # = Lv + omega * Lf

    return Lx


def air_density(p, T, q, qt=None):
    """
    Computes density of air using the ideal gas equation.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        rho (float or ndarray): air density (kg/m3)

    """
    # (Eq. 9 from Warren 2025)
    Rm = effective_gas_constant(q, qt=qt)
    rho = p / (Rm * T)

    return rho


def dry_air_density(p, T, q, qt=None):
    """
    Computes density of dry air using the ideal gas equation.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        rhod (float or ndarray): dry air density (kg/m3)

    """
    # (Eq. 7 from Warren 2025)
    rho = air_density(p, T, q, qt=qt)
    if qt is None:
        rhod = (1 - q) * rho
    else:
        rhod = (1 - qt) * rho

    return rhod


def virtual_temperature(T, q, qt=None):
    """
    Computes virtual (or density) temperature.

    Args:
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        Tv (float or ndarray): virtual (or density) temperature (K)

    """
    if qt is None:
        # (Eq. 13 from Warren 2025)
        Tv = T * (1 - q + q / eps)  # virtual temperature
    else:
        # (Eq. 11 from Warren 2025)
        Tv = T * (1 - qt + q / eps)  # density temperature

    return Tv


def mixing_ratio(q, qt=None):
    """
    Computes mixing ratio from specific humidity.

    Args:
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    # (Eq. 15 from Warren 2025)
    if qt is None:
        r = q / (1 - q)
    else:
        r = q / (1 - qt)

    return r


def vapour_pressure(p, q, qt=None):
    """
    Computes vapour pressure from pressure and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    # (Eq. 15 from Warren 2025)
    if qt is None:
        e = p * q / (eps * (1 - q) + q)
    else:
        e = p * q / (eps * (1 - qt) + q)

    return e


def saturation_vapour_pressure(T, phase='liquid', omega=0.0):
    """
    Computes saturation vapour pressure (SVP) for a given temperature using
    equations from Ambaum (2020) and Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        es (float or ndarray): saturation vapour pressure (Pa)

    """
   
    if phase == 'liquid':
        
        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute SVP over liquid water
        # (Eq. 26 from Warren 2025; cf. Eq. 13 from Ambaum 2020)
        es = es0 * np.power((T0 / T), ((cpl - cpv) / Rv)) * \
            np.exp((Lv0 / (Rv * T0)) - (Lv / (Rv * T)))
        
    elif phase == 'ice':
        
        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute SVP over ice
        # (Eq. 27 from Warren 2025; cf. Eq. 17 from Ambaum 2020)
        es = es0 * np.power((T0 / T), ((cpi - cpv) / Rv)) * \
            np.exp((Ls0 / (Rv * T0)) - (Ls / (Rv * T)))
        
    elif phase == 'mixed':
        
        # Compute mixed-phase specific heat
        # (Eq. 30 from Warren 2025)
        cpx = (1 - omega) * cpl + omega * cpi
        
        # Compute mixed-phase latent heat at the triple point
        # (Eq. 31 from Warren 2025)
        Lx0 = (1 - omega) * Lv0 + omega * Ls0

        # Compute mixed-phase latent heat
        # (Eq. 32 from Warren 2025)
        Lx = Lx0 + (cpv - cpx) * (T - T0)
        
        # Compute mixed-phase SVP
        # (Eq. 29 from Warren 2025)
        es = es0 * np.power((T0 / T), ((cpx - cpv) / Rv)) * \
            np.exp((Lx0 / (Rv * T0)) - (Lx / (Rv * T)))
        
    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return es


def saturation_specific_humidity(p, T, qt=None, phase='liquid', omega=0.0):
    """
    Computes saturation specific humidity from pressure and temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        qs (float or ndarray): saturation specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    if qt is None:
        # (Eq. 14 from Warren 2025, with qv = qt = qs and e = es)
        qs = eps * es / (p - (1 - eps) * es)
    else:
        # (Eq. 14 from Warren 2025, with qv = qs and e = es)
        qs  = (1 - qt) * eps * es / (p - es)

    return qs


def saturation_mixing_ratio(p, T, phase='liquid', omega=0.0):
    """
    Computes saturation mixing ratio from pressure and temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        rs (float or ndarray): saturation mixing ratio (kg/kg)

    """
    # (Eq. 15 from Warren 2025, with rv = rs and e = es)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    rs = eps * es / (p - es)

    return rs


def relative_humidity(p, T, q, qt=None, phase='liquid', omega=0.0):
    """
    Computes relative humidity with respect to specified phase from pressure, 
    temperature, and specific humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation 
            (default is 0.0)
        
    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = vapour_pressure(p, q, qt=qt)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH


@vectorize(['float32(float32)', 'float64(float64)'], nopython=True)
def _lambertw(x):
    """
    Evaluates the lower branch of the Lambert-W function using PSEM
    approximation from Vazquez-Leal et al. (2019).

    Args:
        x (float): dependent variable

    Returns:
        y (float): Lambert W function (lower branch)

    """

    y = np.nan

    # W_{-1,1}
    # (Eq. 15 from Vazquez-Leal et al. 2019)
    if (x >= -0.3678794411714423) and (x < -0.34):
        a1 = -7.874564067684664
        a2 = -63.11879948166995
        a3 = -168.6110850408981
        a4 = -150.1089086912451
        b1 = 15.97679839497612
        b2 = 98.26612857148953
        b3 = 293.9558944644677
        b4 = 430.4471947824411
        b5 = 247.8576700279611
        alpha = x * (a1 + x * (a2 + x * (a3 + x * a4)))
        beta = 1. + x * (b1 + x * (b2 + x * (b3 + x * (b4 + x * b5))))
        y = (alpha / beta) * (x + np.exp(-1.)) - 1.

    # W_{-1,2}
    # (Eq. 16 from Vazquez-Leal et al. 2019)
    if (x >= -0.34) and (x < -0.1):
        a1 = -1362.78381643109
        a2 = -1386.04132570149
        a3 = 11892.1649836015
        a4 = 16904.0507511421
        b1 = 251.440197724561
        b2 = -1264.99554712435
        b3 = -5687.63429510978
        b4 = -2639.24130979048
        y = (x * (a1 + x * (a2 + x * (a3 + x * a4)))) / \
            (1. + x * (b1 + x * (b2 + x * (b3 + x * b4))))

    # W_{-1,3}
    # (Eq. 17-18 from Vazquez-Leal et al. 2019)
    if (x >= -0.1) and (x < 0.):
        a1 = 1.01999365162218
        a2 = -12.6917365519443
        a3 = -45.1506015092455
        b1 = -22.9809693297808
        b2 = -104.692066099727
        b3 = -95.2085341727207
        k0 = (x * (a1 + x * (a2 + x * a3))) / \
             (1. + x * (b1 + x * (b2 + x * b3)))
        k1 = np.log(-x)
        k2 = k1 - np.log(-k1) + np.log(-k1) / k1
        y = k0 + k2

    # Iterate once to improve accuracy
    # (Eq. 30-32 from Vazquez-Leal et al. 2019)
    z = np.log(x / y) - y
    t = 2. * (1. + y) * (1. + y + (2./3.) * z)
    e = (z / (1. + y)) * (t - z) / (t - 2. * z)
    y = y * (1. + e)

    return y


def _dewpoint_temperature_from_relative_humidity(T, RH):
    """
    Computes dewpoint temperature from temperature and relative humidity over
    liquid water using equations from Romps (2021), as presented in
    Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """

    # Compute dewpoint temperature using Lambert W function
    # (Eq. 38 and 40 from Warren 2025; cf. Eq. 5-6 from Romps 2021)
    alpha = (Lv0 - (cpv - cpl) * T0) / (cpv - cpl)
    fn = np.power(RH, (Rv / (cpl - cpv))) * (alpha / T) * np.exp(alpha / T)
    #W = lambertw(fn, k=-1).real
    W = _lambertw(fn)
    Td = alpha / W

    return Td


def dewpoint_temperature(p, T, q):
    """
    Computes dewpoint temperature from pressure, temperature, and specific
    humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """

    # Compute relative humidity over liquid water
    RH = relative_humidity(p, T, q, phase='liquid')
    
    # Compute dewpoint temperature
    Td = _dewpoint_temperature_from_relative_humidity(T, RH)

    return Td


def _frost_point_temperature_from_relative_humidity(T, RH):
    """
    Computes frost-point temperature from temperature and relative humidity
    over ice using equations from Romps (2021), as presented in Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """

    # Compute frost-point temperature using Lambert W function
    # (Eq. 39 and 41 from Warren 2025; cf. Eq. 7-8 from Romps 2021)
    alpha = (Ls0 - (cpv - cpi) * T0) / (cpv - cpi)
    fn = np.power(RH, (Rv / (cpi - cpv))) * (alpha / T) * np.exp(alpha / T)
    #W = lambertw(fn, k=-1).real  # -1 branch because cpi > cpv
    W = _lambertw(fn)
    Tf = alpha / W

    return Tf


def frost_point_temperature(p, T, q):
    """
    Computes frost-point temperature from pressure, temperature, and specific
    humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Tf (float or ndarray): frost-point temperature (K)

    """    
    # Compute relative humidity over ice
    RH = relative_humidity(p, T, q, phase='ice')

    # Compute frost-point temperature
    Tf = _frost_point_temperature_from_relative_humidity(T, RH)

    return Tf


def _saturation_point_temperature_from_relative_humidity(T, RH, omega):
    """
    Computes saturation-point temperature from temperature, mixed-phase
    relative humidity, and ice fraction at saturation using equations from
    Warren (2025).

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): mixed-phase relative humidity (fraction)
        omega (float or ndarray): ice fraction at saturation

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Compute mixed-phase specific heat
    # (Eq. 30 from Warren 2025)
    cpx = (1 - omega) * cpl + omega * cpi

    # Compute mixed-phase latent heat at the triple point
    # (Eq. 31 from Warren 2025)
    Lx0 = (1 - omega) * Lv0 + omega * Ls0

    # Compute saturation-point temperature
    # (Eq. 42-43 from Warren 2025)
    alpha = (Lx0 - (cpv - cpx) * T0) / (cpv - cpx)
    fn = np.power(RH, (Rv / (cpx - cpv))) * (alpha / T) * np.exp(alpha / T)
    #W = lambertw(fn, k=-1).real
    W = _lambertw(fn)
    Ts = alpha / W

    return Ts


def saturation_point_temperature(p, T, q):
    """
    Computes saturation-point temperature from pressure, temperature, and 
    specific humidity using iterative procedure.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Ts (float or ndarray): saturation-point temperature (K)

    """

    # Intialise the saturation point temperature as the temperature
    Ts = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Ts value
        Ts_prev = Ts

        # Compute the ice fraction
        omega = ice_fraction(Ts)

        # Compute mixed-phase relative humidity
        RH = relative_humidity(p, T, q, phase='mixed', omega=omega)

        # Compute saturation-point temperature
        Ts = _saturation_point_temperature_from_relative_humidity(T, RH, omega)
   
        # Check if solution has converged
        if np.nanmax(np.abs(Ts - Ts_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Ts not converged after {max_n_iter} iterations")
                break

    return Ts


def lifting_condensation_level(p, T, q):
    """
    Computes pressure and temperature at the lifted condensation level (LCL)
    using equations from Romps (2017), as presented in Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        p_lcl (float or ndarray): pressure at the LCL (Pa)
        T_lcl (float or ndarray): temperature at the LCL (K)

    """
    
    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute relative humidity with respect to liquid water
    RH = relative_humidity(p, T, q, phase='liquid')

    # Compute temperature at the LCL using Lambert W function
    # (Eq. 44 and 46-47 from Warren 2025; cf. Eq. 22a,d-f from Romps 2017)
    beta = cpm / Rm + (cpl - cpv) / Rv
    alpha = -(1 / beta) * (Lv0 + (cpl - cpv) * T0) / Rv
    fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
    #W = lambertw(fn, k=-1).real
    W = _lambertw(fn)
    T_lcl = alpha / W

    # Compute pressure at the LCL
    # (Eq. 45 from Warren 2025; cf. Eq. 22b from Romps 2017)
    p_lcl = p * np.power((T_lcl / T), (cpm / Rm))

    # Ensure that LCL temperature and pressure do not exceed initial values
    T_lcl = np.minimum(T_lcl, T)
    p_lcl = np.minimum(p_lcl, p)

    return p_lcl, T_lcl


def lifting_deposition_level(p, T, q):
    """
    Computes pressure and temperature at the lifting deposition level (LDL)
    using equations from Romps (2017), as presented in Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        p_ldl (float or ndarray): pressure at the LDL (Pa)
        T_ldl (float or ndarray): temperature at the LDL (K)

    """

    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute relative humidity with respect to ice
    RH = relative_humidity(p, T, q, phase='ice')

    # Compute temperature at the LDL using Labert W function
    # (Eq. 48 and 50-51 from Warren 2025; cf. Eq. 23a,d-f from Romps 2017)
    beta = cpm / Rm + (cpi - cpv) / Rv
    alpha = -(1 / beta) * (Ls0 + (cpi - cpv) * T0) / Rv
    fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
    #W = lambertw(fn, k=-1).real
    W = _lambertw(fn)
    T_ldl = alpha / W

    # Compute pressure at the LDL
    # (Eq. 49 from Warren 2025; cf. Eq. 23b from Romps 2017)
    p_ldl = p * np.power((T_ldl / T), (cpm / Rm))

    # Ensure that LDL temperature and pressure do not exceed initial values
    T_ldl = np.minimum(T_ldl, T)
    p_ldl = np.minimum(p_ldl, p)

    return p_ldl, T_ldl


def lifting_saturation_level(p, T, q):
    """
    Computes pressure and temperature at the lifting saturation level (LSL)
    using equations from Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        p_lsl (float or ndarray): pressure at the LSL (Pa)
        T_lsl (float or ndarray): temperature at the LSL (K)

    """

    # Compute effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Initialise the LSL temperature as the temperature
    T_lsl = T

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous LSL temperature value
        T_lsl_prev = T_lsl

        # Compute the ice fraction
        omega = ice_fraction(T_lsl)

        # Compute mixed-phase relative humidity
        RH = relative_humidity(p, T, q, phase='mixed', omega=omega)

        # Compute mixed-phase specific heat
        # (Eq. 30 from Warren 2025)
        cpx = (1 - omega) * cpl + omega * cpi

        # Compute mixed-phase latent heat at the triple point
        # (Eq. 31 from Warren 2025)
        Lx0 = (1 - omega) * Lv0 + omega * Ls0

        # Compute temperature at the LSL
        # (Eq. 52 and 54-55 from Warren 2025)
        beta = cpm / Rm + (cpx - cpv) / Rv
        alpha = -(1 / beta) * (Lx0 + (cpx - cpv) * T0) / Rv
        fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
        #W = lambertw(fn, k=-1).real  # -1 branch because cpx > cpv
        W = _lambertw(fn)
        T_lsl = alpha / W

        # Check if solution has converged
        if np.max(np.abs(T_lsl - T_lsl_prev)) < precision:
            converged  = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"T_lsl not converged after {max_n_iter} iterations")
                break

    # Compute pressure at the LSL
    # (Eq. 53 from Warren 2025)
    p_lsl = p * np.power((T_lsl / T), (cpm / Rm))

    # Ensure that LSL temperature and pressure do not exceed initial values
    T_lsl = np.minimum(T_lsl, T)
    p_lsl = np.minimum(p_lsl, p)

    return p_lsl, T_lsl


def ice_fraction(Tstar, phase='mixed'):
    """
    Computes ice fraction given temperature at saturation using nonlinear
    parameterisation of Warren (2025).

    Args:
        Tstar (float or ndarray): temperature at saturation (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'mixed')

    Returns:
        omega (float or ndarray): ice fraction

    """

    # Ensure that Tstar is an array
    Tstar = np.atleast_1d(Tstar)

    # Compute the ice fraction
    # (Eq. 6 from Warren 2025, with T = Tstar)
    if phase == 'liquid':
        omega = np.zeros_like(Tstar)  # ice fraction is zero
    elif phase == 'ice':
        omega = np.ones_like(Tstar)  # ice fraction is one
    elif phase == 'mixed':
        omega = 0.5 * (1 - np.cos(np.pi * ((T_liq - Tstar) / (T_liq - T_ice))))
        omega[Tstar <= T_ice] = 1.0
        omega[Tstar >= T_liq] = 0.0
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if Tstar.size == 1:
        omega = omega.item()

    return omega


def ice_fraction_derivative(Tstar, phase='mixed'):
    """
    Computes derivative of ice fraction with respect to temperature at
    saturation using nonlinear parameterisation of Warren (2025).
    
    Args:
        Tstar (float or ndarray): temperature at saturation (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'mixed')

    Returns:
        domega_dTstar (float or ndarray): derivative of ice fraction (K^-1)
       
    """

    # Ensure that Tstar is an array
    Tstar = np.atleast_1d(Tstar)

    # Compute the derivative of the ice fraction
    # (derivative of Eq. 6 from Warren 2025, with T = Tstar)
    if phase == 'liquid' or phase == 'ice':
        domega_dTstar = np.zeros_like(Tstar)  # derivative is zero
    elif phase == 'mixed':
        domega_dTstar = -0.5 * (np.pi / (T_liq - T_ice)) * \
                np.sin(np.pi * ((T_liq - Tstar) / (T_liq - T_ice)))
        domega_dTstar[(Tstar <= T_ice) | (Tstar >= T_liq)] = 0.0
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if Tstar.size == 1:
        domega_dTstar = domega_dTstar.item()

    return domega_dTstar


def ice_fraction_at_saturation(p, T, q, phase='mixed', saturation='isobaric'):
    """
    Computes ice fraction at saturation for specified saturation process.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'mixed')
        saturation (str, optional): saturation process (valid options are
            'isobaric' or 'adiabatic'; default is 'isobaric')

    Returns:
        omega (float or ndarray): ice fraction at saturation

    """

    if phase == 'mixed':

        if saturation == 'isobaric':

            # Compute saturation-point temperature
            Tstar = saturation_point_temperature(p, T, q)

        elif saturation == 'adiabatic':

            # Compute lifting saturation level (LSL) temperature
            _, Tstar = lifting_saturation_level(p, T, q)

        else:

            raise ValueError("saturation must be 'isobaric' or 'adiabatic'")

    else:

        # Value of Tstar is irrelevant if phase='liquid' or phase='ice'
        Tstar = T

    # Compute the ice fraction
    omega = ice_fraction(Tstar, phase=phase)

    return omega


def dry_adiabatic_lapse_rate(p, T, q):
    """
    Computes dry adiabatic lapse rate in pressure coordinates.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        dT_dp (float or ndarray): dry adiabatic lapse rate (K/Pa)

    """

    # Compute the effective gas constant
    Rm = effective_gas_constant(q)

    # Compute the effective specific heat
    cpm = effective_specific_heat(q)

    # Compute dry adiabatic lapse rate
    dT_dp = (1 / p) * (Rm * T / cpm)

    return dT_dp


def pseudoadiabatic_lapse_rate(p, T, phase='liquid'):
    """
    Computes pseudoadiabatic lapse rate in pressure coordinates using equations
    from Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        dT_dp (float or ndarray): pseudoadiabatic lapse rate (K/Pa)

    """

    # Set the ice fraction
    omega = ice_fraction(T, phase=phase)

    # Compute saturation specific humidity
    qs = saturation_specific_humidity(p, T, phase=phase, omega=omega)

    # Compute Q term
    # (Eq. 71 from Warren 2025)
    Q = qs * (1 - qs + qs / eps)

    # Compute the effective gas constant
    Rm = effective_gas_constant(qs)

    # Compute the effective specific heat
    cpm = effective_specific_heat(qs)

    if phase == 'liquid':

        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute liquid pseudoadiabatic lapse rate
        # (Eq. 74 from Warren 2025)
        dT_dp = (1 / p) * (Rm * T + Lv * Q) / \
            (cpm + (Lv**2 * Q) / (Rv * T**2))

    elif phase == 'ice':

        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute ice pseudoadiabatic lapse rate
        # (Eq. 72 from Warren 2025, with omega = 1, so that Lx = Ls and
        # domega_dT = 0)
        dT_dp = (1 / p) * (Rm * T + Ls * Q) / \
            (cpm + (Ls**2 * Q) / (Rv * T**2))

    elif phase == 'mixed':

        # Compute the derivative of omega with respect to temperature
        domega_dT = ice_fraction_derivative(T)

        # Compute mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute saturation vapour pressues over liquid and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute mixed-phase pseudoadiabatic lapse rate
        # (Eq. 72 from Warren 2025)
        dT_dp = (1 / p) * (Rm * T + Lx * Q) / \
            (cpm + (Lx**2 * Q) / (Rv * T**2) +
             Lx * Q * np.log(esi / esl) * domega_dT)

    return dT_dp


def saturated_adiabatic_lapse_rate(p, T, qt, phase='liquid'):
    """
    Computes saturated adiabatic lapse rate in pressure coordinates using
    equations from Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        qt (float or ndarray): total water mass fraction (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        dT_dp (float or ndarray): saturated adiabatic lapse rate (K/Pa)

    """

    # Set the ice fraction
    omega = ice_fraction(T, phase=phase)

    # Compute saturation specific humidity
    qs = saturation_specific_humidity(p, T, qt=qt, phase=phase, omega=omega)

    # Compute Q term
    # (Eq. 67 from Warren 2025)
    Q = qs * (1 - qt + qs / eps) / (1 - qt)

    # Compute the effective gas constant
    Rm = effective_gas_constant(qs, qt=qt)

    # Compute the effective specific heat
    cpm = effective_specific_heat(qs, qt=qt, omega=omega)

    if phase == 'liquid':

        # Compute latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute liquid-only saturated adiabatic lapse rate
        # (Eq. 73 from Warren 2025)
        dT_dp = (1 / p) * (Rm * T + Lv * Q) / \
            (cpm + (Lv**2 * Q) / (Rv * T**2))

    elif phase == 'ice':

        # Compute latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute ice-only saturated adiabatic lapse rate
        # (Eq. 69 from Warren 2025, with omega = 1, so that Lx = Ls and
        # domega_dT = 0)
        dT_dp = (1 / p) * (Rm * T + Ls * Q) / \
            (cpm + (Ls**2 * Q) / (Rv * T**2))

    elif phase == 'mixed':

        # Compute the derivative of omega with respect to temperature
        domega_dT = ice_fraction_derivative(T)

        # Compute the mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute the latent heat of freezing
        Lf = latent_heat_of_freezing(T)

        # Compute saturation vapour pressues over liquid and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute mixed-phase saturated adiabatic lapse rate
        # (Eq. 69 from Warren 2025)
        dT_dp = (1 / p) * (Rm * T + Lx * Q) / \
            (cpm + (Lx**2 * Q) / (Rv * T**2) +
             (Lx * Q * np.log(esi / esl) - Lf * (qt - qs)) * domega_dT)

    return dT_dp


def follow_dry_adiabat(pi, pf, Ti, q):
    """
    Computes parcel temperature following a dry adiabat.

    Args:
        pi (float or ndarray): initial pressure (Pa)
        pf (float or ndarray): final pressure (Pa)
        Ti (float or ndarray): initial temperature (K)
        q (float or ndarray): specific humidity (kg/kg)

    Returns:
        Tf (float or ndarray): final temperature (K)

    """

    # Set effective gas constant and specific heat
    Rm = effective_gas_constant(q)
    cpm = effective_specific_heat(q)

    # Compute new temperature
    Tf = Ti * np.power((pf / pi), (Rm / cpm))

    return Tf


def follow_moist_adiabat(pi, pf, Ti, qt=None, phase='liquid', pseudo=True,
                         polynomial=True, explicit=False, dp=500.0):
    """
    Computes parcel temperature following a saturated adiabat or pseudoadiabat.
    For descending parcels, a pseudoadiabat is always used. By default,
    pseudoadiabatic calculations use polynomial fits for fast calculations, but
    can optionally use direct integration. Saturated adiabatic ascent can only
    be performed using direct integration (for now). At present, polynomial
    fits are only available for liquid pseudoadiabats.

    Args:
        pi (float or ndarray): initial pressure (Pa)
        pf (float or ndarray): final pressure (Pa)
        Ti (float or ndarray): initial temperature (K)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None; required for saturated adiabatic ascent)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        pseudo (bool): flag indicating whether to perform pseudoadiabatic
            parcel ascent (default is True)
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)

    Returns:
        Tf (float or ndarray): final temperature (K)

    """

    if pseudo and polynomial:

        if phase != 'liquid':
            raise ValueError(
                """Polynomial fits have yet to be created for ice and 
                mixed-phase pseudoadiabats. Calculations must be performed 
                using direct integration by setting polynomial=False."""
                )

        # Compute the wet-bulb potential temperature of the pseudoadiabat
        # that passes through (pi, Ti)
        thw = pseudoadiabat.wbpt(pi, Ti)

        # Compute the temperature on this pseudoadiabat at pf
        Tf = pseudoadiabat.temp(pf, thw)

        # Ensure that final temperature is equal to initial temperature where
        # initial and final pressures are equal (this is not guaraneteed due to
        # small errors introduced in polynomial fits)
        Tf = np.where(pf == pi, Ti, Tf)

    else:

        if not pseudo and qt is None:
            raise ValueError('qt is required for saturated adiabatic ascent')

        pi = np.atleast_1d(pi)
        pf = np.atleast_1d(pf)
        Ti = np.atleast_1d(Ti)

        if Ti.size > 1:
            # multiple initial temperature values
            if pi.size == 1:
                # single initial pressure value
                pi = np.full_like(Ti, pi)
            if pf.size == 1:
                # single final pressure value
                pf = np.full_like(Ti, pf)

        # Set the pressure increment based on whether the parcel is ascending
        # or descending
        dp = np.abs(dp)  # make sure pressure increment is positive
        ascending = (pf < pi)
        descending = np.logical_not(ascending)
        dp = np.where(ascending, -dp, dp)

        # Initialise the pressure and temperature at level 2
        p2 = np.copy(pi)
        T2 = np.copy(Ti)

        # Create an array to store the lapse rate
        dT_dp = np.zeros_like(p2)

        #print(np.min(p2), np.max(p2))
        #print(np.count_nonzero(ascending), np.count_nonzero(descending))
        #print(np.min(dp), np.max(dp))

        # Loop over pressure increments
        while np.nanmax(np.abs(p2 - pf)) > 0.0:

            # Set level 1 values
            p1 = p2
            T1 = T2

            # Update the pressure at level 2
            p2 = p1 + dp

            # Make sure we haven't overshot final pressure level
            if np.any(ascending):
                p2[ascending] = np.maximum(p2[ascending], pf[ascending])
            if np.any(descending):
                p2[descending] = np.minimum(p2[descending], pf[descending])

            #print(np.min(p2), np.max(p2))

            # Compute the pressure at the layer mid-point (in log-space)
            pmid = np.sqrt(p1 * p2)  # = exp(0.5 * (log(p1) + log(p2)))

            # Get initial estimate for the temperature at level 2 by following
            # a dry adiabat (ignoring the contribution of moisture)
            dT_dp = dry_adiabatic_lapse_rate(p1, T1, 0.0)
            #T2 = T1 + dT_dp * (p2 - p1)
            T2 = T1 + pmid * dT_dp * np.log(p2 / p1)  # pmid * dT/dp = dT/dlnp

            # Iterate to get the new temperature at level 2
            converged = False
            count = 0
            while not converged:

                # Update the previous level 2 temperature
                T2_prev = T2

                # Compute the temperature at the layer mid-point
                Tmid = 0.5 * (T1 + T2)

                # Compute the lapse rate for ascending parcels
                if np.any(ascending):
                    if pseudo:
                        dT_dp[ascending] = pseudoadiabatic_lapse_rate(
                            pmid[ascending], Tmid[ascending],
                            phase=phase
                            )
                    else:
                        dT_dp[ascending] = saturated_adiabatic_lapse_rate(
                            pmid[ascending], Tmid[ascending], qt[ascending],
                            phase=phase
                            )

                # Compute the lapse rate for descending parcels
                if np.any(descending):
                    dT_dp[descending] = pseudoadiabatic_lapse_rate(
                        pmid[descending], Tmid[descending],
                        phase=phase
                        )

                # Update the level 2 temperature
                #T2 = T1 + dT_dp * (p2 - p1)
                T2 = T1 + pmid * dT_dp * np.log(p2 / p1)  # pmid * dT/dp = dT/dlnp

                # Check if the solution has converged
                if np.nanmax(np.abs(T2 - T2_prev)) < precision:
                    converged = True
                else:
                    count += 1
                    if count == max_n_iter:
                        # should converge in just a couple of iterations
                        # provided pinc is not too large
                        print(f'Not converged after {max_n_iter} iterations')
                        break

                if explicit:
                    # skip additional iterations
                    converged = True

            #print(np.min(p1), np.min(p2), count)

        # Set the final temperature
        Tf = T2

    return Tf


def pseudo_wet_bulb_temperature(p, T, q, phase='liquid', polynomial=True,
                                explicit=False, dp=500.0):
    """
    Computes pseudo wet-bulb temperature using procedure outlined in section 7
    of Warren (2025).

    Pseudo wet-bulb temperature is the temperature of a parcel of air lifted
    adiabatically to saturation and then brought pseudoadiabatically at
    saturation back to its original pressure. It is always less than the
    isobaric wet-bulb temperature.

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)

    Returns:
        Tw (float or ndarray): pseudo wet-bulb temperature (K)

    """

    if phase == 'liquid':

        # Get pressure and temperature at the LCL
        p_lcl, T_lcl = lifting_condensation_level(p, T, q)

        # Follow a pseudoadiabat from the LCL to the original pressure
        Tw = follow_moist_adiabat(p_lcl, p, T_lcl, phase='liquid', pseudo=True,
                                  polynomial=polynomial, explicit=explicit,
                                  dp=dp)

    elif phase == 'ice':

        # Get pressure and temperature at the LDL
        p_ldl, T_ldl = lifting_deposition_level(p, T, q)

        # Follow a pseudoadiabat from the LDL to the original pressure
        Tw = follow_moist_adiabat(p_ldl, p, T_ldl, phase='ice', pseudo=True,
                                  polynomial=polynomial, explicit=explicit,
                                  dp=dp)

    elif phase == 'mixed':

        # Get pressure and temperature at the LSL
        p_lsl, T_lsl = lifting_saturation_level(p, T, q)

        # Follow a pseudoadiabat from the LSL to the original pressure
        Tw = follow_moist_adiabat(p_lsl, p, T_lsl, phase='mixed', pseudo=True,
                                  polynomial=polynomial, explicit=explicit,
                                  dp=dp)

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    # Ensure that Tw does not exceed T (this can happen due to small errors
    # introduced by polynomial fits or numerical integration)
    Tw = np.minimum(Tw, T)

    if not np.isscalar(Tw) and Tw.size == 1:
        Tw = Tw.item()

    return Tw


def isobaric_wet_bulb_temperature(p, T, q, phase='liquid'):
    """
    Computes isobaric wet-bulb temperature using equations from Warren (2025).

    Isobaric wet-bulb temperature is the temperature of a parcel of air cooled
    isobarically to saturation via the evaporation of water into it, with all
    latent heat supplied by the parcel. It is always greater than the pseudo
    wet-bulb temperature. Isobaric wet-bulb temperature is similar (but not
    identical) to the quantity measured by a wet-bulb thermometer. 

    See https://glossary.ametsoc.org/wiki/Wet-bulb_temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """

    # Compute dewpoint temperature
    Td = dewpoint_temperature(p, T, q)

    # Initialise Tw using the "one-third rule" (Knox et al. 2017)
    Tw = T - (1 / 3) * (T - Td)

    # Compute the latent heat at temperature T
    if phase == 'liquid':
        Lv_T = latent_heat_of_vaporisation(T)
    elif phase == 'ice':
        Ls_T = latent_heat_of_sublimation(T)
    elif phase == 'mixed':
        omega_T = ice_fraction(T)
        Lx_T = mixed_phase_latent_heat(T, omega_T)
    else:
        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Tw value
        Tw_prev = Tw

        # Compute the ice fraction at Tw
        if phase == 'liquid':
            omega_Tw = 0.0
        elif phase == 'ice':
            omega_Tw = 1.0
        elif phase == 'mixed':
            omega_Tw = ice_fraction(Tw)

        # Compute saturation specific humidity at Tw
        qs_Tw = saturation_specific_humidity(p, Tw, phase=phase,
                                             omega=omega_Tw)

        # Compute the effective specific heat at qs(Tw)
        cpm_qs_Tw = effective_specific_heat(qs_Tw)

        if phase == 'liquid':

            # Compute the latent heat of vaporisation at Tw
            Lv_Tw = latent_heat_of_vaporisation(Tw)

            # Compute the derivative of qs with respect to Tw
            # (Eq. 83 from Warren 2025, with omega = 0, so that Lx = Lv and
            # domega_dT = 0)
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * Lv_Tw / (Rv * Tw**2)

            # Compute f(Tw) and f'(Tw)
            # (Eq. 81 and 82 from Warren 2025, with omega = 0, so that Lx = Lv)
            f = cpm_qs_Tw * (T - Tw) - Lv_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Lv_T) * dqs_dTw - cpm_qs_Tw

        elif phase == 'ice':

            # Compute the latent heat of sublimation at Tw
            Ls_Tw = latent_heat_of_sublimation(Tw)

            # Compute the derivative of qs with respect to Tw
            # (Eq. 83 from Warren 2025, with omega = 1, so that Lx = Ls and
            # domega_dT = 0)
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * Ls_Tw / (Rv * Tw**2)

            # Compute f(Tw) and f'(Tw)
            # (Eq. 81 and 82 from Warren 2025, with omega = 1, so that Lx = Ls)
            f = cpm_qs_Tw * (T - Tw) - Ls_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Ls_T) * dqs_dTw - cpm_qs_Tw

        elif phase == 'mixed':

            # Compute the derivative of omega with respect to Tw
            domega_dTw = ice_fraction_derivative(Tw)

            # Compute the mixed-phase latent heat at Tw
            Lx_Tw = mixed_phase_latent_heat(Tw, omega_Tw)

            # Compute the saturation vapour pressues over liquid and ice at Tw
            esl_Tw = saturation_vapour_pressure(Tw, phase='liquid')
            esi_Tw = saturation_vapour_pressure(Tw, phase='ice')

            # Compute the derivative of qs with respect to Tw
            # (Eq. 83 from Warren 2025)
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * \
                (Lx_Tw / (Rv * Tw**2) + np.log(esi_Tw / esl_Tw) * domega_dTw)

            # Compute f(Tw) and f'(Tw)
            # (Eq. 81 and 82 from Warren 2025)
            f = cpm_qs_Tw * (T - Tw) - Lx_T * (qs_Tw - q)
            fprime = ((cpv - cpd) * (T - Tw) - Lx_T) * dqs_dTw - cpm_qs_Tw
       
        # Update Tw using Newton's method
        # (Eq. 84 from Warren 2025)
        Tw = Tw - f / fprime

        # Check for convergence
        if np.nanmax(np.abs(Tw - Tw_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Tw not converged after {max_n_iter} iterations")
                break

    return Tw


def isobaric_wet_bulb_temperature_romps(p, T, q, phase='liquid'):
    """
    Computes isobaric wet-bulb temperature using method from Romps (2026),
    which has been implemented in the heatindex Python library
    (https://github.com/davidromps/heatindex).

    A corrigendum to Warren (2025) will be published in the near future
    containing derivations for the equations implemented in this function.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """

    # Compute dewpoint temperature
    Td = dewpoint_temperature(p, T, q)

    # Initialise Tw using the "one-third rule" (Knox et al. 2017)
    Tw = T - (1 / 3) * (T - Td)

    # Compute the effective specific heat
    cpm = effective_specific_heat(q)

    # Iterate to convergence
    converged = False
    count = 0
    while not converged:

        # Update the previous Tw value
        Tw_prev = Tw

        # Compute the ice fraction at Tw
        if phase == 'liquid':
            omega_Tw = 0.0
        elif phase == 'ice':
            omega_Tw = 1.0
        elif phase == 'mixed':
            omega_Tw = ice_fraction(Tw)

        # Compute saturation specific humidity at Tw
        qs_Tw = saturation_specific_humidity(p, Tw, phase=phase,
                                             omega=omega_Tw)

        if phase == 'liquid':

            # Compute the latent heat of vaporisation at Tw
            Lv_Tw = latent_heat_of_vaporisation(Tw)

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * Lv_Tw / (Rv * Tw**2)

            # Compute the derivative of Lv with respect to Tw
            dLv_dTw = (cpv - cpl)

            # Compute f(Tw) and f'(Tw)
            f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Lv_Tw
            fprime = -(cpm * (T - Tw) + Lv_Tw) * dqs_dTw - \
                cpm * (1 - qs_Tw) - (qs_Tw - q) * dLv_dTw

        elif phase == 'ice':

            # Compute the latent heat of sublimation at Tw
            Ls_Tw = latent_heat_of_sublimation(Tw)

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * Ls_Tw / (Rv * Tw**2)

            # Compute the derivative of Ls with respect to Tw
            dLs_dTw = (cpv - cpi)

            # Compute f(Tw) and f'(Tw)
            f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Ls_Tw
            fprime = -(cpm * (T - Tw) + Ls_Tw) * dqs_dTw - \
                cpm * (1 - qs_Tw) - (qs_Tw - q) * dLs_dTw

        elif phase == 'mixed':

            # Compute the derivative of omega with respect to Tw
            domega_dTw = ice_fraction_derivative(Tw)

            # Compute the mixed-phase latent heat at Tw
            Lx_Tw = mixed_phase_latent_heat(Tw, omega_Tw)

            # Compute the mixed-phase isobaric specific heat
            cpx = (1 - omega_Tw) * cpl + omega_Tw * cpi

            # Compute the derivate of Lx with respect to Tw
            dLx_dTw = (cpv - cpx) + (Tw - T0) * (cpl - cpi) * domega_dTw

            # Compute the saturation vapour pressues over liquid and ice at Tw
            esl_Tw = saturation_vapour_pressure(Tw, phase='liquid')
            esi_Tw = saturation_vapour_pressure(Tw, phase='ice')

            # Compute the derivative of qs with respect to Tw
            dqs_dTw = qs_Tw * (1 - qs_Tw + qs_Tw / eps) * \
                (Lx_Tw / (Rv * Tw**2) + np.log(esi_Tw / esl_Tw) * domega_dTw)

            # Compute f(Tw) and f'(Tw)
            f = cpm * (T - Tw) * (1 - qs_Tw) - (qs_Tw - q) * Lx_Tw
            fprime = -(cpm * (T - Tw) + Lx_Tw) * dqs_dTw - \
                cpm * (1 - qs_Tw) - (qs_Tw - q) * dLx_dTw
       
        # Update Tw using Newton's method
        Tw = Tw - f / fprime

        # Check for convergence
        if np.nanmax(np.abs(Tw - Tw_prev)) < precision:
            converged = True
        else:
            count += 1
            if count == max_n_iter:
                print(f"Tw not converged after {max_n_iter} iterations")
                break

    return Tw


def wet_bulb_temperature(p, T, q, saturation='pseudo',
                         isobaric_method='Warren', phase='liquid',
                         polynomial=True, explicit=False, dp=500.0):
    """
    Computes wet-bulb temperature for specified saturation process.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        saturation (str, optional): saturation process (valid options are
            'pseudo' or 'isobaric'; default is 'pseudo')
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Warren')
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)

    Returns:
        Tw: wet-bulb temperature (K)

    """

    if saturation == 'pseudo':
        Tw = pseudo_wet_bulb_temperature(p, T, q, phase=phase,
                                         polynomial=polynomial,
                                         explicit=explicit, dp=dp)
    elif saturation == 'isobaric':
        if isobaric_method == 'Warren':
            Tw = isobaric_wet_bulb_temperature(p, T, q, phase=phase)
        elif isobaric_method == 'Romps':
            Tw = isobaric_wet_bulb_temperature_romps(p, T, q, phase=phase)
        else:
           raise ValueError(
            "isobaric_method must be one of 'Warren' or 'Romps'"
           )
    else:
        raise ValueError("saturation must be one of 'pseudo' or 'isobaric'")

    return Tw


def dry_potential_temperature(p, T):
    """
    Computes potential temperature of dry air.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)

    Returns:
        thd (float or ndarray): potential temperature of dry air (K)

    """

    # Compute potential temperature of dry air
    thd = T * (p_ref / p) ** (Rd / cpd)

    return thd


def moist_potential_temperature(p, T, q, qt=None, omega=0.0):
    """
    Computes potential temperature of moist air.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        omega (float or ndarray, optional): ice fraction (default is 0.0)

    Returns:
        thm (float or ndarray): potential temperature of moist air (K)

    """

    # Set effective gas constant and specific heat
    Rm = effective_gas_constant(q, qt=qt)
    cpm = effective_specific_heat(q, qt=qt, omega=omega)

    # Compute potential temperature of moist air
    thm = T * (p_ref / p) ** (Rm / cpm)

    return thm


def potential_temperature(p, T, q=None, qt=None, omega=0.0):
    """
    Computes potential temperature for dry or moist air.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray, optional): specific humidity (kg/kg) (default is
            None, in which case dry potential temperature is returned)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        omega (float or ndarray, optional): ice fraction (default is 0.0)

    Returns:
        th (float or ndarray): potential temperature (K)

    """

    if q is None:
        th = dry_potential_temperature(p, T)
    else:
        th = moist_potential_temperature(p, T, q, qt=qt, omega=omega)

    return th


def virtual_potential_temperature(p, T, q, qt=None, omega=0.0):
    """
    Computes virtual (or density) potential temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        omega (float or ndarray, optional): ice fraction (default is 0.0)

    Returns:
        thv (float or ndarray): virtual potential temperature (K)

    """

    # Set effective gas constant and specific heat
    Rm = effective_gas_constant(q, qt=qt)
    cpm = effective_specific_heat(q, qt=qt, omega=omega)

    # Compute the virtual temperature
    Tv = virtual_temperature(T, q, qt=qt)

    # Compute virtual potential temperature
    thv = Tv * (p_ref / p) ** (Rm / cpm)

    return thv


def equivalent_potential_temperature(p, T, q, qt=None, phase='liquid',
                                     omega=0.0):
    """
    Computes equivalent potential temperature using equations from
    Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction (default is 0.0)

    Returns:
        theq (float or ndarray): equivalent potential temperature (K)

    """

    if qt is None:
        qt = q

    # Compute the dry pressure
    e = vapour_pressure(p, q, qt=qt)
    pd = p - e

    # Compute the relative humidity
    RH = relative_humidity(p, T, q, qt=qt, phase=phase, omega=omega)

    # Compute cpml
    cpml = (1 - qt) * cpd + qt * cpl

    if phase == 'liquid':

        # Compute the latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute the equivalent potential temperature
        # (Eq. 96 from Warren 2025)
        theq = T * (p_ref / pd) ** ((1 - qt) * Rd / cpml) * \
            RH ** (-q * Rv / cpml) * \
            np.exp(Lv * q / (cpml * T))

    elif phase == 'ice':

        # Compute the saturation vapour pressures with respect to liquid water
        # and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute the latent heat of sublimation
        Ls = latent_heat_of_sublimation(T)

        # Compute the latent heat of freezing
        Lf = latent_heat_of_freezing(T)

        # Compute the equivalent potential temperature
        # (Eq. 95 from Warren 2025, with omega = 1, so that Ls = Lx)
        theq = T * (p_ref / pd) ** ((1 - qt) * Rd / cpml) * \
            RH ** (-q * Rv / cpml) * \
            (esi / esl) ** (-qt * Rv / cpml) * \
            np.exp((Ls * q - Lf * qt) / (cpml * T))

    elif phase== 'mixed':

        # Compute the saturation vapour pressures with respect to liquid water
        # and ice
        esl = saturation_vapour_pressure(T, phase='liquid')
        esi = saturation_vapour_pressure(T, phase='ice')

        # Compute the mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute the latent heat of freezing
        Lf = latent_heat_of_freezing(T)

        # Compute the equivalent potential temperature
        # (Eq. 95 from Warren 2025)
        theq = T * (p_ref / pd) ** ((1 - qt) * Rd / cpml) * \
            RH ** (-q * Rv / cpml) * \
            (esi / esl) ** (-omega * qt * Rv / cpml) * \
            np.exp((Lx * q - omega * Lf * qt) / (cpml * T))

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return theq


def ice_liquid_water_potential_temperature(p, T, q, qt=None, phase='liquid',
                                           omega=0.0):
    """
    Computes ice-liquid water potential temperature using equations from
    Bryan and Fristch (2004) and Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        qt (float or ndarray, optional): total water mass fraction (kg/kg)
            (default is None, which implies qt = q)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction (default is 0.0)

    Returns:
        thil (float or ndarray): ice-liquid water potential temperature (K)

    """

    if qt is None:
        qt = q

    # Compute the relative humidity with respect to liquid
    RH = relative_humidity(p, T, q, qt=qt, phase=phase, omega=omega)

    # Set constants
    Rmv = (1 - qt) * Rd + qt * Rv
    cpmv = (1 - qt) * cpd + qt * cpv

    if phase == 'liquid':

        # Compute the latent heat of vaporisation
        Lv = latent_heat_of_vaporisation(T)

        # Compute the liquid water potential temperature
        # (Eq. 101 from Warren 2025)
        thil = T * (p_ref / p) ** (Rmv / cpmv) * \
            ((1 - qt + q / eps) / (1 - qt + qt / eps)) ** (Rmv / cpmv) * \
            (q / qt) ** (-qt * Rv / cpmv) * \
            RH ** ((qt - q) * Rv / cpmv) * \
            np.exp(- Lv * (qt - q) / (cpmv * T))

    elif phase == 'ice':

        # Compute the latent heat of vaporisation
        Ls = latent_heat_of_sublimation(T)

        # Compute the ice water potential temperature
        # (Eq. 101 from Warren 2025, with omega = 1, so that Lv = Ls)
        thil = T * (p_ref / p) ** (Rmv / cpmv) * \
            ((1 - qt + q / eps) / (1 - qt + qt / eps)) ** (Rmv / cpmv) * \
            (q / qt) ** (-qt * Rv / cpmv) * \
            RH ** ((qt - q) * Rv / cpmv) * \
            np.exp(- Ls * (qt - q) / (cpmv * T))

    elif phase == 'mixed':

        # Compute the mixed-phase latent heat
        Lx = mixed_phase_latent_heat(T, omega)

        # Compute the ice-liquid water potential temperature
        # (Eq. 100 from Warren 2025; cf. Eq. 25 from Bryan and Fritsch 2004)
        thil = T * (p_ref / p) ** (Rmv / cpmv) * \
            ((1 - qt + q / eps) / (1 - qt + qt / eps)) ** (Rmv / cpmv) * \
            (q / qt) ** (-qt * Rv / cpmv) * \
            RH ** ((qt - q) * Rv / cpmv) * \
            np.exp(- Lx * (qt - q) / (cpmv * T))

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    return thil


def wet_bulb_potential_temperature(p, T, q, phase='liquid', polynomial=True,
                                   explicit=False, dp=500.0):
    """
    Computes wet-bulb potential temperature using procedure outlined in
    section 7 of Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        q (float or ndarray): specific humidity (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)

    Returns:
        thw (float or ndarray): wet-bulb potential temperature (K)

    """

    if phase == 'liquid':

        # Get pressure and temperature at the LCL
        p_lcl, T_lcl = lifting_condensation_level(p, T, q)

        # Follow a pseudoadiabat from the LCL to 1000 hPa
        thw = follow_moist_adiabat(p_lcl, p_ref, T_lcl, phase='liquid',
                                   pseudo=True, polynomial=polynomial,
                                   explicit=explicit, dp=dp)

    elif phase == 'ice':

        # Get pressure and temperature at the LDL
        p_ldl, T_ldl = lifting_deposition_level(p, T, q)

        # Follow a pseudoadiabat from the LDL to 1000 hPa
        thw = follow_moist_adiabat(p_ldl, p_ref, T_ldl, phase='ice',
                                   pseudo=True, polynomial=polynomial,
                                   explicit=explicit, dp=dp)

    elif phase == 'mixed':

        # Get pressure and temperature at the LSL
        p_lsl, T_lsl = lifting_saturation_level(p, T, q)

        # Follow a pseudoadiabat from the LSL to 1000 hPa
        thw = follow_moist_adiabat(p_lsl, p_ref, T_lsl, phase='mixed',
                                   pseudo=True, polynomial=polynomial,
                                   explicit=explicit, dp=dp)

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if not np.isscalar(thw) and thw.size == 1:
        thw = thw.item()

    return thw


def saturation_wet_bulb_potential_temperature(p, T, phase='liquid',
                                              polynomial=True, explicit=False,
                                              dp=500.0):
    """
    Computes saturation wet-bulb potential temperature using procedure outlined
    in section 7 of Warren (2025).

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        polynomial (bool, optional): flag indicating whether to use polynomial
            fits to pseudoadiabats (default is True)
        explicit (bool, optional): flag indicating whether to use explicit
            integration of lapse rate equation (default is False)
        dp (float, optional): pressure increment for integration of lapse rate
            equation (default is 500 Pa = 5 hPa)

    Returns:
        thws (float or ndarray): saturation wet-bulb potential temperature (K)

    """

    # Follow a pseudoadiabat to 1000 hPa
    thws = follow_moist_adiabat(p, p_ref, T, phase=phase, pseudo=True,
                                polynomial=polynomial, explicit=explicit,
                                dp=dp)

    if not np.isscalar(thws) and thws.size == 1:
        thws = thws.item()

    return thws


def precipitable_water(p, q, p_sfc=None, q_sfc=None, p_bot=None, p_top=None,
                       vertical_axis=0):
    """
    Computes precipitable water (a.k.a. column water vapour) from profiles of
    pressure and specific humidity.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        p_bot (float or ndarray, optional): pressure at which to start vertical
            integration (Pa) (default is None)
        p_top (float or ndarray, optional): pressure at which to stop vertical
            integration (Pa) (default is None)
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        PW (float or ndarray): precipitable water (kg/m2)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        q = np.atleast_2d(q).T

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        q_sfc = q[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    q_sfc = np.atleast_1d(q_sfc)

    # If p_bot is not specified, use surface
    if p_bot is None:
        p_bot = p_sfc
    elif np.isscalar(p_bot):
        p_bot = np.full_like(p_sfc, p_bot)

    # If p_top is not specified, use top level
    if p_top is None:
        p_top = p[-1]
    elif np.isscalar(p_top):
        p_top = np.full_like(p_sfc, p_top)

    # Check if bottom of layer is above top of layer
    if np.any(p_bot < p_top):
        n_pts = np.count_nonzero(p_bot < p_top)
        raise ValueError(f'p_bot is above p_top for {n_pts} points')

    # Check if bottom of layer is below surface
    if np.any(p_bot > p_sfc):
        n_pts = np.count_nonzero(p_bot > p_sfc)
        warnings.warn(f'p_bot is below {bottom} for {n_pts} points')

    # Check if top of layer is above highest level
    if np.any(p_top < p[-1]):
        n_pts = np.count_nonzero(p_top < p[-1])
        warnings.warn(f'p_top is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Initialise level 2 pressure and specific humidity
    p2 = p_sfc.copy()
    q2 = q_sfc.copy()

    # Initialise precipitable water
    PW = np.zeros_like(p_sfc)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        q1 = q2.copy()
        if np.all(p1 <= p_top):
            # can break out of loop
            break

        # Update level 2 fields
        above_sfc = (p[k] <= p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        q2 = np.where(above_sfc, q[k], q_sfc)
        if np.all(p2 >= p_bot):
            # can skip this level
            continue

        # If crossing bottom of layer, reset level 1
        cross_bot = (p1 > p_bot) & (p2 < p_bot)
        if np.any(cross_bot):
            weight = np.log(p1[cross_bot] / p_bot[cross_bot]) / \
                np.log(p1[cross_bot] / p2[cross_bot])
            q1[cross_bot] = (1 - weight) * q1[cross_bot] + \
                weight * q2[cross_bot]
            p1[cross_bot] = p_bot[cross_bot]

        # If crossing top of layer, reset level 2
        cross_top = (p1 > p_top) & (p2 < p_top)
        if np.any(cross_top):
            # reset level 2 as p_top
            weight = np.log(p1[cross_top] / p_top[cross_top]) / \
                np.log(p1[cross_top] / p2[cross_top])
            q2[cross_top] = (1 - weight) * q1[cross_top] + \
                weight * q2[cross_top]
            p2[cross_top] = p_top[cross_top]

        # If within layer, update PW
        in_layer = (p1 <= p_bot) & (p2 >= p_top)
        PW[in_layer] += (1 / g) * 0.5 * (q1[in_layer] + q2[in_layer]) * \
            (p1[in_layer] - p2[in_layer])

    return PW.squeeze()  # remove dimensions of length 1


def saturation_fraction(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                        p_bot=None, p_top=None, phase='liquid',
                        vertical_axis=0):
    """
    Computes saturation fraction (a.k.a. column relative humidity) from
    profiles of pressure, temperature, and specific humidity.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        p_bot (float or ndarray, optional): pressure at which to start vertical
            integration (Pa) (default is None)
        p_top (float or ndarray, optional): pressure at which to stop vertical
            integration (Pa) (default is None)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        SF (float or ndarray): saturation fraction

    """

    # Compute saturation specific humidity
    omega = ice_fraction(T, phase=phase)
    qs = saturation_specific_humidity(p, T, phase=phase, omega=omega)
    if p_sfc is None:
        qs_sfc = None
    else:
        qs_sfc = saturation_specific_humidity(p_sfc, T_sfc, phase=phase,
                                              omega=omega)

    # Compute precipitable water
    PW = precipitable_water(p, q, p_sfc=p_sfc, q_sfc=q_sfc, p_bot=p_bot,
                            p_top=p_top, vertical_axis=vertical_axis)

    # Compute saturation precipitable water
    SPW = precipitable_water(p, qs, p_sfc=p_sfc, q_sfc=qs_sfc, p_bot=p_bot,
                             p_top=p_top, vertical_axis=vertical_axis)

    # Compute saturation fraction
    SF = PW / SPW

    return SF


def integrated_vapour_transport(p, q, u, v, p_sfc=None, q_sfc=None, u_sfc=None,
                                v_sfc=None, p_bot=None, p_top=None,
                                vertical_axis=0):
    """
    Computes components of integrated vapour transport (IVT) vector from
    profiles of pressure, specific humidity, and zonal and meridional wind.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        q (ndarray): specific humidity profile(s) (kg/kg)
        u (ndarray): eastward component of wind (m/s)
        v (ndarray): northward component of wind (m/s)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        u_sfc (float or ndarray, optional): eastward component of surface wind
            (m/s)
        v_sfc (float or ndarray, optional): northward component of surface wind
            (m/s)
        p_bot (float or ndarray, optional): pressure at which to start vertical
            integration (Pa) (default is None)
        p_top (float or ndarray, optional): pressure at which to stop vertical
            integration (Pa) (default is None)
        vertical_axis (int, optional): profile array axis corresponding to
            vertical dimension (default is 0)

    Returns:
        IVTu (float or ndarray): eastward component of IVT (kg/m/s)
        IVTv (float or ndarray): northward component of IVT (kg/m/s)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)
        u = np.moveaxis(u, vertical_axis, 0)
        v = np.moveaxis(v, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        q = np.atleast_2d(q).T
        u = np.atleast_2d(u).T
        v = np.atleast_2d(v).T

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        bottom = 'lowest level'
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        q_sfc = q[0]
        u_sfc = u[0]
        v_sfc = v[0]
    else:
        bottom = 'surface'
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    q_sfc = np.atleast_1d(q_sfc)
    u_sfc = np.atleast_1d(u_sfc)
    v_sfc = np.atleast_1d(v_sfc)

    # If p_bot is not specified, use surface
    if p_bot is None:
        p_bot = p_sfc
    elif np.isscalar(p_bot):
        p_bot = np.full_like(p_sfc, p_bot)

    # If p_top is not specified, use top level
    if p_top is None:
        p_top = p[-1]
    elif np.isscalar(p_top):
        p_top = np.full_like(p_sfc, p_top)

    # Check if bottom of layer is above top of layer
    if np.any(p_bot < p_top):
        n_pts = np.count_nonzero(p_bot < p_top)
        raise ValueError(f'p_bot is above p_top for {n_pts} points')

    # Check if bottom of layer is below surface
    if np.any(p_bot > p_sfc):
        n_pts = np.count_nonzero(p_bot > p_sfc)
        warnings.warn(f'p_bot is below {bottom} for {n_pts} points')

    # Check if top of layer is above highest level
    if np.any(p_top < p[-1]):
        n_pts = np.count_nonzero(p_top < p[-1])
        warnings.warn(f'p_top is above highest level for {n_pts} points')

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Initialise level 2 fields
    p2 = p_sfc.copy()
    q2 = q_sfc.copy()
    u2 = u_sfc.copy()
    v2 = v_sfc.copy()

    # Initialise IVT components
    IVTu = np.zeros_like(p_sfc)
    IVTv = np.zeros_like(p_sfc)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        q1 = q2.copy()
        u1 = u2.copy()
        v1 = v2.copy()
        if np.all(p1 <= p_top):
            # can break out of loop
            break

        # Update level 2 fields
        above_sfc = (p[k] <= p_sfc)
        p2 = np.where(above_sfc, p[k], p_sfc)
        q2 = np.where(above_sfc, q[k], q_sfc)
        u2 = np.where(above_sfc, u[k], u_sfc)
        v2 = np.where(above_sfc, v[k], v_sfc)
        if np.all(p2 >= p_bot):
            # can skip this level
            continue

        # If crossing bottom of layer, reset level 1
        cross_bot = (p1 > p_bot) & (p2 < p_bot)
        if np.any(cross_bot):
            weight = np.log(p1[cross_bot] / p_bot[cross_bot]) / \
                np.log(p1[cross_bot] / p2[cross_bot])
            q1[cross_bot] = (1 - weight) * q1[cross_bot] + \
                weight * q2[cross_bot]
            u1[cross_bot] = (1 - weight) * u1[cross_bot] + \
                weight * u2[cross_bot]
            v1[cross_bot] = (1 - weight) * v1[cross_bot] + \
                weight * v2[cross_bot]
            p1[cross_bot] = p_bot[cross_bot]

        # If crossing top of layer, reset level 2
        cross_top = (p1 > p_top) & (p2 < p_top)
        if np.any(cross_top):
            weight = np.log(p1[cross_top] / p_top[cross_top]) / \
                np.log(p1[cross_top] / p2[cross_top])
            q2[cross_top] = (1 - weight) * q1[cross_top] + \
                weight * q2[cross_top]
            u2[cross_top] = (1 - weight) * u1[cross_top] + \
                weight * u2[cross_top]
            v2[cross_top] = (1 - weight) * v1[cross_top] + \
                weight * v2[cross_top]
            p2[cross_top] = p_top[cross_top]

        # If within layer, update IVT components
        in_layer = (p1 <= p_bot) & (p2 >= p_top)
        IVTu[in_layer] += (1 / g) * 0.5 * (
            q1[in_layer] * u1[in_layer] + q2[in_layer] * u2[in_layer]
            ) * (p1[in_layer] - p2[in_layer])
        IVTv[in_layer] += (1 / g) * 0.5 * (
            q1[in_layer] * v1[in_layer] + q2[in_layer] * v2[in_layer]
            ) * (p1[in_layer] - p2[in_layer])

    return IVTu.squeeze(), IVTv.squeeze()  # remove dimensions of length 1


def geopotential_height(p, T, q, p_sfc=None, T_sfc=None, q_sfc=None,
                        Z_sfc=None, vertical_axis=0):
    """
    Computes geopotential height from profiles of pressure, temperature, and
    specific humidity using the hypsometric equation.

    Args:
        p (ndarray): pressure profile(s) (Pa)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray, optional): surface pressure (Pa)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        Z_sfc (float or ndarray, optional): surface geopotential height (m)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)

    Returns:
        Z (float or ndarray): geopotential height (m)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        p = np.moveaxis(p, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if p.ndim == 1:
        p = np.atleast_2d(p).T  # transpose to preserve vertical axis
        T = np.atleast_2d(T).T
        q = np.atleast_2d(q).T

    # If surface-level fields not provided, use lowest level values
    if p_sfc is None:
        k_start = 1  # start loop from second level
        p_sfc = p[0]
        T_sfc = T[0]
        q_sfc = q[0]
        Z_sfc = np.zeros_like(p_sfc)  # assumes surface is at MSL
    else:
        k_start = 0  # start loop from first level

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)
    Z_sfc = np.atleast_1d(Z_sfc)

    # Note the number of vertical levels
    n_lev = p.shape[0]

    # Create array for geopotential height
    Z = np.zeros_like(p)

    # Initialise level 2 pressure and geopotential height
    p2 = p_sfc.copy()
    Z2 = Z_sfc.copy()

    # Compute level 2 virtual temperature
    Tv2 = virtual_temperature(T_sfc, q_sfc)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        p1 = p2.copy()
        Z1 = Z2.copy()
        Tv1 = Tv2.copy()

        # Find points above the surface
        above_sfc = (p[k] <= p_sfc)
        if np.any(above_sfc):

            # Update level 2 pressure
            p2[above_sfc] = p[k][above_sfc]

            # Compute level 2 virtual temperature
            Tv2[above_sfc] = virtual_temperature(T[k][above_sfc],
                                                 q[k][above_sfc])

            # Compute level 2 geopotential height
            Z2[above_sfc] = Z1[above_sfc] + (Rd / g) * \
                0.5 * (Tv1[above_sfc] + Tv2[above_sfc]) * \
                np.log(p1[above_sfc] / p2[above_sfc])

            # Save geopotential height for this level
            Z[k][above_sfc] = Z2[above_sfc]

    return Z.squeeze()  # remove dimensions of length 1


def hydrostatic_pressure(z, T, q, p_sfc, z_sfc=None, T_sfc=None, q_sfc=None,
                         vertical_axis=0):
    """
    Computes hydrostatic pressure from profiles of height, temperature, and
    specific humidity and surface pressure using the hypsometric equation.

    Args:
        z (ndarray): height profile(s) (m)
        T (ndarray): temperature profile(s) (K)
        q (ndarray): specific humidity profile(s) (kg/kg)
        p_sfc (float or ndarray): surface pressure (Pa)
        z_sfc (float or ndarray, optional): surface height (m)
        T_sfc (float or ndarray, optional): surface temperature (K)
        q_sfc (float or ndarray, optional): surface specific humidity (kg/kg)
        vertical_axis (int, optional): profile array axis corresponding to 
            vertical dimension (default is 0)

    Returns:
        P (float or ndarray): hydrostatic pressure (Pa)

    """

    # Reorder profile array dimensions if needed
    if vertical_axis != 0:
        z = np.moveaxis(z, vertical_axis, 0)
        T = np.moveaxis(T, vertical_axis, 0)
        q = np.moveaxis(q, vertical_axis, 0)

    # Make sure that profile arrays are at least 2D
    if z.ndim == 1:
        z = np.atleast_2d(z).T  # transpose to preserve vertical axis
        T = np.atleast_2d(T).T
        q = np.atleast_2d(q).T

    # If surface-level fields not provided, use lowest level values
    if T_sfc is None:
        k_start = 1  # start loop from second level
        T_sfc = T[0]
        q_sfc = q[0]
        z_sfc = z[0]
    else:
        k_start = 0  # start loop from first level
        if z_sfc is None:
            z_sfc = np.zeros_like(p_sfc)  # assumes height AGL

    # Make sure that surface fields are at least 1D
    p_sfc = np.atleast_1d(p_sfc)
    z_sfc = np.atleast_1d(z_sfc)
    T_sfc = np.atleast_1d(T_sfc)
    q_sfc = np.atleast_1d(q_sfc)

    # Note the number of vertical levels
    n_lev = z.shape[0]

    # Create array for hydrostatic pressure
    P = np.zeros_like(z)

    # Initialise level 2 height and hydrostatic pressure
    z2 = z_sfc.copy()
    P2 = p_sfc.copy()

    # Compute level 2 virtual temperature
    Tv2 = virtual_temperature(T_sfc, q_sfc)

    # Loop over levels
    for k in range(k_start, n_lev):

        # Update level 1 fields
        z1 = z2.copy()
        P1 = P2.copy()
        Tv1 = Tv2.copy()

        # Find points above the surface
        above_sfc = (z[k] >= z_sfc)
        if np.any(above_sfc):

            # Update level 2 height
            z2[above_sfc] = z[k][above_sfc]

            # Compute level 2 virtual temperature
            Tv2[above_sfc] = virtual_temperature(T[k][above_sfc],
                                                 q[k][above_sfc])

            # Compute level 2 hydrostatic pressure
            P2[above_sfc] = P1[above_sfc] * np.exp(
                -1 * g * (z2[above_sfc] - z1[above_sfc]) /
                (Rd * 0.5 * (Tv1[above_sfc] + Tv2[above_sfc]))
            )

            # Save hydrostatic pressure for this level
            P[k][above_sfc] = P[above_sfc]

    return P.squeeze()  # remove dimensions of length 1
