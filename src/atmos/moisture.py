"""
Functions for converting between the following moisture variables:
* specific humidity, q
* mixing ratio, r
* vapour pressure, e
* relative humidity, RH
* dewpoint temperature, Td
* wet-bulb temperature, Tw

"""

import numpy as np
from atmos.constant import Rv, eps, cpd, cpv, cpv, cpl, cpi, Lv0, Ls0, T0
from atmos.thermo import precision, max_n_iter
from atmos.thermo import latent_heat_of_vaporisation
from atmos.thermo import latent_heat_of_sublimation
from atmos.thermo import mixed_phase_latent_heat
from atmos.thermo import saturation_vapour_pressure
from atmos.thermo import saturation_specific_humidity
from atmos.thermo import saturation_mixing_ratio
from atmos.thermo import _lambertw
from atmos.thermo import ice_fraction
from atmos.thermo import mixing_ratio as \
    mixing_ratio_from_specific_humidity
from atmos.thermo import vapour_pressure as \
    vapour_pressure_from_specific_humidity
from atmos.thermo import relative_humidity as \
    relative_humidity_from_specific_humidity
from atmos.thermo import dewpoint_temperature as \
    dewpoint_temperature_from_specific_humidity
from atmos.thermo import wet_bulb_temperature as \
    wet_bulb_temperature_from_specific_humidity


def specific_humidity_from_mixing_ratio(r):
    """
    Computes specific humidity from water vapour mixing ratio.

    Args:
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    q = r / (1 + r)

    return q


def specific_humidity_from_vapour_pressure(p, e):
    """
    Computes specific humidity from pressure and vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    q = eps * e / (p - (1 - eps) * e)

    return q


def specific_humidity_from_relative_humidity(
    p, T, RH, phase='liquid', omega=0.0
    ):
    """
    Computes specific humidity from pressure, temperature, and relative
    humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)
        
    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es
    q = specific_humidity_from_vapour_pressure(p, e)

    return q


def specific_humidity_from_dewpoint_temperature(
    p, Td, phase='liquid', omega=0.0
    ):
    """
    Computes specific humidity from pressure and dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Td (float or ndarray): dewpoint temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """    
    q = saturation_specific_humidity(p, Td, phase=phase, omega=omega)

    return q


def specific_humidity_from_wet_bulb_temperature(
    p, T, Tw, phase='liquid', omega=0.0
    ):
    """
    Computes specific humidity from pressure, temperature, and wet-bulb
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        q (float or ndarray): specific humidity (kg/kg)

    """
    e = vapour_pressure_from_wet_bulb_temperature(
        p, T, Tw, phase=phase, omega=omega
    )
    q = specific_humidity_from_vapour_pressure(p, e)

    return q


def mixing_ratio_from_vapour_pressure(p, e):
    """
    Computes mixing ratio from pressure and vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        e (float or ndarray): vapour pressure (Pa)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    r = eps * e / (p - e)

    return r


def mixing_ratio_from_relative_humidity(p, T, RH, phase='liquid', omega=0.0):
    """
    Computes mixing ratio from pressure, temperature, and relative humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es
    r = mixing_ratio_from_vapour_pressure(p, e)

    return r


def mixing_ratio_from_dewpoint_temperature(p, Td, phase='liquid', omega=0.0):
    """
    Computes mixing ratio from pressure and dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        Td (float or ndarray): dewpoint temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """    
    r = saturation_mixing_ratio(p, Td, phase=phase, omega=omega)

    return r


def mixing_ratio_from_wet_bulb_temperature(
    p, T, Tw, phase='liquid', omega=0.0
    ):
    """
    Computes mixing ratio from pressure and wet-bulb temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        r (float or ndarray): mixing ratio (kg/kg)

    """
    e = vapour_pressure_from_wet_bulb_temperature(
        p, T, Tw, phase=phase, omega=omega
    )
    r = mixing_ratio_from_vapour_pressure(p, e)

    return r


def vapour_pressure_from_mixing_ratio(p, r):
    """
    Computes vapour pressure from pressure and mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        r (float or ndarray): mixing ratio (kg/kg)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = p * r / (r + eps)

    return e


def vapour_pressure_from_relative_humidity(T, RH, phase='liquid', omega=0.0):
    """
    Computes vapour pressure from temperature and relative humidity.

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    e = RH * es

    return e


def vapour_pressure_from_dewpoint_temperature(Td, phase='liquid', omega=0.0):
    """
    Computes vapour pressure from dewpoint temperature.

    Args:
        Td (float or ndarray): dewpoint temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """
    e = saturation_vapour_pressure(Td, phase=phase, omega=omega)

    return e


def vapour_pressure_from_wet_bulb_temperature(
    p, T, Tw, phase='liquid', omega=0.0
    ):
    """
    Computes vapour pressure from pressure, temperature, and wet-bulb
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        e (float or ndarray): vapour pressure (Pa)

    """

    # Compute saturation vapour pressure at Tw
    es_Tw = saturation_vapour_pressure(Tw, phase=phase, omega=omega)

    if phase == 'liquid':

        # Compute latent heat of vaporisation at Tw
        Lv_Tw = latent_heat_of_vaporisation(Tw)

        # Compute vapour pressure
        e = p * (eps * Lv_Tw * es_Tw - cpd * (T - Tw) * (p - es_Tw)) / \
            (eps * Lv_Tw * p + (eps * cpv - cpd) * (T - Tw) * (p - es_Tw))

    if phase == 'ice':

        # Compute latent heat of sublimation at Tw
        Ls_Tw = latent_heat_of_sublimation(Tw)

        # Compute vapour pressure
        e = p * (eps * Ls_Tw * es_Tw - cpd * (T - Tw) * (p - es_Tw)) / \
            (eps * Ls_Tw * p + (eps * cpv - cpd) * (T - Tw) * (p - es_Tw))

    else:

        # Compute mixed-phase latent heat at Tw
        Lx_Tw = mixed_phase_latent_heat(Tw, omega)

        # Compute vapour pressure
        e = p * (eps * Lx_Tw * es_Tw - cpd * (T - Tw) * (p - es_Tw)) / \
            (eps * Lx_Tw * p + (eps * cpv - cpd) * (T - Tw) * (p - es_Tw))

    return e
    
    
def relative_humidity_from_mixing_ratio(p, T, r, phase='liquid', omega=0.0):
    """
    Computes relative humidity from pressure, temperature, and mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = vapour_pressure_from_mixing_ratio(p, r)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH
    
    
def relative_humidity_from_vapour_pressure(T, e, phase='liquid', omega=0.0):
    """
    Computes relative humidity from temperature and vapour pressure.

    Args:
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH
    
    
def relative_humidity_from_dewpoint_temperature(
    T, Td, phase='liquid', omega=0.0
    ):
    """
    Computes relative humidity from temperature and dewpoint temperature.

    Args:
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewpoint temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """
    e = saturation_vapour_pressure(Td, phase=phase, omega=omega)
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH


def relative_humidity_from_wet_bulb_temperature(
    p, T, Tw, phase='liquid', omega=0.0
    ):
    """
    Computes relative humidity (RH) from pressure, temperature, and isobaric
    wet-bulb temperature.

    Note that for phase='mixed', the RH produced by this function considers
    saturation with respect to the mixed-phase wet-bulb temperature, rather
    than the saturation-point temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)

    Returns:
        RH (float or ndarray): relative humidity (fraction)

    """

    # Compute the relative humidity
    e = vapour_pressure_from_wet_bulb_temperature(
        p, T, Tw, phase=phase, omega=omega
    )
    es = saturation_vapour_pressure(T, phase=phase, omega=omega)
    RH = e / es

    return RH
    

def dewpoint_temperature_from_mixing_ratio(
    p, T, r, phase='liquid', limit=True
    ):
    """
    Computes dewpoint temperature from pressure, temperature, and mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        limit (bool, optional): flag indicating whether to limit the dewpoint
            temperature to ensure that it does not exceed the temperature
            (default is True)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Td = dewpoint_temperature_from_specific_humidity(
        p, T, q, phase=phase, limit=limit
    )

    return Td


def dewpoint_temperature_from_vapour_pressure(
    p, T, e, phase='liquid', limit=True
    ):
    """
    Computes dewpoint temperature from pressure, temperature, and vapour
    pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        limit (bool, optional): flag indicating whether to limit the dewpoint
            temperature to ensure that it does not exceed the temperature
            (default is True)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Td = dewpoint_temperature_from_specific_humidity(
        p, T, q, phase=phase, limit=limit
    )

    return Td


def dewpoint_temperature_from_relative_humidity(
    T, RH, phase='liquid', limit=True
    ):
    """
    Computes dewpoint temperature from temperature and relative humidity.

    Args:
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        limit (bool, optional): flag indicating whether to limit the dewpoint
            temperature to ensure that it does not exceed the temperature
            (default is True)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """
    if phase == 'liquid':

        # Compute dewpoint temperature using Lambert-W function
        # (Eq. 38 and 40 from Warren 2025; cf. Eq. 5-6 from Romps 2021)
        beta = (cpl - cpv) / Rv
        alpha = -(1 / beta) * (Lv0 + (cpl - cpv) * T0) / Rv
        fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
        #W = lambertw(fn, k=-1).real
        W = _lambertw(fn)
        Td = alpha / W

    elif phase == 'ice':

        # Compute frost-point temperature using Lambert-W function
        # (Eq. 39 and 41 from Warren 2025; cf. Eq. 7-8 from Romps 2021)
        beta = (cpi - cpv) / Rv
        alpha = -(1 / beta) * (Ls0 + (cpi - cpv) * T0) / Rv
        fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
        #W = lambertw(fn, k=-1).real
        W = _lambertw(fn)
        Td = alpha / W

    elif phase == 'mixed':

        # Intialise the saturation point temperature as the temperature
        Td = T

        # Iterate to convergence
        converged = False
        count = 0
        while not converged:

            # Update the previous Ts value
            Td_prev = Td

            # Compute the ice fraction
            omega = ice_fraction(Td)

            # Compute mixed-phase specific heat
            # (Eq. 30 from Warren 2025)
            cpx = (1 - omega) * cpl + omega * cpi

            # Compute mixed-phase latent heat at the triple point
            # (Eq. 31 from Warren 2025)
            Lx0 = (1 - omega) * Lv0 + omega * Ls0

            # Compute saturation-point temperature using Lambert-W function
            # (Eq. 42-43 from Warren 2025)
            beta = (cpx - cpv) / Rv
            alpha = -(1 / beta) * (Lx0 + (cpx - cpv) * T0) / Rv
            fn = np.power(RH, (1 / beta)) * (alpha / T) * np.exp(alpha / T)
            #W = lambertw(fn, k=-1).real
            W = _lambertw(fn)
            Td = alpha / W

            # Check if solution has converged
            if np.nanmax(np.abs(Td - Td_prev)) < precision:
                converged = True
            else:
                count += 1
                if count == max_n_iter:
                    print(f"Saturation-point temperature not converged after {max_n_iter} iterations")
                    break

    else:

        raise ValueError("phase must be one of 'liquid', 'ice', or 'mixed'")

    if limit:
        Td = np.minimum(Td, T)

    return Td


def dewpoint_temperature_from_wet_bulb_temperature(
    p, T, Tw, phase='liquid', omega=0.0, limit=True
    ):
    """
    Computes dewpoint temperature from pressure, temperature, and wet-bulb
    temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw (float or ndarray): wet-bulb temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)
        limit (bool, optional): flag indicating whether to limit the dewpoint
            temperature to ensure that it does not exceed the temperature
            (default is True)

    Returns:
        Td (float or ndarray): dewpoint temperature (K)

    """

    # Compute the vapour pressure
    e = vapour_pressure_from_wet_bulb_temperature(
        p, T, Tw, phase=phase, omega=omega
    )

    # Compute the specific humidity
    q = specific_humidity_from_vapour_pressure(p, e)

    # Compute the dewpoint temperature
    Td = dewpoint_temperature_from_specific_humidity(
        p, T, q, phase=phase, limit=limit
    )

    return Td


def wet_bulb_temperature_from_mixing_ratio(
    p, T, r, phase='liquid', isobaric_method='Romps'
    ):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    mixing ratio.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        r (float or ndarray): mixing ratio (kg/kg)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_mixing_ratio(r)
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase=phase, variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def wet_bulb_temperature_from_vapour_pressure(
    p, T, e, phase='liquid', isobaric_method='Romps'
    ):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    vapour pressure.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        e (float or ndarray): vapour pressure (Pa)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_vapour_pressure(p, e)
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase=phase, variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def wet_bulb_temperature_from_relative_humidity(
    p, T, RH, phase='liquid', omega=0.0, isobaric_method='Romps'
    ):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    relative humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH (float or ndarray): relative humidity (fraction)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_relative_humidity(
        p, T, RH, phase=phase, omega=omega
    )
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase=phase, variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def wet_bulb_temperature_from_dewpoint_temperature(
    p, T, Td, phase='liquid', omega=0.0, isobaric_method='Romps'
    ):
    """
    Computes isobaric wet-bulb temperature from pressure, temperature, and
    dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Td (float or ndarray): dewpoint temperature (K)
        phase (str, optional): condensed water phase (valid options are
            'liquid', 'ice', or 'mixed'; default is 'liquid')
        omega (float or ndarray, optional): ice fraction at saturation
            (default is 0.0)
        isobaric_method (str, optional): method used to calculate isobaric
            wet-bulb temperature (valid options are 'Warren' or 'Romps';
            default is 'Romps')

    Returns:
        Tw (float or ndarray): isobaric wet-bulb temperature (K)

    """
    q = specific_humidity_from_dewpoint_temperature(
        p, Td, phase=phase, omega=omega
    )
    Tw = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase=phase, variant='isobaric',
        isobaric_method=isobaric_method
    )

    return Tw


def convert_relative_humidity_phase(p, T, RH_in, phase_in, phase_out):
    """
    Converts phase of relative humidity.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        RH_in (float or ndarray): input relative humidity (fraction)
        phase_in (str): input condensed water phase (valid options are
            'liquid', 'ice', or 'mixed')
        phase_out (str): output condensed water phase (valid options are
            'liquid', 'ice', or 'mixed')

    Returns:
        RH_out (float or ndarray): output relative humidity (fraction)

    """

    # Compute the input ice fraction
    if phase_in == 'mixed':
        Td_in = dewpoint_temperature_from_relative_humidity(
            T, RH_in, phase='mixed'
        )
        omega_in = ice_fraction(Td_in, phase='mixed')
    else:
        omega_in = ice_fraction(T, phase=phase_in)

    # Compute the output ice fraction
    if phase_out == 'mixed':
        q = specific_humidity_from_relative_humidity(
            p, T, RH_in, phase=phase_in, omega=omega_in
        )
        Td_out = dewpoint_temperature_from_specific_humidity(
            p, T, q, phase='mixed'
        )
        omega_out = ice_fraction(Td_out, phase='mixed')
    else:
        omega_out = ice_fraction(T, phase=phase_out)

    # Compute output relative humidity
    es_in = saturation_vapour_pressure(T, phase=phase_in, omega=omega_in)
    es_out = saturation_vapour_pressure(T, phase=phase_out, omega=omega_out)
    RH_out = RH_in * es_in / es_out

    return RH_out


def convert_dewpoint_temperature_phase(
    p, T, Td_in, phase_in, phase_out, limit=True
    ):
    """
    Converts phase of dewpoint temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Td_in (float or ndarray): input dewpoint temperature (K)
        phase_in (str): input condensed water phase (valid options are
            'liquid', 'ice', or 'mixed')
        phase_out (str): output condensed water phase (valid options are
            'liquid', 'ice', or 'mixed')
        limit (bool, optional): flag indicating whether to limit the dewpoint
            temperature to ensure that it does not exceed the temperature
            (default is True)

    Returns:
        Td_out (float or ndarray): output dewpoint temperature (K)

    """

    # Compute the input ice fraction
    omega_in = ice_fraction(Td_in, phase=phase_in)

    # Compute specific humidity
    q = specific_humidity_from_dewpoint_temperature(
        p, Td_in, phase=phase_in, omega=omega_in
    )

    # Compute the output dewpoint temperature
    Td_out = dewpoint_temperature_from_specific_humidity(
        p, T, q, phase=phase_out, limit=limit
    )

    return Td_out


def convert_wet_bulb_temperature_phase(p, T, Tw_in, phase_in, phase_out):
    """
    Converts phase of wet-bulb temperature.

    Args:
        p (float or ndarray): pressure (Pa)
        T (float or ndarray): temperature (K)
        Tw_in (float or ndarray): input wet-bulb temperature (K)
        phase_in (str): input condensed water phase (valid options are
            'liquid', 'ice', or 'mixed')
        phase_out (str): output condensed water phase (valid options are
            'liquid', 'ice', or 'mixed')
    
    Returns:
        Tw_out (float or ndarray): output wet-bulb temperature (K)

    """

    # Compute the input ice fraction
    omega_in = ice_fraction(Tw_in, phase=phase_in)

    # Compute specific humidity
    q = specific_humidity_from_wet_bulb_temperature(
        p, T, Tw_in, phase=phase_in, omega=omega_in
    )

    # Compute the output wet-bulb temperature
    Tw_out = wet_bulb_temperature_from_specific_humidity(
        p, T, q, phase=phase_out
    )

    return Tw_out
