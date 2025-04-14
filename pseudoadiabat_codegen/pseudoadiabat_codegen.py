import numpy as np

# Read in coefficients
alpha = np.loadtxt("alpha_coeff.txt", dtype=np.str_)
beta = np.loadtxt("beta_coeff.txt", dtype=np.str_)
a = np.loadtxt("a_coeff.txt", dtype=np.str_)
b = np.loadtxt("b_coeff.txt", dtype=np.str_)

# Flip ordering so coefficients apply in order of increasing degree
# (e.g. [1, 2, 3] for a polynomial 1 + 2*x + 3*x**2)
alpha = np.flip(alpha)
beta = np.flip(beta)
a = np.flip(a)
b = np.flip(b)

# Sanity check the loaded polynomial coefficient array shapes:
assert len(alpha.shape) == 2  # should be a 2D array
assert len(a.shape) == 2  # should be a 2D array
assert len(beta.shape) == 1  # should be a 1D array
assert len(b.shape) == 1  # should be a 1D array
assert alpha.shape == a.shape
assert beta.shape == b.shape
assert a.shape[1] == b.shape[0]  # should have common length along these axes


def horner_polyval1_inline(p, xvarname):
    """
    Code generation function.

    Returns a code string representing the Horner's method evaluation of
    an Nth degree polynomial for `xvarname` with constant coefficients `p`.

    Args:
        p (iterable; length N): coefficients for Nth degree polynomial
        xvarname (string): name of the indeterminate variable

    Returns: string, which can be parsed & evaluated under the assumption
             `xvarname` is a variable assigned a numeric value
    """

    num_closing_brackets = len(p) - 1
    s = f"{p[0]}"
    for c in p[1:]:
        s += f" + {xvarname}*({c}"
    s += ")" * num_closing_brackets
    return s


def bipoly_expand(invar1, invar2, cvar, resultvar, a_arr, b_arr, indent=4):
    """
    Code generation function.

    Returns an in-line code string representing the equivalent of:
        cvar = Polynomial(b_arr)(invar2)
        kappa = [Polynomial(a)(invar1) for a in a_arr]
        resultvar = Polynomial(kappa)(cvar)
    where Polynomial is numpy.polynomial.polynomial.Polynomial. This is
    sometimes referred to as a "bivariate" polynomial.

    Args:
        invar1 (string): name of the 1st indeterminate variable
        invar2 (string): name of the 2nd indeterminate variable
        cvar (string): name to assign to variable to hold the result
            of polynomial evaluation of invar2
        resultvar (string): name to assign to variable to hold final
            "bivariate" polynomial evaluation result
        a_arr (iterable; shape (M,N)): coefficients for kappa array,
            i.e. M evaluations of invar1 N degree polynomials
        b_arr (iterable; shape (N,)):  coefficients for invar2 N degree
            polynomial assigned to cvar
        indent (int, optional): leading spaces for string indent

    Returns: string, which can be parsed & evaluated under the assumption
             `invar1` and `invar2` are variables assigned a numeric value
    """
    ignore_line_len = "  # noqa: E501"  # flake8 codegen warning suppression
    s = f"{cvar} = " + horner_polyval1_inline(b_arr, invar2)
    s += ignore_line_len + "\n" + " " * indent
    kappa = [horner_polyval1_inline(a_row, invar1) for a_row in a_arr]
    s += f"{resultvar} = " + horner_polyval1_inline(kappa, cvar)
    s += ignore_line_len
    return s


wbpt_poly = bipoly_expand("T_", "p_", "Tref", "thw", alpha, beta)
temp_poly = bipoly_expand("thw_", "p_", "thref", "T", a, b)

python_code = f"""\
# *** This file is generated by pseudoadiabat_codegen/pseudoadiabat_codegen.py ***
# *** Please ensure any updates are made in pseudoadiabat_codegen.py           ***
import numpy as np


def wbpt(p, T):
    \"""
    Computes the wet-bulb potential temperature (WBPT) thw of the
    pseudoadiabat that passes through pressure p and temperature T.

    Uses polynomial approximations from Moisseeva and Stull (2017)
    with revised coefficients.

    Moisseeva, N. and Stull, R., 2017. A noniterative approach to
        modelling moist thermodynamics. Atmospheric Chemistry and
        Physics, 17, 15037-15043.

    Args:
        p: pressure (Pa)
        T: temperature (K)

    Returns:
        thw: wet-bulb potential temperature (K)

    \"""

    # Convert scalar inputs to arrays and promote to float64
    p = np.atleast_1d(p).astype(np.float64)
    T = np.atleast_1d(T).astype(np.float64)

    # Convert p to hPa and T to degC
    p_ = p / 100.
    T_ = T - 273.15

    # Compute theta-w using Eq. 4-6 from Moisseeva & Stull 2017
    {wbpt_poly}

    # Mask points outside the polynomial fits
    mask = (T_ < -100.) | (T_ > 50.) | (p_ > 1100.) | (p_ < 50.)
    thw[mask] = np.nan
    
    # Return theta-w converted to K
    return thw + 273.15


def temp(p, thw):
    \"""
    Computes the temperature T at pressure p on a pseudoadiabat with
    wet-bulb potential temperature thw.

    Uses polynomial approximations from Moisseeva and Stull (2017) with
    revised coefficients.

    Moisseeva, N. and Stull, R., 2017. A noniterative approach to
        modelling moist thermodynamics. Atmospheric Chemistry and
        Physics, 17, 15037-15043.

    Args:
        p: pressure (Pa)
        thw: wet-bulb potential temperature (K)

    Returns:
        T: temperature (K)

    \"""

    # Convert scalar inputs to arrays and promote to float64
    p = np.atleast_1d(p).astype(np.float64)
    thw = np.atleast_1d(thw).astype(np.float64)

    # Convert p to hPa and theta-w to degC
    p_ = p / 100.
    thw_ = thw - 273.15

    # Compute T using Eq. 1-3 from Moisseeva & Stull 2017
    {temp_poly}

    # Mask points outside the polynomial fits
    mask = (thw_ < -70.) | (thw_ > 50.) | (p_ > 1100.) | (p_ < 50.)
    T[mask] = np.nan

    # Return T converted to K
    return T + 273.15\
"""

print(python_code)
