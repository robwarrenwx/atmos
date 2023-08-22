Files in this folder pertain to the approach to representing pseudoadiabats
originally presented by Moisseeva and Stull (2017). They used high-order
polynomials to model (1) wet-bulb potential temperature (WBPT) as a function
of pressure and temperature and (2) temperature as a function of pressure and 
WBPT. Their fits have been reproduced using a slightly more accurate equation
for the pseudoadiabatic lapse rate from Bakhshaii and Stull (2013). The 
resulting polynomial coefficients are contained in `alpha.txt`, `beta.txt`, 
`a.txt`, and `b.txt`. The fit for WBPT is valid for temperatures in the range
-100 <= T <= 50degC. The fit for T is valid for WBPTs in the range
-50 <= thw <= 50degC. Both fits are valid for pressures in the range
1100 <= p <= 100 hPa.

To generate `pseudoadiabat.py`, which contains the functions for evaluating
thw(p,T) and  T(p,thw), pseudoadiabat_codegen.py should be run from this 
directory as follows:

`python pseudoadiabat_codegen.py > ../atmos/pseudoadiabat.py`

References:

Moisseeva, N. and Stull, R., 2017. A noniterative approach to modelling moist
    thermodynamics. Atmospheric Chemistry and Physics, 17, 15037-15043.

Bakhshaii, A. and Stull, R., 2013. Saturated Pseudoadiabats - A noniterative 
    approximation. Journal of Applied Meteorology and Climatology, 52, 5-15.
