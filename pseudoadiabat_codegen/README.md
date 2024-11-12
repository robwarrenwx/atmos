Files in this folder pertain to the approach to representing pseudoadiabats
originally presented by Moisseeva and Stull (2017). They used high-order
polynomials to model (1) wet-bulb potential temperature (WBPT) as a function
of pressure and temperature and (2) temperature as a function of pressure and 
WBPT. Their polynomial fits have been reproduced using a more accurate equation
for the pseudoadiabatic lapse rate. The resulting polynomial coefficients are
contained in `alpha.txt`, `beta.txt`, `a.txt`, and `b.txt`. The fit for WBPT
is valid for temperatures in the range -100 <= T <= 50degC. The fit for
temperature is valid for WBPTs in the range -50 <= thw <= 50degC. Both fits
are valid for pressures in the range 1100 <= p <= 100 hPa.

To generate `pseudoadiabat.py`, which contains the functions for evaluating
thw(p,T) and  T(p,thw), pseudoadiabat_codegen.py should be run from this 
directory as follows:

`python pseudoadiabat_codegen.py > ../atmos/pseudoadiabat.py`

The polynomial fits can be modified by running `fit_pseudoadiabat.py`. This
function takes as input the range and increments for p, T, and thw, together
with the order of the polynomials, and outputs the polynomial coefficients.
The function also prints error statistics for the polynomial fits.

Note that this method has been found to only produce acceptable results for
liquid-only pseudoadiabats. For mixed-phase and ice pseudoadiabats, it seems
to produce unacceptably large errors.

References:

Moisseeva, N. and Stull, R., 2017. A noniterative approach to modelling moist
    thermodynamics. Atmospheric Chemistry and Physics, 17, 15037-15043.
