# atmos: fast and accurate calculations for applications in atmospheric science


## What is atmos?

__atmos__ is a Python library for computing physical quantities commonly used in atmospheric science. The code comprises the following modules:
* _thermo_ - functions for calculating various thermodynamic quantities (density, wet-bulb temperature, potential temperature, etc.)
* _moisture_ - functions for convecting between different measures of atmospheric moisture (specific humidity, relative humidity, dewpoint temperature, etc.)
* _parcel_ - functions for performing adiabatic and pseudoadiabatic parcel ascents to calculate convective available potential energy (CAPE) and convective inhibition (CIN) for surface-based, mixed-layer, most-unstable, and effective parcels
* _kinematic_ - functions for converting between wind speed/direction and wind components (u and v) and for calculating kinematic quantities (bulk wind difference, storm-relative helicity, etc.)
* _pseudoadiabat_ - functions for performing fast pseudoadiabatic calculations (see below)
* _utils_ - generic functions for peforming interpolation and layer averaging of scalar and vector quantities
* _constant_ - specifies physical constants used in the code

The functions in __atmos__ are designed to work with numpy arrays, though many of them can also be applied to scalar variables. This makes __atmos__ well suited for processing gridded numerical model output, as well as observational data.


## Advantages of atmos

__atmos__ offers several key advantages over existing libraries for calculating meteorological variables (such as [MetPy](https://unidata.github.io/MetPy/latest/index.html)).

#### 1. Accuracy and physical consistency

All of the thermodynamic equations in __atmos__ are analytical and derived from a common set of assumptions, known as the _Rankine-Kirchhoff_ approximations [(Romps, 2021)](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.4154). No empircal equations are used in __atmos__. The core assumptions of the Rankine-Kirchhoff approximations are: (i) ideal gases, (ii), constant specific heat capacities, and (iii) zero volume of condensates (liquid and ice). Together, these allow for the derivation of highly accurate analytical equations for moist thermodynamics.

#### 2. Speed of calculations

All of the functions in __atmos__ are vectorised to allow for fast processing of multidimensional arrays. To speed up pseudoadiabatic parcel calculations, __atmos__ uses high-order polynomial fits to (i) parcel temperature as a function of wet-bulb potential temperature (WBPT) and pressure and (ii) WBPT as a function of parcel temperature and pressure following [Moisseeva and Stull (2017)](https://acp.copernicus.org/articles/17/15037/2017/). This is also the basis for the Noniterative Evaluation of Wet-bulb Temperature (NEWT) method ([Rogers and Warren, 2023](https://doi.org/10.22541/essoar.170560423.39769387/v1)).

#### 3. Representation of saturation

Another novel feature in __atmos__ is the treatment of saturation, which can be represented with respect to liquid, ice, or a combination of the two (via the `phase` argument) following [Warren (2025)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4866). When using the mixed-phase option (`phase="mixed"`), liquid water and ice are assumed to coexist between temperatures of 0°C and -20°C, with the ice fraction (denoted `omega` in the code) increasing nonlinearly from 0 to 1 across this range. This temperature range can be adjusted by altering the values of `T_liq` and/or `T_ice` in atmos/constant.py.

#### 4. Ability to handle different vertical grids

The functions in __atmos__ that perform vertical integration, averaging, or interpolation are designed to be agnostic of the vertical grid; so long as arrays of pressure and/or height are provided, the calculations should work. However, when working with data on pressure levels, it is important to provide surface/screen-level variables (specified using the `_sfc` keyword arguments) in addition to the pressure-level arrays. This will ensure that values below the surface are excluded from the calculations.


## Examples

Below are a few examples of using __atmos__ functions. Note that all variables are assumed to be in SI units; i.e., m for heights, Pa for pressures, K for temperatures, kg/kg for mass fractions (specific humidities) and mixing ratios, m/s for wind velocities. Relative humidities are expressed as fractions rather than percentages. For functions that perform vertical integration, averaging, or interpolation, it is assumed that the first array dimension corresponds to the vertical axis (unless the `vertical_axis` keyword is specified) and that pressure decreases and height increases along this axis. By default, saturation is calculated with respect to liquid water only.

### Basic thermodynamic variables

Calculating saturation vapour pressure with respect to liquid water `esl`, ice `esi`, and mixed-phase condensate `esx` from temperature `T`:
```python
esl = atmos.thermo.saturation_vapour_pressure(T, phase="liquid")
esi = atmos.thermo.saturation_vapour_pressure(T, phase="ice")
omega = atmos.thermo.ice_fraction(T)  # ice fraction assuming saturation at temperature T
esx = atmos.thermo.saturation_vapour_pressure(T, phase="mixed", omega=omega)
```

Calculating "pseudo" (aka "adiabatic") wet-bulb temperature `Twp` and "isobaric" (aka "thermodynamic") wet-bulb temperature `Twi` from pressure `p`, temperature `T`, and specific humidity `q`:
```python
Twp = atmos.thermo.pseudo_wet_bulb_temperature(p, T, q)
Twi = atmos.thermo.isobaric_wet_bulb_temperature(p, T, q)
```

Calculating potential temperature `th`, virtual potential temperature `thv`, and equivalent potential temperature `theq` from pressure `p`, temperature `T`, specific humidity `q`, and total water mass fraction `qt`:
```python
th = atmos.thermo.potential_temperature(p, T, q, qt=qt)
thv = atmos.thermo.virtual_potential_temperature(p, T, q, qt=qt)
theq = atmos.thermo.equivalent_potential_temperature(p, T, q, qt=qt)
```

Converting from relative humidity with respect to ice `RHi` to vapour pressure `e`, mixing ratio `r`, and dewpoint temperature `Td` given pressure `p` and temperature `T`:
```python
# Compute vapour pressure
e = atmos.moisture.vapour_pressure_from_relative_humidity(T, RHi, phase='ice')

# Compute mixing ratio
r = atmos.moisture.mixing_ratio_from_vapour pressure(p, e)

# Compute relative humidity with respect to liquid water
RHl = atmos.moisture.convert_relative_humidity(T, RHi, 'ice', 'liquid')

# Compute dewpoint temperature
Td = atmos.moisture.dewpoint_temperature_from_relative_humidity(T, RHl)
```

### Convective parameters

Calculating surface-based (SB), mixed-layer (ML), and most-unstable (MU) CAPE and CIN and the associated lifting condensation level (LCL), level of free convection (LFC), level of maximum buoyancy (LMB), and equilibrium level (EL) given arrays of pressure `p`, temperature `T` and specific humidity `q`:
```python
SBCAPE, SBCIN, SBLCLp, SBLFCp, SBLMBp, SBELp = atmos.parcel.surface_based_parcel(p, T, q)
MLCAPE, MLCIN, MLLCLp, MLLFCp, MLLMBp, MLELp = atmos.parcel.mixed_layer_parcel(p, T, q)
MUCAPE, MUCIN, MULPLp, MULCLp, MULFCp, MULMBp, MUELp = atmos.parcel.most_unstable_parcel(p, T, q)
```
Note that the MU parcel function also outputs the lifted parcel level (LPL), which is the starting level of the parcel ascent (for the SB and ML parcels, this is the lowest level). The levels output by these function are all pressures. Given a corresponding array of heights `z`, levels can be converted to heights using the _utils_ function `height_of_pressure_level`. For example, the height of the MLLCL can be calculated as:
```python
MLLCLz = atmos.utils.height_of_pressure_level(p, z, MLLCLp)
```

Calculating 0-6 km bulk wind difference (BWD06) given arrays of height `z`, zonal wind `u`, and meridional wind `v`:
```python
# Compute BWD components
BWD06u, BWD06v = atmos.kinematic.bulk_wind_difference(z, u, v, 0.0, 6000.0)

# Compute BWD magnitude
BWD06 = np.hypot(BWD06u, BWD06v)
```

Calculating effective storm-relative helicity (ESRH):
```python
# Use the MU parcel function with the "max_cape" option to get the effective inflow
# layer base (EIB) and top (EIT)
*_, EIBp, EITp = atmos.parcel.most_unstable_parcel(p, T, q, mu_parcel="max_cape")

# Convert from pressure to height
EIBz = atmos.utils.height_of_pressure_level(p, z, EIBp)
EITz = atmos.utils.height_of_pressure_level(p, z, EITp)

# Compute Bunkers left and right storm motion vectors:
u_bl, v_bl, u_br, v_br = atmos.kinematic.bunkers_storm_motion(z, u, v)

# Compute ESRH for Bunkers left storm motion vector:
ESRH_bl = atmos.kinematic.storm_relative_helicity(z, u, v, u_bl, v_bl, EIBz, EITz)

# Compute ESRH for Bunkers right storm motion vector:
ESRH_br = atmos.kinematic.storm_relative_helicity(z, u, v, u_br, v_br, EIBz, EITz)
```


## Developer notes

The development of __atmos__ has been something of a side project for me and, as yet, I have not had time to create unit tests in order to rigorously test the code. As such it is possible that a few bugs are present. If you identify one please raise an issue and I will aim to fix it ASAP. That said, users should find that the code is well commented and easy to read. The choice of variable names and the formatting of equations matches more or less exactly the notation in [Warren (2025)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4866), which should make it easy to understand the origins of each function.

If you have any suggestions for new features you would like to see added to __atmos__, please don't hesitate to get in touch.

Rob Warren (13-01-2026)
