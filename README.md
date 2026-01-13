# atmos: fast and accurate calculations for applications in atmospheric science


## What is atmos?

__atmos__ is a Python library for computing physical quantities commonly used in atmospheric science. The code comprises the following modules:
* thermo - functions for calculating various thermodynamic quantities (density, wet-bulb temperature, potential temperature, etc.)
* moisture - functions for convecting between different measures of atmospheric moisture (specific humidity, relative humidity, dewpoint temperature, etc.)
* parcel - functions for performing adiabatic and pseudoadiabatic parcel ascents to calculate convectiva available potential energy (CAPE), convective inhibition (CIN), and related quantities
* kinematic - functions for converting between wind speed/direction and u/v and for calculating kinematic quantities (bulk wind difference, storm-relative helicity, etc.)
* pseudoadiabat - functions for performing fast pseudoadiabatic calculations (see below)
* utils - generic functions for peforming interpolation and layer averaging of scalar and vector quantities

The functions in __atmos__ are designed to work with numpy arrays, though many of them can also be applied to scalar variables. This makes __atmos__ well suited for processing gridded numerical weather prediction and climate model output, as well as observational data.


## Advantages of atmos

### 1. Accuracy and physical consistency

Unlike other scientific libraries for analysing meteorological data, __atmos__ was designed with physical consistency as a foremost consideration. All of the thermodynamic equations in __atmos__ are analytical and derived from a common set of assumptions, known as the _Rankine-Kirchhoff_ approximations [(Romps, 2021)](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.4154). The core assumptions are: (i) ideal gases, (ii), constant specific heat capacities, and (iii) zero volume of condensates (liquid and ice). These approximations allow for the derivation of highly accurate analytical equations for moist thermodynamics.

### 2. Speed of calculations

All of the functions in __atmos__ are vectorised to allow for fast processing of multidimensional arrays. To speed up pseudoadiabatic parcel calculations, __atmos__ uses high-order polynomial fits to (i) parcel temperature as a function of wet-bulb potential temperature (WBPT) and pressure and (ii) WBPT as a function of parcel temperature and pressure following [Moisseeva and Stull (2017)](https://acp.copernicus.org/articles/17/15037/2017/). This is also the basis for the Noniterative Evaluation of Wet-bulb Temperature (NEWT) method ([Rogers and Warren, 2023](https://doi.org/10.22541/essoar.170560423.39769387/v1)).

### 3. Representation of saturation

Another novel feature in __atmos__ is the treatment of saturation, which can be represented with respect to liquid, ice, or a combination of the two (via the `phase` argument) following [Warren (2025)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4866).


## Examples

Below are a few examples of using __atmos__ functions. Note that all variables are assumed to be in SI units (i.e., m for heights, Pa for pressures, K for temperatures, kg/kg for mass fractions (specific humidities) and mixing ratios, fraction for relative humidities, m/s for wind velocities). For the `parcel` and `utils` functions, it is assumed by default that the first array dimension corresponds to the vertical axis and that pressure decreases and height increases along this axis.

Calculating saturation vapour pressure with respect to liquid water `esl`, ice `esi`, and mixed-phase condensate `esx` from temperature `T`:
```python
esl = atmos.thermo.saturation_vapour_pressure(T, phase="liquid")
esi = atmos.thermo.saturation_vapour_pressure(T, phase="ice")
omega = atmos.thermo.ice_fraction(T)  # ice fraction assuming saturation at temperature T
esx = atmos.thermo.saturation_vapour_pressure(T, phase="mixed", omega=omega)
```

Calculating pseudo wet-bulb temperature `Twp` and isobaric/thermodynamic wet-bulb temperature `Twi` from pressure `p`, temperature `T`, and specific humidity `q`:
```python
Twp = atmos.thermo.pseudo_wet_bulb_temperature(p, T, q)
Twi = atmos.thermo.isobaric_wet_bulb_temperature(p, T, q)
```

Calculating air density `rho`, vapour pressure `e`, and virtual (density) temperature `Tv` from pressure `p`, temperature `T`, specific humidity `q`, and total water mass fraction `qt`:
```python
rho = atmos.thermo.air_density(p, T, q, qt=qt)
e = atmos.thermo.vapour_pressure(q, qt=qt)
Tv = atmos.thermo.virtual_temperature(T, q, qt=qt)
```

Converting from relative humidity with respect to ice `RHi` to dewpoint temperature `Td` given pressure `p` and temperature `T`:
```python
# Compute relative humidity with respect to liquid water
RHl = atmos.moisture.convert_relative_humidity(T, RHi, 'ice', 'liquid')

# Compute dewpoint temperature
Td = atmos.moisture.dewpoint_temperature_from_relative_humidity(T, RHl)
```

Calculating mixed-layer CAPE and CIN and the associated lifting condensation level (LCL), level of free convection (LFC), level of maximum buoyancy (LMB), and equilibrium level (EL) given 1D profiles or arrays of pressure `p`, temperature `T` and specific humidity `q`:
```python
CAPE, CIN, LCLp, LFCp, LMBp, ELp = atmos.parcel.mixed_layer_parcel(p, T, q)
```
The levels output by this function are pressures. Given a corresponding profile/array of height `z`, these can be converted to heights using the `utils.height_of_pressure_level` function. For example, the height of the LCL can be calculated as:
```python
LCLz = atmos.utils.height_of_pressure_level(p, z, LCLp)
```

Calculating 0-6 km bulk wind difference (BWD):
```python
# Compute BWD components
BWD06u, BWD06v = atmos.kinematic.bulk_wind_difference(z, u, v, 0.0, 6000.0)

# Compute BWD magnitude
BWD06 = np.hypot(BWD06u, BWD06v)
```

## Developer notes

The development of __atmos__ has been something of a side project for me and, as yet, I have not had time to create unit tests in order to rigorously test the code. As such it is possible that a few bugs are present. If you identify one please raise an issue and I will aim to fix it ASAP. That said, users should find that the code is well commented and easy to read. The choice of variable names and the formatting of equations matches more or less exactly the notation in [Warren (2025)](https://rmets.onlinelibrary.wiley.com/doi/10.1002/qj.4866), which should make it easy to understand the origins of each function.

If you have any suggestions for new features you would like to see added to __atmos__, please don't hesitate to get in touch.

Rob Warren (13-01-2026)
