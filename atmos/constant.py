"""
References:

Ambaum, M. H., 2020. Accurate, simple equation for saturated vapour pressure
    over water and ice. Quart. J. Roy. Met. Soc., 146, 4252-4258.

Feistel, R., and W. Wagner, 2006: A New Equation of State for H2O Ice Ih.
    J. Phys. Chem. Ref. Data, 35, 2021-1047.

Guildner, L. A., D. P. Johnson, and F. E. Jones, 1976: Vapor pressure of water
    at its triple point.J. Res. Natl. Bur. Stand., 80A, 505-521.

Wagner, W., and A. Pruß, 2002: The IAPWS Formulation 1995 for the 
    Thermodynamic Properties of Ordinary Water Substance for General and
    Scientific Use. J. Phys. Chem. Ref. Data, 31, 387-535.

"""

# Acceleration due to gravity (m/s2)
g = 9.81

# Specific gas constant for dry air (J/kg/K)
Rd = 287.0

# Specific gas constant for water vapour (J/kg/K)
Rv = 461.5

# Ratio of gas constants
eps = Rd / Rv

# Isobaric specific heat of dry air (J/kg/K)
cpd = 1005.0

# Isobaric specific heat of water vapour (J/kg/K)
cpv = 2040.0  # Optimised value from Ambaum (2020)

# Isobaric specific heat of liquid water (J/kg/K)
cpl = 4220.0  # triple-point value from Wagner and Pruß (2002)

# Isobaric specific heat of ice (J/kg/K)
cpi = 2097.0  # triple-point value from Feistel and Wagner (2006)

# Triple point temperature (K)
T0 = 273.16

# Saturation vapour pressure at the triple point (Pa)
es0 = 611.657  # Guildner et al. (1976)

# latent heat of vaporisation at the triple point (J/kg)
Lv0 = 2.501e6  # Wagner and Pruß (2002)

# latent heat of freezing at the triple point (J/kg)
Lf0 = 0.333e6  # Feistel and Wagner (2006)

# Latent heat of sublimation at the triple point (J/kg)
Ls0 = Lv0 + Lf0  
