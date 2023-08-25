"""
References:

Wagner, W., and A. Pruß, 2002: The IAPWS Formulation 1995 for the 
    Thermodynamic Properties of Ordinary Water Substance for General and
    Scientific Use. J. Phys. Chem. Ref. Data, 31, 387-535.

Feistel, R., and W. Wagner, 2006: A New Equation of State for H2O Ice Ih.
    J. Phys. Chem. Ref. Data, 35, 2021-1047.

Ambaum, M. H., 2020. Accurate, simple equation for saturated vapour
    pressure over water and ice. Quart. J. Roy. Met. Soc., 146, 4252-4258.

"""


# Acceleration due to gravity (m2/s2)
g = 9.80665      

# Specific gas constant for dry air (J/kg/K)
Rd = 287.04

# Specific gas constant for water vapour (J/kg/K)
Rv = 461.52  # Wagner and Pruß (2002)

# Ratio of gas constants
eps = Rd / Rv

# Isobaric specific heat of dry air (J/kg/K)
cpd = 1005.7

# Isobaric specific heat of water vapour (J/kg/K)
cpv = 1884.4  # Wagner and Pruß (2002)

# Isobaric specific heat of liquid water (J/kg/K)
cpl = 4219.9  # Wagner and Pruß (2002)

# Isobaric specific heat of ice (J/kg/K)
cpi = 2096.8  # Feistel and Wagner (2006)

# Triple point temperature (K)
T0 = 273.16

# Saturation vapour pressure at the triple point (Pa)
es0 = 611.655  # Wagner and Pruß (2002)

# latent heat of vaporisation at the triple point (J/kg)
Lv0 = 2.501e6  # Ambaum (2020)

# latent heat of freezing at the triple point (J/kg)
Lf0 = 3.334e5  # Ambaum (2020)

# Latent heat of sublimation at the triple point (J/kg)
Ls0 = Lv0 + Lf0  
