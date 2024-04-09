# atmos: fast and accurate calculations for applications in atmospheric science

__atmos__ is a Python library for computing physical quantities commonly used in atmospheric science. Currently, the code comprises vectorised functions for calculating a wide array of thermodynamic variables (the "thermo" module), for translating between different measures of atmospheric moisture (the "moisture" module), and for performing adiabatic/pseudoadiabatic parcel ascents to calculate CAPE and CIN (the "parcel" module). All of the equations in __atmos__ are analytical and derived from a common set of assumptions (the so-called _Rankine-Kirchhoff_ approximations; [Romps, 2021](https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.4154)); no empirical equations are used. Another novel feature of __atmos__ is the treatment of saturation, which can be represented with respect to liquid, ice, or a combination of the two (via the `phase` argument). A new treatment of mixed-phase saturation has been developed, which will be documented in a forthcoming paper (currently in review).

To speed up pseudoadiabatic parcel calculations, __atmos__ uses high-order polynomial fits to (i) parcel temperature as a function of wet-bulb potential temperature (WBPT) and pressure and (ii) WBPT as a function of parcel temperature and pressure following [Moisseeva and Stull (2017)](https://acp.copernicus.org/articles/17/15037/2017/). This is also the basis for the Noniterative Evaluation of Wet-bulb Temperature (NEWT) method, which was advertised by my colleague Cass Rogers at the 2023 AGU Annual Meeting ([Rogers and Warren, 2023](https://doi.org/10.22541/essoar.170560423.39769387/v1)). The NEWT method and its performance relative to other approximate methods for calculating wet-bulb temperature will be presented in another forthcoming paper. Code to perform the polynomial fits and for generating the "pseudoadiabat" module, which applies them, can be found in the "pseudoadiabat_codegen" directory. Note that, at present, the polynomial approach is only available for liquid pseudoadiabats (with the default options `phase="liquid"` and `pseudo_method="polynomial"`) as it seems to introduce unacceptably large errors for ice and mixed-phase pseudoadiabats. In the future, I plan to explore the use of [Symbolic Regression](https://github.com/MilesCranmer/PySR) as an alternative to the high-order polynomial fits. However, for now, ice and mixed-phase pseudoadiabatic parcel calculations must be performed iteratively (by setting `pseudo_method="iterative"`). Calculations involving saturated adiabats are performed iteratively by default due to the additional dependence on total water mass fraction.

The development of __atmos__ has been something of a side project for me and, as yet, I have not had time to create unit tests in order to rigorously test the code. As such it is possible (likely even) that some bugs are present. If you identify one please raise an issue and I will aim to fix it ASAP. That said, users should find that the code is well commented and easy to read. The choice of variable names and the formatting of equations matches more or less exactly that in my forthcoming paper on mixed-phase saturation. This should make it easy to understand the origins of each function once the paper is published.

In the near future, I plan to add a handful of demonstration notebooks to the repository, which will illustrate some of the functionality of __atmos__. An additional module to calculate kinematic parameters, such as bulk wind difference and storm-relative helicity, may also be added further down the line. If you have any suggestions for new features you would like to see added, please don't hesitate to get in touch.

Rob Warren (09-04-2024)
