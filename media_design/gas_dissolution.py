import numpy as np
import pandas as pd
#importing pychnosz environment to get around documentation error
try:
    from pychnosz import thermo, subcrt
    _ = thermo("WORM")
except ImportError:
    thermo = None
    subcrt = None

MM = {"N2": 28.0134, "O2": 31.998, "Ar": 39.948, "CO2": 44.0095, "CH4": 16.043}

def subcrt_out(res):
    """
    Extract the reaction output table from a pyCHNOSZ `subcrt` result.

    Parameters
    ----------
    res : object
        Result returned by `pychnosz.subcrt`. Supported return types are:
        - an object with an `.out` attribute
        - a dictionary containing an `"out"` key

    Returns
    -------
    pandas.DataFrame
        Output table containing thermodynamic results from `subcrt`.

    Raises
    ------
    TypeError
        If `res` is not a supported `subcrt` return type.

    Notes
    -----
    This helper provides compatibility across different pyCHNOSZ return
    formats.
    """
    if hasattr(res, "out"):
        return res.out
    if isinstance(res, dict) and "out" in res:
        return res["out"]
    raise TypeError(f"Unexpected subcrt return type: {type(res)}")


def vapor_pressure_H2O_bar(T_C: float) -> float:
    """
    Calculate water vapor pressure in bar using the Antoine equation.

    Parameters
    ----------
    T_C : float
        Temperature in degrees Celsius.

    Returns
    -------
    float
        Water vapor pressure in bar.

    Notes
    -----
    - Uses Antoine equation constants for water:
      A = 8.07131, B = 1730.63, C = 233.426
    - Intermediate pressure is calculated in mmHg and then converted to bar.
    - This parameterization is appropriate approximately over 1-100 °C.
    """
    A, B, C = 8.07131, 1730.63, 233.426  # water; P in mmHg
    P_mmHg = 10 ** (A - B / (C + T_C))
    return P_mmHg / 750.062  # mmHg -> bar


def logK_gas_to_aq(gas: str, T_C: float, P_bar: float) -> float:
    """
    Calculate the equilibrium constant for dissolution of a gas into aqueous form.

    Parameters
    ----------
    gas : str
        Gas species name recognized by the WORM thermodynamic database.
    T_C : float
        Temperature in degrees Celsius.
    P_bar : float
        Pressure in bar.

    Returns
    -------
    float
        Equilibrium constant expressed as logK for the reaction:

        gas(g) = gas(aq)

    Notes
    -----
    The calculation is performed using `pychnosz.subcrt` with the WORM
    thermodynamic database loaded at module import.
    """
   
    if subcrt is None:
        raise ImportError("pychnosz is required for logK_gas_to_aq().")
        
    res = subcrt(
        species=[gas, gas],
        coeff=[-1, 1],
        state=["gas", "aq"],
        T=T_C,
        P=P_bar,
    )
    out = subcrt_out(res)
    return float(out["logK"].iloc[0])


def dissolved_gases_from_headspace(
    headspace_x: dict[str, float],
    *,
    T_C: float,
    P_total_bar: float = 1.0,
    assume_wet_headspace: bool = True,
    molar_masses_g_per_mol: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Convert gas-phase mole fractions in a headspace to dissolved gas concentrations.

    Parameters
    ----------
    headspace_x : dict of {str : float}
        Dictionary mapping gas names to headspace mole fractions.
    T_C : float
        Temperature in degrees Celsius.
    P_total_bar : float, optional
        Total headspace pressure in bar. Default is 1.0.
    assume_wet_headspace : bool, optional
        If True, subtract water vapor pressure from total pressure to obtain
        dry gas pressure. If False, water vapor pressure is ignored. Default
        is True.
    molar_masses_g_per_mol : dict of {str : float} or None, optional
        Dictionary of molar masses for conversion from molar concentration to
        mg/L. If None, `mg_per_L` is reported as NaN. Default is None.

    Returns
    -------
    pandas.DataFrame
        DataFrame sorted by gas name containing the following columns:

        - gas : str
            Gas species name.
        - x : float
            Input mole fraction.
        - P_total_bar : float
            Total headspace pressure.
        - P_H2O_bar : float
            Water vapor pressure at the specified temperature.
        - P_dry_bar : float
            Dry gas pressure after subtracting water vapor pressure.
        - f_bar : float
            Gas fugacity, approximated as mole fraction × dry pressure.
        - logK : float
            Equilibrium constant for gas dissolution.
        - log_a : float
            Log activity of dissolved gas.
        - a : float
            Dissolved gas activity.
        - M_molar : float
            Dissolved concentration assuming M ≈ m ≈ a.
        - mmol_per_L : float
            Dissolved concentration in mmol/L.
        - mg_per_L : float
            Dissolved concentration in mg/L when molar mass is available.

    Raises
    ------
    ValueError
        If any mole fraction is negative or if the total mole fraction sum
        is zero or less.

    Assumptions
    -----------
    - Wet headspace by default:
      P_dry = P_total - P_H2O(T)
    - Ideal fugacity approximation:
      f_i ≈ x_i × P_dry
    - Dilute solution:
      a ≈ m
    - Simplified conversion:
      M ≈ m, so mol/L ≈ mol/kg

    Notes
    -----
    For gases with fugacity less than or equal to zero, concentration-related
    outputs are returned as NaN and `log_a` is returned as `-inf`.
    """
    # validate
    for g, x in headspace_x.items():
        if x < 0:
            raise ValueError(f"Negative mole fraction for {g}: {x}")
    if sum(headspace_x.values()) <= 0:
        raise ValueError("Headspace composition sums to 0.")

    P_H2O = vapor_pressure_H2O_bar(T_C) if assume_wet_headspace else 0.0
    P_dry = max(P_total_bar - P_H2O, 0.0)

    rows = []
    for gas, x in headspace_x.items():
        f = float(x) * P_dry  # bar
        if f <= 0:
            rows.append(
                dict(
                    gas=gas, x=float(x),
                    P_total_bar=float(P_total_bar),
                    P_H2O_bar=float(P_H2O),
                    P_dry_bar=float(P_dry),
                    f_bar=float(f),
                    logK=np.nan,
                    log_a=-np.inf,
                    a=np.nan,
                    M_molar=np.nan,
                    mmol_per_L=np.nan,
                    mg_per_L=np.nan,
                )
            )
            continue

        logK = logK_gas_to_aq(gas, T_C, P_total_bar)
        log_a = logK + np.log10(f)
        a = 10 ** log_a

        # simplified: M ≈ m ≈ a
        M = a
        mmol_L = M * 1e3

        mg_L = np.nan
        if molar_masses_g_per_mol is not None and gas in molar_masses_g_per_mol:
            mg_L = M * molar_masses_g_per_mol[gas] * 1e3  # g/L -> mg/L

        rows.append(
            dict(
                gas=gas, x=float(x),
                P_total_bar=float(P_total_bar),
                P_H2O_bar=float(P_H2O),
                P_dry_bar=float(P_dry),
                f_bar=float(f),
                logK=float(logK),
                log_a=float(log_a),
                a=float(a),
                M_molar=float(M),
                mmol_per_L=float(mmol_L),
                mg_per_L=float(mg_L) if np.isfinite(mg_L) else np.nan,
            )
        )

    return pd.DataFrame(rows).sort_values("gas").reset_index(drop=True)


def headspace_from_dissolved_target(
    targets: dict[str, float],
    *,
    units: str,
    T_C: float,
    P_total_bar: float = 1.0,
    assume_wet_headspace: bool = True,
    molar_masses_g_per_mol: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Calculate required headspace fugacity and mole fraction for target dissolved gases.

    Parameters
    ----------
    targets : dict of {str : float}
        Dictionary mapping gas names to target dissolved values.
    units : {"molality", "ppm", "fugacity"}
        Units of the input target values.
        - "molality": target values are treated as mol/kg
        - "ppm": target values are treated as mg/L-equivalent and converted
          to molality using supplied molar masses
        - "fugacity": target values are treated directly as fugacity in bar
    T_C : float
        Temperature in degrees Celsius.
    P_total_bar : float, optional
        Total headspace pressure in bar. Default is 1.0.
    assume_wet_headspace : bool, optional
        If True, subtract water vapor pressure from total pressure to obtain
        dry gas pressure. If False, water vapor pressure is ignored. Default
        is True.
    molar_masses_g_per_mol : dict of {str : float} or None, optional
        Dictionary of molar masses used when `units="ppm"`. If None, the
        module-level `MM` dictionary is used. Default is None.

    Returns
    -------
    pandas.DataFrame
        DataFrame sorted by gas name containing the following columns:

        - gas : str
            Gas species name.
        - target_input : float
            Original target value supplied by the user.
        - molality : float
            Target converted to molality when applicable.
        - required_f_bar : float
            Required fugacity in bar.
        - required_x : float
            Required headspace mole fraction based on dry gas pressure.

    Assumptions
    -----------
    - Wet headspace by default:
      P_dry = P_total - P_H2O(T)
    - Ideal fugacity approximation:
      f_i ≈ x_i × P_dry
    - Dilute solution:
      a ≈ m

    Notes
    -----
    - For `units="fugacity"`, the input is used directly and no equilibrium
      conversion is performed.
    - For `units="molality"` and `units="ppm"`, the required fugacity is
      calculated from:
      log_f = log_a - logK
    - Scientific notation inputs are supported as long as they are valid
      numeric values.
    """
    if units not in {"molality", "ppm", "fugacity"}:
        raise ValueError("units must be 'molality', 'ppm', or 'fugacity'")

    if molar_masses_g_per_mol is None:
        molar_masses_g_per_mol = MM

    P_H2O = vapor_pressure_H2O_bar(T_C) if assume_wet_headspace else 0.0
    P_dry = max(P_total_bar - P_H2O, 0.0)

    rows = []

    for gas, val in targets.items():

        # Convert input → fugacity

        if units == "fugacity":

            f = float(val)
            m = np.nan
            logK = np.nan

        else:

            if units == "molality":
                m = float(val)

            else:  # ppm
                if gas not in molar_masses_g_per_mol:
                    raise KeyError(f"Molar mass not defined for {gas}")
                M_g = molar_masses_g_per_mol[gas]
                m = float(val) / (M_g * 1e3)

            logK = logK_gas_to_aq(gas, T_C, P_total_bar)

            log_a = np.log10(m)
            log_f = log_a - logK
            f = 10 ** log_f

        # Convert fugacity → mole fraction

        x = f / P_dry if P_dry > 0 else np.nan

        rows.append(
            dict(
                gas=gas,
                target_input=float(val),
                molality=float(m) if np.isfinite(m) else np.nan,
                required_f_bar=float(f),
                required_x=float(x),
            )
        )

    return pd.DataFrame(rows).sort_values("gas").reset_index(drop=True)