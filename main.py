
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize_scalar


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Chamber conditions
Pc = 5e6                  # Chamber pressure [Pa]
Tc = 3000.0               # Chamber temperature [K]
gamma = 1.25              # Specific heat ratio [-]
R = 350.0                 # Specific gas constant [J/(kg*K)]

# Nozzle geometry
At = 0.01                 # Throat area [m^2]
BASE_EXPANSION_RATIO = 5.0
CHAMBER_RADIUS = 0.12     # [m]
CONVERGING_LENGTH = 0.15 # [m]
DIVERGING_LENGTH = 0.30  # [m]
N_POINTS = 100

# Ambient conditions
Pa = 101325.0             # Sea-level pressure [Pa]
g0 = 9.80665              # Standard gravity [m/s^2]

# Plot settings
SHOW_GEOMETRY = True
SHOW_MACH = True
SHOW_TEMPERATURE = False
SHOW_VELOCITY = False
SHOW_THRUST_COMPARISON = True
SHOW_OPTIMIZATION = True
SHOW_ALTITUDE_COMPARISON = True

ATMOSPHERIC_SCALE_HEIGHT = 8500.0  # [m]


# ============================================================
# 2. GAS DYNAMICS
# ============================================================

def choked_mass_flow():
    """Calculate ideal choked mass flow at the throat."""
    return (
        At * Pc
        * np.sqrt(gamma / (R * Tc))
        * (2 / (gamma + 1))
        ** ((gamma + 1) / (2 * (gamma - 1)))
    )


def area_mach(M, gamma):
    """Isentropic area-Mach relation A/A*."""
    term = (
        2 / (gamma + 1)
        * (1 + (gamma - 1) / 2 * M**2)
    )
    exponent = (gamma + 1) / (2 * (gamma - 1))

    return term**exponent / M


def mach_residual(M, area_ratio, gamma):
    return area_mach(M, gamma) - area_ratio


def solve_mach(area_ratio, gamma, supersonic=False):
    """Solve the subsonic or supersonic area-Mach branch."""
    if np.isclose(area_ratio, 1.0):
        return 1.0

    if supersonic:
        lower, upper = 1.000001, 10.0
    else:
        lower, upper = 0.000001, 0.999999

    return brentq(
        mach_residual,
        lower,
        upper,
        args=(area_ratio, gamma)
    )


def flow_properties(M):
    """Calculate static flow properties from Mach number."""
    factor = 1 + (gamma - 1) / 2 * M**2

    T = Tc / factor
    P = Pc / factor**(gamma / (gamma - 1))
    rho = P / (R * T)
    a = np.sqrt(gamma * R * T)
    u = M * a

    return T, P, rho, a, u


# ============================================================
# 3. NOZZLE GEOMETRY AND FLOW DISTRIBUTION
# ============================================================

def nozzle_geometry(expansion_ratio):
    """Generate a piecewise-linear converging-diverging nozzle."""
    Ae = expansion_ratio * At

    rc = CHAMBER_RADIUS
    rt = np.sqrt(At / np.pi)
    re = np.sqrt(Ae / np.pi)

    Lc = CONVERGING_LENGTH
    Ld = DIVERGING_LENGTH

    x_conv = np.linspace(-Lc, 0, N_POINTS)
    x_div = np.linspace(0, Ld, N_POINTS)

    r_conv = rt + (rc - rt) * (-x_conv / Lc)
    r_div = rt + (re - rt) * (x_div / Ld)

    x = np.concatenate((x_conv, x_div[1:]))
    r = np.concatenate((r_conv, r_div[1:]))
    A = np.pi * r**2

    return x, r, A


def nozzle_flow(expansion_ratio):
    """Calculate the ideal Mach and flow-property distributions."""
    x, r, A = nozzle_geometry(expansion_ratio)

    Mach = np.zeros_like(x)

    for i, position in enumerate(x):
        area_ratio = A[i] / At

        if position < 0:
            Mach[i] = solve_mach(area_ratio, gamma)
        elif position > 0:
            Mach[i] = solve_mach(
                area_ratio, gamma, supersonic=True
            )
        else:
            Mach[i] = 1.0

    T, P, rho, a, u = flow_properties(Mach)
    mdot_check = rho * u * A

    return {
        "x": x,
        "r": r,
        "A": A,
        "Mach": Mach,
        "T": T,
        "P": P,
        "rho": rho,
        "a": a,
        "u": u,
        "mdot_check": mdot_check
    }


# ============================================================
# 4. NOZZLE PERFORMANCE
# ============================================================

def exit_conditions(expansion_ratio):
    """Calculate ideal supersonic exit conditions."""
    Me = solve_mach(
        expansion_ratio, gamma, supersonic=True
    )

    Te, Pe, rhoe, ae, ue = flow_properties(Me)

    return Me, Te, Pe, ue


def nozzle_performance(expansion_ratio, ambient_pressure):
    """Calculate thrust and specific impulse."""
    _, _, Pe, ue = exit_conditions(expansion_ratio)

    mdot = choked_mass_flow()
    Ae = expansion_ratio * At

    F_momentum = mdot * ue
    F_pressure = (Pe - ambient_pressure) * Ae
    F_total = F_momentum + F_pressure

    Isp = F_total / (mdot * g0)

    return F_total, Isp


def performance_coefficients(expansion_ratio):
    """Calculate sea-level/vacuum performance coefficients."""
    mdot = choked_mass_flow()

    F_sea, Isp_sea = nozzle_performance(
        expansion_ratio, Pa
    )
    F_vac, Isp_vac = nozzle_performance(
        expansion_ratio, 0.0
    )

    Cf_sea = F_sea / (Pc * At)
    Cf_vac = F_vac / (Pc * At)
    c_star = Pc * At / mdot

    return {
        "F_sea": F_sea,
        "F_vac": F_vac,
        "Isp_sea": Isp_sea,
        "Isp_vac": Isp_vac,
        "Cf_sea": Cf_sea,
        "Cf_vac": Cf_vac,
        "c_star": c_star
    }


# ============================================================
# 5. EXPANSION RATIO OPTIMIZATION
# ============================================================

def optimize_expansion_ratio(ambient_pressure):
    """Find the expansion ratio maximizing ideal thrust."""

    def objective(eps):
        thrust, _ = nozzle_performance(
            eps, ambient_pressure
        )
        return -thrust

    result = minimize_scalar(
        objective,
        bounds=(1.01, 40),
        method="bounded"
    )

    if not result.success:
        raise RuntimeError("Nozzle optimization failed.")

    optimal_ratio = result.x
    maximum_thrust = -result.fun

    return optimal_ratio, maximum_thrust


# ============================================================
# 6. ALTITUDE MODEL
# ============================================================

def ambient_pressure(h):
    """Calculate atmospheric pressure at altitude h [m]."""
    return Pa * np.exp(-np.asarray(h) / ATMOSPHERIC_SCALE_HEIGHT)


def thrust_at_altitude(expansion_ratio, h):
    Pa_h = ambient_pressure(h)

    F, Isp = nozzle_performance(expansion_ratio, Pa_h)

    return F, Isp


# ============================================================
# 7. PHYSICS VALIDATION
# ============================================================

def validate_simulator():
    """Check five internal consistency conditions for the ideal model."""
    flow = nozzle_flow(BASE_EXPANSION_RATIO)
    mdot = choked_mass_flow()

    # 1. Continuity through the nozzle
    relative_error = np.max(np.abs(flow["mdot_check"] - mdot)) / mdot
    assert relative_error < 1e-6, "Mass conservation failed"

    # 2. Choked throat
    throat_idx = np.argmin(flow["A"])
    assert np.isclose(flow["Mach"][throat_idx], 1.0), "Throat Mach check failed"

    # 3. Vacuum thrust exceeds sea-level thrust
    F_sea, _ = nozzle_performance(BASE_EXPANSION_RATIO, Pa)
    F_vac, _ = nozzle_performance(BASE_EXPANSION_RATIO, 0.0)
    assert F_vac > F_sea, "Vacuum thrust check failed"

    # 4. Optimal exit pressure approximately equals ambient pressure
    optimal_ratio, _ = optimize_expansion_ratio(Pa)
    _, _, Pe_opt, _ = exit_conditions(optimal_ratio)
    assert abs(Pe_opt - Pa) / Pa < 1e-4, "Optimal pressure check failed"

    # 5. Specific impulse coefficient identity
    perf = performance_coefficients(BASE_EXPANSION_RATIO)
    isp_from_coefficients = perf["Cf_sea"] * perf["c_star"] / g0
    assert np.isclose(isp_from_coefficients, perf["Isp_sea"]), "Isp check failed"

    print("All five physics validation checks passed!")


# ============================================================
# 8. PLOTTING
# ============================================================

def plot_nozzle_geometry(flow):
    x, r = flow["x"], flow["r"]

    plt.figure(figsize=(10, 4))
    plt.plot(x, r, label="Upper wall")
    plt.plot(x, -r, label="Lower wall")
    plt.axvline(0, color="gray", linestyle="--")

    plt.xlabel("Axial position (m)")
    plt.ylabel("Radius (m)")
    plt.title("Converging-Diverging Nozzle")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/nozzle_geometry.png", dpi=300)


def plot_flow_distribution(flow, variable, ylabel, title):
    plt.figure(figsize=(10, 5))
    plt.plot(flow["x"], flow[variable], linewidth=2)
    plt.axvline(0, color="gray", linestyle="--")

    plt.xlabel("Axial position (m)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    if variable == "Mach":
        plt.tight_layout()
        plt.savefig("figures/mach_distribution.png", dpi=300)


def plot_thrust_comparison():
    ratios = np.linspace(1.1, 40, 150)

    thrust_sea = []
    thrust_vac = []

    for eps in ratios:
        F_sea, _ = nozzle_performance(eps, Pa)
        F_vac, _ = nozzle_performance(eps, 0.0)

        thrust_sea.append(F_sea)
        thrust_vac.append(F_vac)

    plt.figure(figsize=(10, 5))
    plt.plot(
        ratios, np.array(thrust_sea) / 1000,
        label="Sea level"
    )
    plt.plot(
        ratios, np.array(thrust_vac) / 1000,
        label="Vacuum"
    )

    plt.xlabel("Expansion ratio (Ae/At)")
    plt.ylabel("Thrust (kN)")
    plt.title("Thrust vs. Nozzle Expansion Ratio")
    plt.grid(True)
    plt.legend()


def plot_optimization(optimal_ratio, maximum_thrust):
    ratios = np.linspace(1.1, 40, 300)

    thrust = np.array([
        nozzle_performance(eps, Pa)[0]
        for eps in ratios
    ])

    plt.figure(figsize=(10, 5))
    plt.plot(ratios, thrust / 1000, label="Sea-level thrust")
    plt.scatter(
        optimal_ratio,
        maximum_thrust / 1000,
        color="red",
        zorder=5,
        label="Optimal design"
    )

    plt.xlabel("Expansion ratio (Ae/At)")
    plt.ylabel("Thrust (kN)")
    plt.title("Sea-Level Expansion Ratio Optimization")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/optimization.png", dpi=300)


def plot_altitude_comparison(ratio_a, ratio_b):
    """Compare thrust for two fixed nozzles from 0 to 100 km."""
    altitudes = np.linspace(0, 100000, 200)
    thrust_a = np.array([thrust_at_altitude(ratio_a, h)[0] for h in altitudes])
    thrust_b = np.array([thrust_at_altitude(ratio_b, h)[0] for h in altitudes])

    plt.figure(figsize=(10, 5))
    plt.plot(altitudes / 1000, thrust_a / 1000, label=f"Nozzle A (ε={ratio_a:.3f})")
    plt.plot(altitudes / 1000, thrust_b / 1000, label=f"Nozzle B (ε={ratio_b:.3f})")
    plt.xlabel("Altitude (km)")
    plt.ylabel("Thrust (kN)")
    plt.title("Rocket Nozzle Thrust vs Altitude")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("figures/altitude_comparison.png", dpi=300)

    return thrust_b[0] - thrust_a[0], thrust_b[-1] - thrust_a[-1]


# ============================================================
# 9. REPORTING
# ============================================================

def print_flow_report(flow):
    throat_idx = np.argmin(flow["A"])
    mdot = choked_mass_flow()

    print("\n========== FLOW PROPERTIES ==========")

    for name, key, unit, scale in [
        ("Temperature", "T", "K", 1),
        ("Pressure", "P", "MPa", 1e6),
        ("Density", "rho", "kg/m^3", 1),
        ("Velocity", "u", "m/s", 1)
    ]:
        values = flow[key] / scale

        print(f"\n{name} [{unit}]")
        print(f"  Inlet:  {values[0]:.3f}")
        print(f"  Throat: {values[throat_idx]:.3f}")
        print(f"  Exit:   {values[-1]:.3f}")

    mdot_check = flow["mdot_check"]
    error = np.max(np.abs(mdot_check - mdot)) / mdot

    print(f"\nChoked mass flow: {mdot:.4f} kg/s")
    print(f"Mass conservation error: {error:.2e}")


def print_performance_report(expansion_ratio):
    perf = performance_coefficients(expansion_ratio)
    _, _, Pe, ue = exit_conditions(expansion_ratio)

    mdot = choked_mass_flow()
    Ae = expansion_ratio * At

    F_momentum = mdot * ue
    F_pressure = (Pe - Pa) * Ae

    print("\n========== NOZZLE PERFORMANCE ==========")
    print(f"Expansion ratio: {expansion_ratio:.3f}")
    print(f"Momentum thrust: {F_momentum / 1000:.2f} kN")
    print(f"Pressure thrust: {F_pressure / 1000:.2f} kN")
    print(f"Sea-level thrust: {perf['F_sea'] / 1000:.2f} kN")
    print(f"Vacuum thrust: {perf['F_vac'] / 1000:.2f} kN")
    print(f"Sea-level Isp: {perf['Isp_sea']:.2f} s")
    print(f"Vacuum Isp: {perf['Isp_vac']:.2f} s")
    print(f"Sea-level Cf: {perf['Cf_sea']:.4f}")
    print(f"Vacuum Cf: {perf['Cf_vac']:.4f}")
    print(f"Characteristic velocity: {perf['c_star']:.2f} m/s")

    Isp_check = perf["Cf_sea"] * perf["c_star"] / g0
    print(f"Isp consistency check: {Isp_check:.4f} s")


def print_optimization_report(optimal_ratio, maximum_thrust):
    Me, Te, Pe, ue = exit_conditions(optimal_ratio)

    print("\n========== OPTIMIZATION ==========")
    print(f"Optimal expansion ratio: {optimal_ratio:.6f}")
    print(f"Maximum thrust: {maximum_thrust / 1000:.6f} kN")
    print(f"Exit pressure: {Pe / 1000:.6f} kPa")
    print(f"Ambient pressure: {Pa / 1000:.6f} kPa")


def print_altitude_report(ratio_a, ratio_b):
    print("\n========== ALTITUDE PERFORMANCE ==========")
    for h in [0, 10000, 20000, 50000, 100000]:
        pressure = ambient_pressure(h)
        thrust, isp = thrust_at_altitude(ratio_a, h)
        print(
            f"Altitude: {h / 1000:6.1f} km | "
            f"Pressure: {pressure / 1000:8.3f} kPa | "
            f"Thrust A: {thrust / 1000:7.3f} kN | Isp A: {isp:.2f} s"
        )

    sea_a, _ = thrust_at_altitude(ratio_a, 0)
    sea_b, _ = thrust_at_altitude(ratio_b, 0)
    high_a, _ = thrust_at_altitude(ratio_a, 100000)
    high_b, _ = thrust_at_altitude(ratio_b, 100000)
    print(f"Thrust advantage at sea level: {(sea_b - sea_a) / 1000:.3f} kN")
    print(f"Thrust advantage at 100 km: {(high_b - high_a) / 1000:.3f} kN")


# ============================================================
# 10. MAIN PROGRAM
# ============================================================

def main():
    validate_simulator()

    # Baseline and optimized nozzle reports
    flow = nozzle_flow(BASE_EXPANSION_RATIO)
    optimal_ratio, maximum_thrust = optimize_expansion_ratio(Pa)

    print_flow_report(flow)
    print_performance_report(BASE_EXPANSION_RATIO)
    print_optimization_report(optimal_ratio, maximum_thrust)
    print_altitude_report(BASE_EXPANSION_RATIO, optimal_ratio)

    # Optional plots
    if SHOW_GEOMETRY:
        plot_nozzle_geometry(flow)
    if SHOW_MACH:
        plot_flow_distribution(flow, "Mach", "Mach number", "Mach Distribution")
    if SHOW_TEMPERATURE:
        plot_flow_distribution(flow, "T", "Temperature (K)", "Temperature Distribution")
    if SHOW_VELOCITY:
        plot_flow_distribution(flow, "u", "Velocity (m/s)", "Velocity Distribution")
    if SHOW_THRUST_COMPARISON:
        plot_thrust_comparison()
    if SHOW_OPTIMIZATION:
        plot_optimization(optimal_ratio, maximum_thrust)
    if SHOW_ALTITUDE_COMPARISON:
        plot_altitude_comparison(BASE_EXPANSION_RATIO, optimal_ratio)

    if plt.get_fignums():
        plt.show()


if __name__ == "__main__":
    main()
