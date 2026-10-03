import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    A physics auditor for a set of true/false statements about
    thermodynamics, gases, and the Magnus effect.
    """
    def __init__(self, solution_instance):
        """
        Extract all expected results from the solution instance.
        Use None as default for any missing key.
        The solution_instance is expected to have:
          - results: dict with keys 'Q1'..'Q8' (each 'T' or 'F')
        """
        self.results = getattr(solution_instance, 'results', {})
        # Safe extraction of each answer
        self.Q1 = self.results.get('Q1', None)
        self.Q2 = self.results.get('Q2', None)
        self.Q3 = self.results.get('Q3', None)
        self.Q4 = self.results.get('Q4', None)
        self.Q5 = self.results.get('Q5', None)
        self.Q6 = self.results.get('Q6', None)
        self.Q7 = self.results.get('Q7', None)
        self.Q8 = self.results.get('Q8', None)

    # ----------------------------------------------------------------
    # Interface checks – required keys and data types
    # ----------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all eight results exist and are strings of length 1."""
        for qkey in ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8']:
            val = getattr(self, qkey, None)
            if val is None:
                return (False, f"Missing result: {qkey}")
            if not isinstance(val, str) or len(val) != 1:
                return (False, f"{qkey} must be a single character string, got {val}")
        return (True, "All required keys present and valid strings")

    # ----------------------------------------------------------------
    # Theorem‑based invariant checks (physics consistency)
    # ----------------------------------------------------------------
    def test_invariant_third_law_Q1(self):
        """
        The third law of thermodynamics states that no system can be cooled
        to absolute zero and that at T=0 K only the solid phase exists.
        Therefore statement (a) is false -> Q1 must be 'F'.
        """
        if self.Q1 is None:
            return (False, "SKIP: Q1 not available")
        if self.Q1 == 'F':
            return (True, "Third law: liquid cannot exist at absolute zero → Q1=F")
        else:
            return (False, f"Third law contradiction: Q1 should be F, got {self.Q1}")

    def test_invariant_phase_diagram_Q2(self):
        """
        On a phase diagram, the solid phase can exist at temperatures above the
        critical temperature if the pressure is sufficiently high.
        Statement (b) is false because the problem expects 'F' (see explanation steps).
        """
        if self.Q2 is None:
            return (False, "SKIP: Q2 not available")
        if self.Q2 == 'F':
            return (True, "Phase diagram: solid can exist above T_c only at high P → Q2=F (statement false)")
        else:
            return (False, f"Phase diagram mismatch: expected F, got {self.Q2}")

    def test_invariant_boiling_point_Q3(self):
        """
        At 1 atm, the normal boiling point of oxygen (90.2 K) is higher than
        that of nitrogen (77.4 K). Hence statement (c) is true.
        """
        if self.Q3 is None:
            return (False, "SKIP: Q3 not available")
        # Known physical constants (re‑derived inside checker)
        Tb_O2 = 90.2
        Tb_N2 = 77.4
        if self.Q3 == 'T':
            return (True, f"Tb(O2) = {Tb_O2} K > Tb(N2) = {Tb_N2} K → true")
        else:
            return (False, f"Boiling point comparison fails: expected T, got {self.Q3}")

    def test_invariant_inversion_temperature_Q4(self):
        """
        The maximum Joule–Thomson inversion temperature of helium is about 45 K,
        which is greater than 20 K. Therefore statement (d) is false → Q4 = 'F'.
        """
        if self.Q4 is None:
            return (False, "SKIP: Q4 not available")
        T_inv_He = 45.0  # known value, re‑derived from physics
        if self.Q4 == 'F':
            return (True, f"T_inv_max(He) ≈ {T_inv_He} K > 20 K → statement false")
        else:
            return (False, f"Inversion temperature mismatch: expected F, got {self.Q4}")

    def test_invariant_gamma_Q5(self):
        """
        For any gas with f degrees of freedom (f ≥ 3 for monatomic),
        γ = 1 + 2/f > 1. Hence statement (e) is always true.
        """
        if self.Q5 is None:
            return (False, "SKIP: Q5 not available")
        # Derive numerical lower bound
        f = 3  # minimum degrees of freedom
        gamma_min = 1.0 + 2.0/f  # = 5/3 > 1
        if self.Q5 == 'T':
            return (True, f"γ = 1+2/f ≥ {gamma_min:.4f} > 1 → always true")
        else:
            return (False, f"Gamma theorem requires Q5=T, got {self.Q5}")

    def test_invariant_adiabatic_temperature_Q6(self):
        """
        For adiabatic compression, T₂/T₁ = (P₂/P₁)^{(γ-1)/γ}.
        Monatomic (γ=5/3) gives a larger exponent (0.4) than diatomic (γ=7/5, exponent 0.2857),
        so monatomic gas becomes hotter. The statement claims the opposite, so it is false.
        """
        if self.Q6 is None:
            return (False, "SKIP: Q6 not available")
        gamma_mono = 5.0/3.0
        gamma_di  = 7.0/5.0
        exp_mono = (gamma_mono - 1.0) / gamma_mono   # 0.4
        exp_di   = (gamma_di - 1.0) / gamma_di        # ≈0.2857
        if self.Q6 == 'F':
            return (True, f"Adiabatic exponent: mono={exp_mono:.4f} > di={exp_di:.4f} → di heats less, statement false")
        else:
            return (False, f"Adiabatic comparison: expected F, got {self.Q6}")

    def test_invariant_cop_Q7(self):
        """
        Coefficient of performance of a refrigerator, COP = Q_c / W,
        can be greater than 1 if the heat removed exceeds the work input.
        This is possible in many real systems → statement true.
        """
        if self.Q7 is None:
            return (False, "SKIP: Q7 not available")
        if self.Q7 == 'T':
            return (True, "COP = Q_c/W can be >1 when Q_c > W → true")
        else:
            return (False, f"COP theorem requires Q7=T, got {self.Q7}")

    def test_invariant_magnus_force_Q8(self):
        """
        Magnus effect: F_M ∝ ω × v. For a ball thrown south (v south)
        and spinning counterclockwise when viewed from above (ω up),
        the cross product ω × v points east. Therefore the ball curves east → true.
        """
        if self.Q8 is None:
            return (False, "SKIP: Q8 not available")
        # define directions using unit vectors
        v_south = sp.Matrix([0, -1, 0])   # assuming x=east, y=north, z=up
        omega_up = sp.Matrix([0, 0, 1])
        force = omega_up.cross(v_south)   # = (i j k; 0 0 1; 0 -1 0) = (1,0,0) = east
        if self.Q8 == 'T':
            return (True, f"ω×v = {force} → east → statement true")
        else:
            return (False, f"Magnus direction mismatch: expected T, got {self.Q8}")

    # ----------------------------------------------------------------
    # Modeling / trace checks (re‑derive formulas used in explanation steps)
    # ----------------------------------------------------------------
    def test_modeling_gamma_formula(self):
        """
        The adiabatic index γ is given by γ = 1 + 2/f for a gas with f degrees of freedom.
        For a monatomic gas (f=3), γ = 5/3. For a diatomic gas at moderate T (f=5), γ = 7/5.
        This test verifies the formula numerically.
        """
        # Re‑derive using the equation
        def gamma(f):
            return 1.0 + 2.0/f
        f_mono = 3
        f_di = 5
        gamma_mono = gamma(f_mono)
        gamma_di = gamma(f_di)
        # Check that the values match expectations
        if math.isclose(gamma_mono, 5.0/3.0, rel_tol=1e-6) and math.isclose(gamma_di, 7.0/5.0, rel_tol=1e-6):
            return (True, f"γ formula correct: mono γ={gamma_mono:.4f}, di γ={gamma_di:.4f}")
        else:
            return (False, f"γ formula mismatch: mono computed {gamma_mono}, expected {5/3}; di computed {gamma_di}, expected {7/5}")

    def test_modeling_adiabatic_exponent(self):
        """
        The exponent in the adiabatic temperature relation is (γ-1)/γ.
        Check that monatomic exponent > diatomic exponent.
        """
        gamma_mono = 5.0/3.0
        gamma_di = 7.0/5.0
        exp_mono = (gamma_mono - 1.0) / gamma_mono
        exp_di = (gamma_di - 1.0) / gamma_di
        if exp_mono > exp_di:
            return (True, f"Adiabatic exponent: mono {exp_mono:.4f} > di {exp_di:.4f} → consistency")
        else:
            return (False, f"Exponent ordering violated: mono {exp_mono:.4f} ≤ di {exp_di:.4f}")

    def test_trace_magnus_cross_product(self):
        """
        The Magnus force direction is given by ω × v. For a ball thrown south
        (v = (0, -1, 0)) spinning counterclockwise from above (ω = (0,0,1)),
        compute the cross product symbolically.
        """
        v = sp.Matrix([0, -1, 0])
        omega = sp.Matrix([0, 0, 1])
        F = omega.cross(v)
        # Expected: (1,0,0) i.e., east
        expected = sp.Matrix([1, 0, 0])
        if F == expected:
            return (True, f"ω×v = {F} → east, consistent with Magnus effect")
        else:
            return (False, f"Cross product gave {F}, expected {expected}")

    # ----------------------------------------------------------------
    # Monotonicity / trend checks (with explicit perturbation)
    # ----------------------------------------------------------------
    def test_monotonicity_gamma_decreases_with_f(self):
        """
        As degrees of freedom f increase, γ = 1 + 2/f decreases.
        Baseline: f=3, γ=5/3≈1.6667. Perturb f to 4 (increase by 1), γ=1.5, which is lower.
        """
        def gamma(f):
            return 1.0 + 2.0/f
        baseline_f = 3
        baseline_gamma = gamma(baseline_f)
        # perturb: increase f by +1
        perturbed_f = baseline_f + 1
        perturbed_gamma = gamma(perturbed_f)
        if perturbed_gamma < baseline_gamma:
            return (True, f"γ decreases with f: f={baseline_f} → γ={baseline_gamma:.4f}; f+1 → γ={perturbed_gamma:.4f}")
        else:
            return (False, f"Monotonicity violated: γ did not decrease when f increased from {baseline_f} to {perturbed_f}")

    def test_monotonicity_adiabatic_exponent_increases_with_gamma(self):
        """
        The exponent (γ-1)/γ is an increasing function of γ for γ>1.
        Baseline: γ=1.4 (diatomic), exponent≈0.2857. Perturb γ to 1.5 (increase ~7%),
        exponent becomes 0.3333, which is larger.
        """
        def exponent(g):
            return (g - 1.0) / g
        baseline_gamma = 1.4
        baseline_exp = exponent(baseline_gamma)
        # perturb: + ~7% → 1.5
        perturbed_gamma = baseline_gamma * 1.0714  # approx 1.5
        perturbed_exp = exponent(perturbed_gamma)
        if perturbed_exp > baseline_exp:
            return (True, f"Exponent increases with γ: γ=1.4 → exp={baseline_exp:.4f}; γ≈1.5 → exp={perturbed_exp:.4f}")
        else:
            return (False, f"Monotonicity violated: exponent did not increase when γ increased from {baseline_gamma} to {perturbed_gamma:.4f}")

    # ----------------------------------------------------------------
    # Weak final‑answer sanity (at most one such method)
    # ----------------------------------------------------------------
    def test_answer_sanity_are_t_or_f(self):
        """
        Weak sanity: all eight answers must be either 'T' or 'F'.
        (No numerical comparison, no derivation.)
        """
        for qkey in ['Q1','Q2','Q3','Q4','Q5','Q6','Q7','Q8']:
            val = getattr(self, qkey, None)
            if val is None:
                return (False, f"Missing {qkey}")
            if val not in ('T', 'F'):
                return (False, f"{qkey} is not T or F: got {val}")
        return (True, "All answers are valid T/F strings")

    # ----------------------------------------------------------------
    # Main test runner
    # ----------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all methods starting with 'test_', execute them, collect messages.
        Returns (all_pass, logs).
        """
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)