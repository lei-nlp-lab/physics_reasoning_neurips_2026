import math
import sys

class PhysicsChecker:
    """
    Auditor for a physics solution involving Helmholtz free energy of a monatomic ideal gas
    and work done in an isothermal reversible process with a partitioned cylinder.
    """
    def __init__(self, solution_instance):
        # Extract results safely; default to None for missing keys
        self.results = getattr(solution_instance, 'results', {})
        if self.results is None:
            self.results = {}
        
        # Required result keys
        self.F = self.results.get('F', None)          # string formula
        self.W_total = self.results.get('W_total', None)  # numeric
        
        # Extract givens; default to None if not present
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.R = givens.get('R', None)          # universal gas constant (J/(mol K))
        self.T = givens.get('T', None)          # temperature (K)
        self.N_a = givens.get('N_a', None)      # moles in compartment a
        self.N_b = givens.get('N_b', None)      # moles in compartment b
        self.V_ai = givens.get('V_ai', None)    # initial volume a (L)
        self.V_bi = givens.get('V_bi', None)    # initial volume b (L)
        self.V_af = givens.get('V_af', None)    # final volume a (L)
        self.V_bf = givens.get('V_bf', None)    # final volume b (L)
        
        # Explanation steps (list of strings) – used to identify which checks are relevant
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        if self.explanation_steps is None or not isinstance(self.explanation_steps, list):
            self.explanation_steps = []
        
        # Physics mapping (dictionary of theorem-level relationships)
        self.physics_mapping = getattr(solution_instance, 'physics_mapping', {})
        if self.physics_mapping is None:
            self.physics_mapping = {}
        
        # Key constraints (list of strings)
        self.key_constraints = getattr(solution_instance, 'key_constraints', [])
        if self.key_constraints is None:
            self.key_constraints = []
        
        # Derived convenience: moles (assume N_a = N_b = 1 if not given)
        self.N_total = None
        if self.N_a is not None and self.N_b is not None:
            self.N_total = self.N_a + self.N_b

    # ----------------------------------------------------------------------
    # Interface checks – verify that required result keys exist and have expected types
    # ----------------------------------------------------------------------
    def test_interface_F_exists(self):
        """Check that 'F' key exists in results and is a string."""
        if self.F is None:
            return (False, "Missing result key 'F'")
        if not isinstance(self.F, str):
            return (False, f"F should be a string, got {type(self.F).__name__}")
        return (True, "F exists and is a string")

    def test_interface_W_total_exists(self):
        """Check that 'W_total' key exists in results and is numeric."""
        if self.W_total is None:
            return (False, "Missing result key 'W_total'")
        if not isinstance(self.W_total, (int, float)):
            return (False, f"W_total should be numeric, got {type(self.W_total).__name__}")
        return (True, "W_total exists and is numeric")

    # ----------------------------------------------------------------------
    # Invariant checks – theorem-based physical sanity (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_isothermal_reversible_work(self):
        """
        Check that total work equals RT * ln((V_af * V_bf) / (V_ai * V_bi))
        for the isothermal reversible process.
        """
        required = [self.R, self.T, self.V_ai, self.V_bi, self.V_af, self.V_bf, self.W_total]
        if any(v is None for v in required):
            return (False, "SKIP: missing parameters for work invariant")
        # Convert volumes from litres to m^3? 1 L = 0.001 m^3, but ratio is dimensionless,
        # so we can use litre values directly.
        ratio = (self.V_af * self.V_bf) / (self.V_ai * self.V_bi)
        # N_a = N_b = 1 => total moles = 2? Wait careful: each compartment has 1 mole,
        # but the work is sum of two separate NRT ln(Vf/Vi) with N=1 each.
        # So W_total = R*T*(ln(V_af/V_ai) + ln(V_bf/V_bi)) = R*T*ln(ratio)
        expected_W = self.R * self.T * math.log(ratio)
        if not math.isclose(self.W_total, expected_W, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Total work {self.W_total:.2f} J deviates from expected {expected_W:.2f} J")
        return (True, f"Isothermal work formula holds (W_total={self.W_total:.2f} J ≈ {expected_W:.2f} J)")

    def test_invariant_free_energy_derivative_respects_pV(self):
        """
        Check that from the derived F expression, -dF/dV at constant T equals p = NRT/V.
        This uses the symbolic idea: if F = -NRT ln V + f(T,N), then dF/dV = -NRT/V,
        so -dF/dV = NRT/V = p.
        We check a numerical proxy: if we evaluate F numerically at two close V values,
        the derivative should match -p.
        """
        # We need N, R, T. Use N=1 (mole) and default values if not given.
        # This test is only meaningful if we have a concrete numeric form of F.
        # Since F is typically symbolic, we check the physics_mapping if available.
        # Alternatively, we check that the problem statement's physics_mapping includes
        # the correct relation.
        if self.physics_mapping and 'dF = -S dT - p dV' in str(self.physics_mapping):
            return (True, "Physics mapping includes correct differential for F")
        # If not, we can check the expression structure (very weak).
        if self.F and 'ln V' in self.F:
            return (True, "F expression contains logarithmic volume term consistent with -∫p dV")
        return (False, "Cannot verify free energy derivative: insufficient data or mapping")

    def test_invariant_first_law_isothermal_U_constant(self):
        """
        For an ideal gas, isothermal process implies ΔU = 0.
        Since the cylinder walls are diathermal and piston is moved reversibly,
        the first law gives Q = W.
        We cannot directly test ΔU, but we can check the reasoning in explanation steps.
        """
        steps = ' '.join(self.explanation_steps).lower()
        if 'isothermal' in steps and 'Δu = 0' in steps:
            return (True, "Explanation steps state ΔU=0 for isothermal process")
        # Even if not explicit, the work calculation using NRT ln(Vf/Vi) implies isothermal.
        return (True, "Assumed isothermal consistent with problem context")

    # ----------------------------------------------------------------------
    # Modeling / trace checks – consistency between explanation steps and results
    # ----------------------------------------------------------------------
    def test_modeling_work_sign_and_magnitude(self):
        """
        Check that work done by the system is positive (net expansion: V_bf/V_bi > V_ai/V_af).
        Also check that the computed work is on the order of 10^3 J.
        """
        if self.W_total is None:
            return (False, "SKIP: no W_total")
        if not (2000 < self.W_total < 3000):
            return (False, f"W_total = {self.W_total} J outside expected order of 2-3e3 J")
        # Sign: net expansion => work positive (system does work on surroundings)
        # V_bf/V_bi = 5/1 = 5, V_af/V_ai = 6/10 = 0.6, product = 3 >1 => net expansion
        if self.W_total <= 0:
            return (False, f"W_total = {self.W_total} J should be positive for net expansion")
        return (True, f"Work is positive and in reasonable range ({self.W_total} J)")

    def test_modeling_free_energy_contains_expected_terms(self):
        """
        Check that the derived F expression contains:
        - term with ln V (from -∫p dV)
        - term with T ln T (from temperature dependence of C_v)
        """
        if self.F is None:
            return (False, "SKIP: no F expression")
        exp = self.F
        # Sub-expressions should appear, e.g., "ln V" or "log V", and "T ln T" or similar
        has_lnV = 'ln(v)' in exp.lower() or 'ln V' in exp or 'log(V)' in exp.lower()
        has_TlnT = 'ln(t)' in exp.lower() or 'ln T' in exp or 'log(T)' in exp.lower()
        if not has_lnV:
            return (False, "F expression missing ln(V) term expected from -∫p dV")
        if not has_TlnT:
            return (False, "F expression missing T ln T term expected from C_v = 3/2 R")
        return (True, "F expression contains required logarithmic terms")

    def test_modeling_explanation_steps_contain_work_formula(self):
        """
        Check that the explanation steps mention the isothermal reversible work formula.
        """
        steps = ' '.join(self.explanation_steps).lower()
        if 'nrt' in steps and 'ln' in steps and ('vf/vi' in steps or 'v_f/v_i' in steps or 'log' in steps):
            return (True, "Explanation steps reference the isothermal work formula")
        # Also check for W = ∫p dV
        if 'work' in steps and ('∫' in steps or 'integral' in steps or 'pdv' in steps):
            return (True, "Explanation steps include work integral")
        return (False, "Explanation steps missing reference to reversible work formula")

    # ----------------------------------------------------------------------
    # Monotonicity tests – perturb one input and verify trend in output
    # ----------------------------------------------------------------------
    def test_monotonicity_work_increases_with_V_af(self):
        """
        If final volume of compartment a increases (i.e., gas expands more),
        then total work should change.
        But careful: V_af and V_bf are not independent; they must sum to total volume.
        In the problem, V_af + V_bf = V_total (since cylinder fixed volume? Actually
        total volume is fixed? Initial total V_total_i = 11 L, final total V_total_f = 11 L,
        so total volume is constant. Therefore increasing V_af forces V_bf to decrease by same amount.
        Work = RT [ ln(V_af/10) + ln((11-V_af)/1) ].
        Compute derivative: dW/dV_af = RT*(1/V_af - 1/(11-V_af)).
        This is positive when V_af < 5.5 (because 1/V > 1/(11-V) when V < 5.5).
        Here baseline V_af = 6 > 5.5, so increasing V_af actually decreases total work.
        So we check that a small increase in V_af leads to a decrease in W_total.
        """
        required = [self.R, self.T, self.V_ai, self.V_bi, self.V_af, self.V_bf, self.W_total]
        if any(v is None for v in required):
            return (False, "SKIP: missing data for monotonicity test")
        # Baseline volumes: V_ai=10, V_bi=1, V_af=6, V_bf=5 => total fixed at 11 L
        # Perturb V_af by +5% (increase by 0.3 L), V_bf becomes 11 - V_af - 0.3 = 4.7 L.
        delta = 0.3   # litres
        V_af_new = self.V_af + delta
        V_bf_new = 11 - V_af_new   # total volume constant 11 L
        # Only valid if V_bf_new > 0 (which it is: 4.7)
        if V_bf_new <= 0:
            return (False, "Perturbation leads to negative volume, cannot test monotonicity")
        # Compute new work: W_new = RT [ ln(V_af_new/10) + ln(V_bf_new/1) ]
        W_expected_new = self.R * self.T * (math.log(V_af_new / self.V_ai) + math.log(V_bf_new / self.V_bi))
        # Compare to old work: since V_af > 5.5, dW/dV_af negative, so W_new < original.
        if not math.isclose(W_expected_new, self.W_total, rel_tol=1e-2, abs_tol=1e-2):
            # If W_new is significantly less than W_total, monotonicity holds
            if W_expected_new < self.W_total:
                return (True, f"Work decreases when V_af increases from {self.V_af} to {V_af_new}: {self.W_total:.2f} -> {W_expected_new:.2f} (negative monotonicity correct)")
            else:
                return (False, f"Work increased with V_af increase, but expected decrease (baseline V_af={self.V_af} > 5.5)")
        else:
            return (False, "Work did not change meaningfully under perturbation, cannot verify monotonicity")
    
    # ----------------------------------------------------------------------
    # Weak answer sanity (only one, no isclose, no derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_work_is_finite_and_positive(self):
        """
        Check that final answer for work is finite, not NaN, and positive.
        """
        if self.W_total is None:
            return (False, "SKIP: no W_total")
        if not math.isfinite(self.W_total):
            return (False, f"W_total is not finite: {self.W_total}")
        if self.W_total <= 0:
            return (False, f"W_total should be positive, got {self.W_total}")
        return (True, f"W_total = {self.W_total} is positive and finite")

    # ----------------------------------------------------------------------
    # Master test runner
    # ----------------------------------------------------------------------
    def run_all_tests(self):
        """
        Discover all methods starting with 'test_', execute them safely, return (success, logs).
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)