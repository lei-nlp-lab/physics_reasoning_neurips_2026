import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results and givens safely
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}

        # Main result keys
        self.Delta_E = results.get('Delta_E', None)
        self.order = results.get('order', None)

        # Optional extraction for modeling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        self.pseudocode = getattr(solution_instance, 'pseudocode', None)

        # Store givens if provided (for monotonicity)
        self.a0_given = givens.get('a0', None)  # optional, not strictly needed

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_delta_e_exists(self):
        if self.Delta_E is None:
            return (False, "Missing key 'Delta_E' in results")
        return (True, "'Delta_E' present")

    def test_interface_order_exists(self):
        if self.order is None:
            return (False, "Missing key 'order' in results")
        return (True, "'order' present")

    def test_interface_delta_e_is_numeric(self):
        if not isinstance(self.Delta_E, (int, float)):
            return (False, f"'Delta_E' is not numeric: {type(self.Delta_E)}")
        return (True, "'Delta_E' is numeric")

    def test_interface_order_is_integer(self):
        if not isinstance(self.order, int):
            return (False, f"'order' is not an integer: {type(self.order)}")
        return (True, "'order' is integer")

    # ----------------------------------------------------------------------
    # Weak answer sanity (only sign, finite, broad range)
    # ----------------------------------------------------------------------
    def test_answer_sanity_weak(self):
        """Single weak answer sanity: check positivity and plausibility."""
        if self.Delta_E is None:
            return (False, "SKIP: 'Delta_E' missing")
        if self.Delta_E <= 0:
            return (False, f"Delta_E = {self.Delta_E} is not positive")
        # Broad range: hydrogen hyperfine splitting is ~10^-5 eV, accept 1e-7 to 1e-3
        if not (1e-7 <= self.Delta_E <= 1e-3):
            return (False, f"Delta_E = {self.Delta_E} outside expected [1e-7, 1e-3] eV")
        return (True, "Delta_E positive and within plausible broad magnitude")

    # ----------------------------------------------------------------------
    # Invariant/theorem-based checks (use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_order_of_magnitude(self):
        if self.order is None:
            return (False, "SKIP: 'order' missing")
        # Known order from Fermi contact evaluation: 10^{-5} eV
        if self.order != -5:
            return (False, f"order = {self.order}, expected -5")
        return (True, "order is -5 (10^{-5} eV)")

    def test_invariant_hyperfine_value(self):
        if self.Delta_E is None:
            return (False, "SKIP: 'Delta_E' missing")
        # Recompute the expected hyperfine splitting using Fermi contact formula
        # Constants in SI (J, T, m)
        a0 = 5.29177210903e-11  # m
        mu_0 = 4e-7 * math.pi   # H/m
        g_e = 2.00231930436256
        g_p = 5.5856946892
        mu_B_J = 9.274009994e-24  # J/T
        mu_N_J = 5.0507837461e-27 # J/T
        psi0_sq = 1.0 / (math.pi * a0**3)   # 1/m^3
        Delta_E_J = (2.0 * mu_0 / 3.0) * g_e * g_p * mu_B_J * mu_N_J * psi0_sq
        Delta_E_eV = Delta_E_J / 1.602176634e-19  # convert J to eV
        # Known experimental value ~5.9e-6 eV
        # Allow 2% relative tolerance (rel_tol=0.02)
        if not math.isclose(self.Delta_E, Delta_E_eV, rel_tol=0.02, abs_tol=1e-8):
            return (False,
                    f"Delta_E = {self.Delta_E:.3e} eV, but Fermi contact gives "
                    f"{Delta_E_eV:.3e} eV (tolerance 2%)")
        return (True, "Hyperfine value consistent with Fermi contact formula")

    def test_invariant_delta_e_positive_and_small(self):
        # Already covered by answer_sanity, but here we add a theorem‑based check
        # that it is consistent with known order 10^{-5} eV.
        if self.Delta_E is None:
            return (False, "SKIP: 'Delta_E' missing")
        # The known value is 5.9e-6 eV, so it must be between 1e-6 and 1e-4
        if not (1e-6 <= self.Delta_E <= 1e-4):
            return (False, f"Delta_E = {self.Delta_E} not in [1e-6, 1e-4]")
        return (True, "Delta_E in plausible range for hydrogen hyperfine splitting")

    # ----------------------------------------------------------------------
    # Modeling / trace checks (based on explanation_steps)
    # ----------------------------------------------------------------------
    def test_modeling_fermi_contact_formula_used(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        steps = self.explanation_steps if isinstance(self.explanation_steps, str) else str(self.explanation_steps)
        # Check that the Fermi contact interaction is mentioned
        keywords = ['Fermi', 'contact', 'hyperfine', '8π/3', '8*pi/3', 'mu_B', 'mu_N']
        found = any(k in steps for k in keywords)
        if not found:
            return (False, "Explanation steps do not reference Fermi contact formula")
        return (True, "Fermi contact formula referenced in explanation")

    def test_modeling_wavefunction_correct(self):
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps available")
        steps = self.explanation_steps if isinstance(self.explanation_steps, str) else str(self.explanation_steps)
        # Check that hydrogen 1s wavefunction or |ψ(0)|^2 is mentioned
        if '|ψ(0)|^2' not in steps and 'psi(0)' not in steps.lower():
            return (False, "No mention of electron density at nucleus")
        return (True, "Electron density at nucleus included")

    # ----------------------------------------------------------------------
    # Monotonicity tests
    # ----------------------------------------------------------------------
    def test_monotonicity_a0_decreases_energy(self):
        if self.Delta_E is None:
            return (False, "SKIP: 'Delta_E' missing")
        # Perturb a0: increase by 10%
        a0_original = 5.29177210903e-11
        a0_increased = a0_original * 1.10
        # Recompute Delta_E using the same formula as in test_invariant_hyperfine_value
        mu_0 = 4e-7 * math.pi
        g_e = 2.00231930436256
        g_p = 5.5856946892
        mu_B_J = 9.274009994e-24
        mu_N_J = 5.0507837461e-27
        psi0_sq_orig = 1.0 / (math.pi * a0_original**3)
        psi0_sq_new = 1.0 / (math.pi * a0_increased**3)
        # Original computed value (not the provided one, we use our calculation)
        Delta_E_orig = (2.0 * mu_0 / 3.0) * g_e * g_p * mu_B_J * mu_N_J * psi0_sq_orig / 1.602176634e-19
        Delta_E_new = (2.0 * mu_0 / 3.0) * g_e * g_p * mu_B_J * mu_N_J * psi0_sq_new / 1.602176634e-19
        # Verify that the provided Delta_E matches the original calculated value (it should)
        if not math.isclose(self.Delta_E, Delta_E_orig, rel_tol=0.02, abs_tol=1e-8):
            return (False, f"Provided Delta_E {self.Delta_E:.2e} differs from recomputed {Delta_E_orig:.2e}")
        # Now check monotonicity: increasing a0 should decrease Delta_E
        if Delta_E_new >= Delta_E_orig - 1e-10:  # allow tiny numerical error
            return (False, f"Energy did not decrease when a0 increased: orig={Delta_E_orig:.3e}, new={Delta_E_new:.3e}")
        return (True, "Monotonicity: increasing a0 decreases hyperfine energy (Delta_E ∝ 1/a0^3)")

    def test_monotonicity_energy_scales_with_g_e(self):
        if self.Delta_E is None:
            return (False, "SKIP: 'Delta_E' missing")
        # Increase g_e by 5% and recompute (should increase proportionally)
        mu_0 = 4e-7 * math.pi
        a0 = 5.29177210903e-11
        g_e_orig = 2.00231930436256
        g_p = 5.5856946892
        mu_B_J = 9.274009994e-24
        mu_N_J = 5.0507837461e-27
        psi0_sq = 1.0 / (math.pi * a0**3)
        Delta_E_orig = (2.0 * mu_0 / 3.0) * g_e_orig * g_p * mu_B_J * mu_N_J * psi0_sq / 1.602176634e-19
        # Provided value should match original
        if not math.isclose(self.Delta_E, Delta_E_orig, rel_tol=0.02, abs_tol=1e-8):
            return (False, f"Provided Delta_E {self.Delta_E:.2e} differs from recomputed {Delta_E_orig:.2e}")
        # Now perturb g_e upward by 5%
        g_e_new = g_e_orig * 1.05
        Delta_E_new = (2.0 * mu_0 / 3.0) * g_e_new * g_p * mu_B_J * mu_N_J * psi0_sq / 1.602176634e-19
        if not (Delta_E_new > Delta_E_orig + 1e-10):
            return (False, "Energy did not increase when g_e increased")
        # Also confirm proportional: ratio should be 1.05
        ratio = Delta_E_new / Delta_E_orig
        if not math.isclose(ratio, 1.05, rel_tol=1e-2):
            return (False, f"Energy ratio {ratio:.4f} not 1.05")
        return (True, "Monotonicity: energy increases proportionally with g_e")

    # ----------------------------------------------------------------------
    # Runner (exactly as specified)
    # ----------------------------------------------------------------------
    def run_all_tests(self):
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