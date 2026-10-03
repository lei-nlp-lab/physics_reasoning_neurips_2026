import math
import sys
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract results using None defaults
        self.pion_properties = getattr(solution_instance, 'pion_properties', None) or solution_instance.results.get('pion_properties', None)
        self.J_pi_plus = getattr(solution_instance, 'J_pi_plus', None) or solution_instance.results.get('J_pi_plus', None)
        self.J_pi_minus = getattr(solution_instance, 'J_pi_minus', None) or solution_instance.results.get('J_pi_minus', None)
        self.J_pi0 = getattr(solution_instance, 'J_pi0', None) or solution_instance.results.get('J_pi0', None)
        self.P_pi_minus = getattr(solution_instance, 'P_pi_minus', None) or solution_instance.results.get('P_pi_minus', None)
        self.P_pi_plus = getattr(solution_instance, 'P_pi_plus', None) or solution_instance.results.get('P_pi_plus', None)
        self.P_pi0 = getattr(solution_instance, 'P_pi0', None) or solution_instance.results.get('P_pi0', None)
        self.I_pion = getattr(solution_instance, 'I_pion', None) or solution_instance.results.get('I_pion', None)

        # Explanation steps for modeling tests
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---------- Interface tests ----------
    def test_interface_pion_properties_exists(self):
        if self.pion_properties is None:
            return (False, "SKIP: pion_properties not available")
        if not isinstance(self.pion_properties, str):
            return (False, "pion_properties should be a string")
        return (True, "pion_properties is present and is a string")

    def test_interface_J_pi_plus_type(self):
        if self.J_pi_plus is None:
            return (False, "SKIP: J_pi_plus not available")
        if not isinstance(self.J_pi_plus, (int, float)):
            return (False, "J_pi_plus should be numeric")
        return (True, "J_pi_plus is numeric")

    def test_interface_J_pi_minus_type(self):
        if self.J_pi_minus is None:
            return (False, "SKIP: J_pi_minus not available")
        if not isinstance(self.J_pi_minus, (int, float)):
            return (False, "J_pi_minus should be numeric")
        return (True, "J_pi_minus is numeric")

    def test_interface_J_pi0_type(self):
        if self.J_pi0 is None:
            return (False, "SKIP: J_pi0 not available")
        if not isinstance(self.J_pi0, (int, float)):
            return (False, "J_pi0 should be numeric")
        return (True, "J_pi0 is numeric")

    def test_interface_P_pi_minus_type(self):
        if self.P_pi_minus is None:
            return (False, "SKIP: P_pi_minus not available")
        if not isinstance(self.P_pi_minus, (int, float)):
            return (False, "P_pi_minus should be numeric")
        return (True, "P_pi_minus is numeric")

    def test_interface_P_pi_plus_type(self):
        if self.P_pi_plus is None:
            return (False, "SKIP: P_pi_plus not available")
        if not isinstance(self.P_pi_plus, (int, float)):
            return (False, "P_pi_plus should be numeric")
        return (True, "P_pi_plus is numeric")

    def test_interface_P_pi0_type(self):
        if self.P_pi0 is None:
            return (False, "SKIP: P_pi0 not available")
        if not isinstance(self.P_pi0, (int, float)):
            return (False, "P_pi0 should be numeric")
        return (True, "P_pi0 is numeric")

    def test_interface_I_pion_type(self):
        if self.I_pion is None:
            return (False, "SKIP: I_pion not available")
        if not isinstance(self.I_pion, (int, float)):
            return (False, "I_pion should be numeric")
        return (True, "I_pion is numeric")

    # ---------- Invariant tests (theorem-based physical sanity) ----------
    def test_invariant_spin_boson(self):
        # All pions are bosons, spin must be integer
        if any(v is None for v in [self.J_pi_plus, self.J_pi_minus, self.J_pi0]):
            return (False, "SKIP: one or more spin values missing")
        for J in [self.J_pi_plus, self.J_pi_minus, self.J_pi0]:
            if not isinstance(J, int):
                return (False, f"Spin {J} is not integer; pions are bosons")
        return (True, "All spins are integers")

    def test_invariant_parity_pseudoscalar(self):
        # Pions are pseudoscalars, parity should be -1
        if any(v is None for v in [self.P_pi_plus, self.P_pi_minus, self.P_pi0]):
            return (False, "SKIP: one or more parity values missing")
        if not (math.isclose(self.P_pi_plus, -1, rel_tol=1e-9) and
                math.isclose(self.P_pi_minus, -1, rel_tol=1e-9) and
                math.isclose(self.P_pi0, -1, rel_tol=1e-9)):
            return (False, "All pions should have parity -1 (pseudoscalar)")
        return (True, "All parity values are -1")

    def test_invariant_isospin_triplet(self):
        # Pions form an isospin triplet with I=1
        if self.I_pion is None:
            return (False, "SKIP: I_pion not available")
        if not math.isclose(self.I_pion, 1, rel_tol=1e-9):
            return (False, "Isospin should be 1")
        return (True, "Isospin is 1")

    def test_invariant_spin_uniform(self):
        # All pions have same spin (0)
        if any(v is None for v in [self.J_pi_plus, self.J_pi_minus, self.J_pi0]):
            return (False, "SKIP: one or more spin values missing")
        if not (math.isclose(self.J_pi_plus, 0, abs_tol=1e-9) and
                math.isclose(self.J_pi_minus, 0, abs_tol=1e-9) and
                math.isclose(self.J_pi0, 0, abs_tol=1e-9)):
            return (False, "All spins must be 0 (spinless bosons)")
        return (True, "All spins equal 0")

    def test_invariant_parity_uniform(self):
        # All pions have same parity
        if any(v is None for v in [self.P_pi_plus, self.P_pi_minus, self.P_pi0]):
            return (False, "SKIP: one or more parity values missing")
        if not (math.isclose(self.P_pi_plus, self.P_pi_minus, rel_tol=1e-9) and
                math.isclose(self.P_pi_plus, self.P_pi0, rel_tol=1e-9)):
            return (False, "All parity values must be equal")
        return (True, "All parity values are equal")

    def test_invariant_isospin_positive_integer(self):
        if self.I_pion is None:
            return (False, "SKIP: I_pion not available")
        if self.I_pion <= 0 or not isinstance(self.I_pion, int):
            return (False, "Isospin must be a positive integer")
        return (True, "Isospin is a positive integer")

    # ---------- Modeling/trace tests ----------
    def test_modeling_parity_reaction_matches_explanation(self):
        # Check that explanation_steps contain key statements about parity determination
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        if not isinstance(self.explanation_steps, str):
            return (False, "explanation_steps should be a string")
        keywords = ['pi-', 'd', 'nn', 'parity', 'L=1', 'p-wave']
        missing = [kw for kw in keywords if kw not in self.explanation_steps]
        if missing:
            return (False, f"Missing key terms in explanation_steps: {missing}")
        return (True, "Explanation steps contain correct parity derivation language")

    def test_modeling_isospin_cross_section_ratio(self):
        # Check that explanation mentions the 3:1 ratio or Delta resonance
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        keywords = ['sigma(pi+ p)', 'sigma(pi- p)', '3:1', 'Delta', 'I=3/2']
        found = any(kw in self.explanation_steps for kw in keywords)
        if not found:
            return (False, "Explanation steps should discuss cross section ratio for isospin")
        return (True, "Explanation steps mention isospin determination from cross sections")

    def test_modeling_spin_decay_angular_distribution(self):
        # Check that explanation mentions isotropy for spin-0 decay
        if self.explanation_steps is None:
            return (False, "SKIP: explanation_steps not available")
        keywords = ['isotropic', 'angular distribution', 'cos²θ', 'J=0']
        found = any(kw in self.explanation_steps for kw in keywords)
        if not found:
            return (False, "Explanation steps should discuss decay angular distribution for spin")
        return (True, "Explanation steps mention angular distribution for spin determination")

    # ---------- Monotonicity tests ----------
    def test_monotonicity_isospin_number_of_states(self):
        """
        Check that number of pion charge states (2I+1) increases with I.
        Perturb I_pion by +0.1 and verify the number increases.
        """
        if self.I_pion is None:
            return (False, "SKIP: I_pion not available")
        if not isinstance(self.I_pion, (int, float)):
            return (False, "I_pion must be numeric")
        baseline_states = 2 * self.I_pion + 1
        # Perturb I slightly upward
        perturbed_I = self.I_pion + 0.1
        perturbed_states = 2 * perturbed_I + 1
        if perturbed_states <= baseline_states:
            return (False, "Number of states did not increase when isospin increased")
        return (True, f"Number of states increases with isospin (baseline: {baseline_states}, perturbed: {perturbed_states:.1f})")

    def test_monotonicity_spin_polarization_count(self):
        """
        Check that the number of spin polarization states (2J+1) increases with J.
        Perturb J_pi_plus by +1 and verify the count increases.
        """
        if self.J_pi_plus is None:
            return (False, "SKIP: J_pi_plus not available")
        if not isinstance(self.J_pi_plus, (int, float)):
            return (False, "J_pi_plus must be numeric")
        baseline_pol = 2 * self.J_pi_plus + 1
        perturbed_J = self.J_pi_plus + 1.0
        perturbed_pol = 2 * perturbed_J + 1
        if perturbed_pol <= baseline_pol:
            return (False, "Number of polarization states did not increase when spin increased")
        return (True, f"Number of polarization states increases with spin (baseline: {baseline_pol}, perturbed: {perturbed_pol:.1f})")

    # ---------- Weak answer sanity test (only one) ----------
    def test_answer_sanity_plausibility(self):
        """
        Weak sanity: check that spin >= 0 integer, parity = ±1, isospin positive integer.
        No derivation, no math.isclose.
        """
        ok = True
        msg_parts = []
        spins = [('J_pi_plus', self.J_pi_plus), ('J_pi_minus', self.J_pi_minus), ('J_pi0', self.J_pi0)]
        for name, val in spins:
            if val is None:
                continue
            if not isinstance(val, int) or val < 0:
                ok = False
                msg_parts.append(f"{name} not non‑negative integer")
        parities = [('P_pi_plus', self.P_pi_plus), ('P_pi_minus', self.P_pi_minus), ('P_pi0', self.P_pi0)]
        for name, val in parities:
            if val is None:
                continue
            if val not in (1, -1):
                ok = False
                msg_parts.append(f"{name} not ±1")
        if self.I_pion is not None:
            if not isinstance(self.I_pion, int) or self.I_pion <= 0:
                ok = False
                msg_parts.append("I_pion not positive integer")
        if not ok:
            return (False, "; ".join(msg_parts))
        return (True, "All present values pass weak plausibility checks")

    # ---------- Run all tests ----------
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
                traceback.print_exc()
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)