import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for the thermal interaction problem:
    Two systems with heat capacities C1 and C2, initial temperatures T1 and T2,
    reach common final temperature Tf. Energy is conserved.
    The expected result is T2.
    """
    def __init__(self, solution_instance):
        # Extract inputs and results safely with None defaults
        self.C1 = getattr(solution_instance, 'C1', solution_instance.results.get('C1', None) if hasattr(solution_instance, 'results') else None)
        self.C2 = getattr(solution_instance, 'C2', solution_instance.results.get('C2', None) if hasattr(solution_instance, 'results') else None)
        self.T1 = getattr(solution_instance, 'T1', solution_instance.results.get('T1', None) if hasattr(solution_instance, 'results') else None)
        self.Tf = getattr(solution_instance, 'Tf', solution_instance.results.get('Tf', None) if hasattr(solution_instance, 'results') else None)
        self.T2 = getattr(solution_instance, 'T2', solution_instance.results.get('T2', None) if hasattr(solution_instance, 'results') else None)
        # Also store the solution_instance reference for explanation_steps etc.
        self.solution_instance = solution_instance

    # ---------- Interface Tests ----------
    def test_interface_has_C1(self):
        if self.C1 is None:
            return (False, "C1 is missing (None)")
        if not isinstance(self.C1, (int, float)):
            return (False, f"C1 has type {type(self.C1).__name__}, expected numeric")
        return (True, "C1 present and numeric")

    def test_interface_has_C2(self):
        if self.C2 is None:
            return (False, "C2 is missing (None)")
        if not isinstance(self.C2, (int, float)):
            return (False, f"C2 has type {type(self.C2).__name__}, expected numeric")
        return (True, "C2 present and numeric")

    def test_interface_has_T1(self):
        if self.T1 is None:
            return (False, "T1 is missing (None)")
        if not isinstance(self.T1, (int, float)):
            return (False, f"T1 has type {type(self.T1).__name__}, expected numeric")
        return (True, "T1 present and numeric")

    def test_interface_has_Tf(self):
        if self.Tf is None:
            return (False, "Tf is missing (None)")
        if not isinstance(self.Tf, (int, float)):
            return (False, f"Tf has type {type(self.Tf).__name__}, expected numeric")
        return (True, "Tf present and numeric")

    def test_interface_has_T2(self):
        if self.T2 is None:
            return (False, "T2 (result) is missing (None)")
        if not isinstance(self.T2, (int, float)):
            return (False, f"T2 has type {type(self.T2).__name__}, expected numeric")
        return (True, "T2 present and numeric")

    # ---------- Invariant Tests (use math.isclose) ----------
    def test_invariant_energy_conservation(self):
        missing = []
        for name, val in [('C1', self.C1), ('C2', self.C2), ('T1', self.T1), ('Tf', self.Tf), ('T2', self.T2)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"SKIP: missing {', '.join(missing)}")
        # Conservation: C1*(Tf - T1) + C2*(Tf - T2) = 0
        lhs = self.C1 * (self.Tf - self.T1) + self.C2 * (self.Tf - self.T2)
        if math.isclose(lhs, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Energy conserved: lhs = {lhs:.6e}")
        else:
            return (False, f"Energy not conserved: lhs = {lhs:.6e} (expected near 0)")

    def test_invariant_temperature_order(self):
        missing = []
        for name, val in [('T1', self.T1), ('Tf', self.Tf), ('T2', self.T2)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"SKIP: missing {', '.join(missing)}")
        # Tf must lie between T1 and T2 (strictly if both are distinct)
        if self.Tf == self.T1 or self.Tf == self.T2:
            pass
        if (self.T1 < self.Tf < self.T2) or (self.T2 < self.Tf < self.T1):
            return (True, f"Temperature order correct: T1={self.T1}, Tf={self.Tf}, T2={self.T2}")
        else:
            return (False, f"Temperature order violated: T1={self.T1}, Tf={self.Tf}, T2={self.T2}")

    def test_invariant_heat_capacities_positive(self):
        missing = []
        if self.C1 is None:
            missing.append('C1')
        if self.C2 is None:
            missing.append('C2')
        if missing:
            return (False, f"SKIP: missing {', '.join(missing)}")
        if self.C1 <= 0:
            return (False, f"C1 = {self.C1} is not positive")
        if self.C2 <= 0:
            return (False, f"C2 = {self.C2} is not positive")
        return (True, "Heat capacities are positive")

    # ---------- Modeling / Trace Tests (at least two) ----------
    def test_modeling_T2_derivation(self):
        """Check that T2 matches the analytic formula derived from energy conservation."""
        missing = []
        for name, val in [('C1', self.C1), ('C2', self.C2), ('T1', self.T1), ('Tf', self.Tf), ('T2', self.T2)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"SKIP: missing {', '.join(missing)}")
        try:
            T2_formula = self.Tf + (self.C1 / self.C2) * (self.Tf - self.T1)
        except ZeroDivisionError:
            return (False, "C2 is zero, cannot compute T2 formula")
        if math.isclose(self.T2, T2_formula, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"T2 = {self.T2} matches formula-derived T2 = {T2_formula}")
        else:
            return (False, f"T2 = {self.T2} does not match formula-derived T2 = {T2_formula}")

    def test_modeling_heat_transfer_equality(self):
        """
        Check the equality stated in the explanation steps:
        C1*(Tf - T1) = C2*(T2 - Tf)
        This is the form used in the reasoning and must hold.
        """
        missing = []
        for name, val in [('C1', self.C1), ('C2', self.C2), ('T1', self.T1), ('Tf', self.Tf), ('T2', self.T2)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"SKIP: missing {', '.join(missing)}")
        lhs = self.C1 * (self.Tf - self.T1)
        rhs = self.C2 * (self.T2 - self.Tf)
        if math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"C1*(Tf-T1) = {lhs:.6e} equals C2*(T2-Tf) = {rhs:.6e}")
        else:
            return (False, f"C1*(Tf-T1) = {lhs:.6e} does not equal C2*(T2-Tf) = {rhs:.6e}")

    # ---------- Monotonicity Test (perturbation) ----------
    def test_monotonicity_T2_with_C1(self):
        missing = []
        for name, val in [('C1', self.C1), ('C2', self.C2), ('T1', self.T1), ('Tf', self.Tf), ('T2', self.T2)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"SKIP: missing {', '.join(missing)}")
        delta_C1 = 0.1 * self.C1  # +10%
        C1_perturbed = self.C1 + delta_C1
        try:
            T2_perturbed = self.Tf + (C1_perturbed / self.C2) * (self.Tf - self.T1)
        except ZeroDivisionError:
            return (False, "C2 is zero, cannot compute perturbed T2")
        delta_T2 = T2_perturbed - self.T2
        expected_sign = (delta_C1 * (self.Tf - self.T1)) > 0
        actual_sign = (delta_T2 > 0)
        if expected_sign == actual_sign:
            return (True, f"Monotonicity holds: +10% C1 changes T2 by {delta_T2:.6f} (consistent sign)")
        else:
            return (False, f"Monotonicity violated: C1 increase by {delta_C1:.6f} leads to T2 change {delta_T2:.6f}, expected sign {expected_sign}")

    # ---------- Answer Sanity (weak, at most one) ----------
    def test_answer_sanity_T2_finite_positive(self):
        if self.T2 is None:
            return (False, "SKIP: T2 not available")
        if not math.isfinite(self.T2):
            return (False, f"T2 is not finite (value: {self.T2})")
        if self.T2 <= 0:
            return (False, f"T2 = {self.T2} is not positive (but may still be physically acceptable)")
        return (True, "T2 is finite and positive (weak sanity)")

    # ---------- Run All Tests ----------
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)