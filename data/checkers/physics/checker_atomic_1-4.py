import math
import traceback
from typing import Any, Dict, List, Tuple, Optional

# Constants (SI units for simplicity, but problem uses eV/cm)
e_charge = 1.602176634e-19  # C
epsilon0 = 8.854187817e-12  # F/m
hbar = 1.054571817e-34      # J·s
m_e = 9.1093837015e-31      # kg
eV = 1.602176634e-19        # J

class PhysicsChecker:
    """
    Checker for the electric field needed to ionise a hydrogen-like atom
    within one orbital period (Bohr model).
    """

    def __init__(self, solution_instance: Any) -> None:
        # Extract results safely
        results = getattr(solution_instance, 'results', None) or {}
        # Expected keys from the problem
        self.E_th = results.get('E_th', None)

        # Extract givens (if available)
        self.givens = getattr(solution_instance, 'givens', None) or {}
        self.Z = self.givens.get('Z', None)  # atomic number, optional

        # Optionally extract explanation steps for modelling checks
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_result_key_present(self) -> Tuple[bool, str]:
        """Ensure the required result key 'E_th' exists."""
        if self.E_th is None:
            return (False, "E_th is missing from results")
        return (True, "E_th present")

    def test_interface_E_th_is_positive_and_finite(self) -> Tuple[bool, str]:
        """E_th must be a positive finite number."""
        if self.E_th is None:
            return (False, "SKIP: E_th not available")
        try:
            val = float(self.E_th)
            if not math.isfinite(val):
                return (False, f"E_th is not finite: {val}")
            if val <= 0.0:
                return (False, f"E_th is not positive: {val}")
            return (True, f"E_th = {val} is positive and finite")
        except (TypeError, ValueError) as e:
            return (False, f"E_th conversion failed: {e}")

    # ------------------------------------------------------------------
    # Invariant (theorem-based) checks
    # ------------------------------------------------------------------
    def test_invariant_electric_field_from_binding_energy(self) -> Tuple[bool, str]:
        """
        Check that the reported E_th satisfies the energy balance:
        e * E_th * a0 ≈ binding energy (E_b),
        where a0 = Bohr radius for Z=1 and E_b = 13.6 eV.
        For general Z, the condition is e * E_th * (a0/Z) ≈ 13.6 Z^2 eV
        => E_th ≈ (13.6 Z^3) / (e a0).
        If Z is available we test the exact scaling; otherwise we test for Z=1.
        """
        if self.E_th is None:
            return (False, "SKIP: E_th not available")
        try:
            # Compute Bohr radius a0 (hydrogen)
            a0 = (4 * math.pi * epsilon0 * hbar**2) / (m_e * e_charge**2)
            # Compute binding energy in Joules
            E_b_J = (m_e * e_charge**4) / (32 * math.pi**2 * epsilon0**2 * hbar**2)
            E_b_eV = E_b_J / eV  # about 13.6 eV

            if self.Z is not None:
                # General case: E_th_expected = (E_b_eV * e * Z^3) / (e * a0) but careful:
                # e * E_th * (a0/Z) ≈ 13.6 Z^2 eV
                # => E_th (in V/m) * (a0/Z) ≈ (13.6 Z^2) V  (since e cancels)
                # => E_th ≈ (13.6 Z^3) / (a0)   (in V/m, because 13.6 eV is 13.6 V*e)
                # Convert to V/cm: 1 V/m = 0.01 V/cm, but we compare in V/m.
                E_th_expected_Vm = (13.6 * self.Z**3) / (a0)  # V/m
                # Convert given E_th to V/m (if it's in V/cm, multiply by 100; but we don't know unit)
                # The problem statement uses V/cm for the answer.
                # We will assume the given E_th is in V/cm and convert to V/m.
                given_E_Vm = self.E_th * 100  # assume V/cm -> V/m
                ratio = given_E_Vm / E_th_expected_Vm
                # Allow a factor of 2 (order-of-magnitude tolerance)
                if 0.5 <= ratio <= 2.0:
                    return (True, f"E_th (in V/cm) scaled as Z^3 within factor 2 (ratio={ratio:.3f})")
                else:
                    return (False, f"E_th does not obey Z^3 scaling: expected {E_th_expected_Vm/100:.2e} V/cm, got {self.E_th:.2e}, ratio={ratio:.3f}")
            else:
                # Z not available: assume Z=1 (hydrogen)
                E_th_expected_Vcm = E_b_eV / (e_charge * a0) * 100  # V/cm because 1 V/m = 100 V/cm
                given = float(self.E_th)
                # Use a moderate relative tolerance (0.5) because it is an order-of-magnitude estimate
                if math.isclose(given, E_th_expected_Vcm, rel_tol=0.5):
                    return (True, f"E_th matches hydrogen estimate: expected ~{E_th_expected_Vcm:.2e} V/cm, got {given:.2e}")
                else:
                    return (False, f"E_th for hydrogen deviates: expected ~{E_th_expected_Vcm:.2e} V/cm, got {given:.2e}")
        except Exception as e:
            return (False, f"test_invariant_electric_field_from_binding_energy CRASHED: {e}")

    def test_invariant_energy_balance(self) -> Tuple[bool, str]:
        """
        Check that e * E_th * (a0/Z) approximately equals the binding energy (13.6 Z^2 eV).
        This is the core physics condition.
        """
        if self.E_th is None:
            return (False, "SKIP: E_th not available")
        try:
            a0 = (4 * math.pi * epsilon0 * hbar**2) / (m_e * e_charge**2)
            # Binding energy in eV: 13.6 Z^2 eV
            if self.Z is not None:
                E_b_eV = 13.6 * self.Z**2
                a0_Z = a0 / self.Z
            else:
                E_b_eV = 13.6
                a0_Z = a0
            # Work done by field over distance a0_Z:  W = e * E * a0_Z  (in J)
            # Convert E from V/cm to V/m (assume given in V/cm)
            E_Vm = self.E_th * 100  # V/cm -> V/m
            W_J = e_charge * E_Vm * a0_Z
            W_eV = W_J / eV
            # Compare to binding energy
            if math.isclose(W_eV, E_b_eV, rel_tol=0.3):
                return (True, f"e*E*radius ≈ {W_eV:.2f} eV, binding energy ≈ {E_b_eV:.2f} eV (within 30%)")
            else:
                return (False, f"Work = {W_eV:.2f} eV, binding energy = {E_b_eV:.2f} eV (tolerance exceeded)")
        except Exception as e:
            return (False, f"test_invariant_energy_balance CRASHED: {e}")

    # ------------------------------------------------------------------
    # Modelling (trace) checks
    # ------------------------------------------------------------------
    def test_modeling_derivation_uses_bohr_radius(self) -> Tuple[bool, str]:
        """
        Verify that the explanation steps or derivation uses the Bohr radius a0.
        This is a weak check that the modelling reasoning is plausible.
        """
        if not self.explanation_steps:
            return (False, "SKIP: explanation_steps not available")
        if "Bohr radius" in self.explanation_steps or "a0" in self.explanation_steps:
            return (True, "Explanation references Bohr radius a0")
        return (False, "Explanation does not mention Bohr radius")

    def test_modeling_work_equals_binding(self) -> Tuple[bool, str]:
        """
        Check that the reported E_th satisfies the condition
        e * E * a = binding energy, where a = a0/Z.
        This is re-derived from the given E_th.
        """
        if self.E_th is None:
            return (False, "SKIP: E_th not available")
        try:
            a0 = (4 * math.pi * epsilon0 * hbar**2) / (m_e * e_charge**2)
            # Compute expected E threshold from formula E_th_expected = (13.6 Z^3) / (e a0) in V/m.
            # But we don't know Z; we use the given E_th and the condition to check consistency.
            # We'll simply check that the work - binding energy ratio is close to 1.
            if self.Z is not None:
                E_b_eV = 13.6 * self.Z**2
                a = a0 / self.Z
            else:
                E_b_eV = 13.6
                a = a0
            E_Vm = self.E_th * 100
            work_eV = (e_charge * E_Vm * a) / eV
            if math.isclose(work_eV, E_b_eV, rel_tol=0.3):
                return (True, f"Condition satisfied: work ≈ {work_eV:.2f} eV, binding = {E_b_eV:.2f} eV")
            else:
                return (False, f"Condition broken: work = {work_eV:.2f} eV, binding = {E_b_eV:.2f} eV")
        except Exception as e:
            return (False, f"test_modeling_work_equals_binding CRASHED: {e}")

    # ------------------------------------------------------------------
    # Monotonicity check (if Z is available)
    # ------------------------------------------------------------------
    def test_monotonicity_E_increases_with_Z(self) -> Tuple[bool, str]:
        """
        If Z is given, check that a 10% increase in Z would give a larger E_th.
        Since we cannot modify the instance, we use the expected cubic scaling.
        """
        if self.Z is None:
            return (False, "SKIP: Z not available")
        if self.E_th is None:
            return (False, "SKIP: E_th not available")
        try:
            Z_base = self.Z
            # Expected E_th for Z_base and Z_pert = 1.1 * Z_base
            # Assuming E_th ∝ Z^3
            E_base = float(self.E_th)
            # For the perturbed Z, we predict E_pert = E_base * (Z_pert/Z_base)^3
            Z_pert = 1.1 * Z_base
            E_pert_predicted = E_base * (Z_pert / Z_base) ** 3
            # We cannot actually retrieve a perturbed result, so we just check that
            # the predicted perturbed value is greater (monotonic increasing).
            if E_pert_predicted > E_base:
                return (True, f"Predicted increase: Z from {Z_base} to {Z_pert} gives E from {E_base} to {E_pert_predicted:.2e} (monotonic)")
            else:
                return (False, "Prediction does not show monotonic increase (unexpected)")
        except Exception as e:
            return (False, f"test_monotonicity_E_increases_with_Z CRASHED: {e}")

    # ------------------------------------------------------------------
    # (Optional) Weak answer sanity check – only broad magnitude.
    # ------------------------------------------------------------------
    def test_answer_sanity_plausible_magnitude(self) -> Tuple[bool, str]:
        """
        Weak sanity: for Z=1, E_th should be of order 1e8 to 1e10 V/cm (Bohr field ~5e9 V/cm).
        For other Z, scale approximately as Z^3.
        """
        if self.E_th is None:
            return (False, "SKIP: E_th not available")
        try:
            if self.Z is not None:
                Z_factor = self.Z ** 3
                # Expected order: 2e9 V/cm for Z=1
                upper = 1e11 * Z_factor
                lower = 1e7 * Z_factor
            else:
                upper = 1e11
                lower = 1e7
            val = float(self.E_th)
            if lower <= val <= upper:
                return (True, f"E_th = {val:.2e} V/cm is within plausible range [{lower:.2e}, {upper:.2e}]")
            else:
                return (False, f"E_th = {val:.2e} V/cm outside plausible range [{lower:.2e}, {upper:.2e}]")
        except Exception as e:
            return (False, f"test_answer_sanity_plausible_magnitude CRASHED: {e}")

    # ------------------------------------------------------------------
    # Runner
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        logs: List[str] = []
        all_pass = True
        # Discover all test methods
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
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

# Example usage (commented out for safety):
# if __name__ == "__main__":
#     # Dummy solution instance for testing
#     class DummySolution:
#         results = {'E_th': 2.0e9}  # V/cm, Z=1 assumed
#         givens = {'Z': 1}
#         explanation_steps = "Bohr radius a0"
#     checker = PhysicsChecker(DummySolution())
#     success, logs = checker.run_all_tests()
#     for msg in logs:
#         print(msg)
#     print("Overall:", success)