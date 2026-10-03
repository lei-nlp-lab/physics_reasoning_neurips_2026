import sympy
import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Physics checker for heat flow through a wire with linearly varying radius.
    Checks interface consistency, physical invariants, modeling validity,
    monotonicity, and a weak answer sanity.
    """

    def __init__(self, solution_instance: Any) -> None:
        # Attempt to extract givens from the solution instance.
        # Supported sources: instance attributes, instance.results dict.
        self.L: Optional[float] = self._get_value(solution_instance, 'L')
        self.r_A: Optional[float] = self._get_value(solution_instance, 'r_A')
        self.r_B: Optional[float] = self._get_value(solution_instance, 'r_B')
        self.T_A: Optional[float] = self._get_value(solution_instance, 'T_A')
        self.T_B: Optional[float] = self._get_value(solution_instance, 'T_B')
        self.K: Optional[float] = self._get_value(solution_instance, 'K')

        # Extract results from the instance's results dict or attributes.
        self.Q: Optional[float] = self._get_value(solution_instance, 'Q', target='results')
        self.T_half: Optional[float] = self._get_value(solution_instance, 'T_half', target='results')

    def _get_value(self, inst: Any, key: str, target: str = 'both') -> Optional[float]:
        """
        Helper to extract a value from the solution instance.
        Priority: instance attribute (if target is 'attr' or 'both'),
                  instance.results dict (if target is 'results' or 'both').
        Returns None if not found.
        """
        val = None
        if target in ('attr', 'both'):
            if hasattr(inst, key):
                val = getattr(inst, key)
        if val is None and target in ('results', 'both'):
            if hasattr(inst, 'results') and isinstance(inst.results, dict):
                val = inst.results.get(key)
        return val

    def _is_close(self, a: float, b: float, rel_tol: float = 1e-2, abs_tol: float = 1e-2) -> bool:
        """Fuzzy numerical equality with relative and absolute tolerances."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)

    # ------------- Interface Tests -------------
    def test_interface_required_results_keys(self) -> Tuple[bool, str]:
        """Check that the results dict contains mandatory keys: 'Q' and 'T_half'."""
        if self.Q is None:
            return (False, "Required result 'Q' (heat flow rate) is missing.")
        if self.T_half is None:
            return (False, "Required result 'T_half' (temperature at L/2) is missing.")
        return (True, "Both 'Q' and 'T_half' present in results.")

    def test_interface_givens_types(self) -> Tuple[bool, str]:
        """Check that all givens are numeric (float or int) if provided."""
        givens = {
            'L': self.L, 'r_A': self.r_A, 'r_B': self.r_B,
            'T_A': self.T_A, 'T_B': self.T_B, 'K': self.K
        }
        missing = [k for k, v in givens.items() if v is None]
        non_numeric = [k for k, v in givens.items() if v is not None and not isinstance(v, (int, float))]
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing givens: {', '.join(missing)}")
        if non_numeric:
            msg_parts.append(f"Non‑numeric givens: {', '.join(non_numeric)}")
        if not msg_parts:
            return (True, "All provided givens are numeric.")
        return (False, "; ".join(msg_parts))

    # ------------- Invariant Tests -------------
    def test_invariant_constant_A_dT_dx(self) -> Tuple[bool, str]:
        """
        Invariant: In steady‑state, A(x)*dT/dx is constant along the wire.
        Using finite‑difference approximations at x=0 and x=L,
        we compare two estimates of Q computed from the reported T_half and givens.
        The two estimates should be equal within tolerances, and both should match the reported Q.
        """
        # Need all quantities for this check.
        req = [self.L, self.r_A, self.r_B, self.T_A, self.T_B, self.K, self.Q, self.T_half]
        if any(v is None for v in req):
            return (False, "SKIP: Missing required data for invariant test.")

        # Finite difference derivative at x=0: (T_half - T_A) / (L/2)
        dT_dx_0 = 2.0 * (self.T_half - self.T_A) / self.L
        A_0 = math.pi * self.r_A**2
        Q_estimate_0 = -self.K * A_0 * dT_dx_0

        # Finite difference derivative at x=L: (T_B - T_half) / (L/2)
        dT_dx_L = 2.0 * (self.T_B - self.T_half) / self.L
        A_L = math.pi * self.r_B**2
        Q_estimate_L = -self.K * A_L * dT_dx_L

        # Check that the two estimates are close to each other
        if not self._is_close(Q_estimate_0, Q_estimate_L):
            return (False, f"Q estimates differ: at x=0 Q={Q_estimate_0:g}, at x=L Q={Q_estimate_L:g}.")

        # Check that both estimates are close to the reported Q
        if self._is_close(Q_estimate_0, self.Q) and self._is_close(Q_estimate_L, self.Q):
            return (True, f"Invariant holds: Q_0={Q_estimate_0:g}, Q_L={Q_estimate_L:g}, reported Q={self.Q:g}.")
        else:
            return (False, f"Estimates do not match reported Q ({self.Q:g}): Q_0={Q_estimate_0:g}, Q_L={Q_estimate_L:g}.")

    def test_invariant_q_positive_when_temperature_decreases(self) -> Tuple[bool, str]:
        """
        Invariant: If T_A > T_B, then Q should be positive (heat flows from hot to cold).
        This is a direct consequence of the second law of thermodynamics.
        """
        if self.T_A is None or self.T_B is None or self.Q is None:
            return (False, "SKIP: Missing temperatures or Q for invariant check.")

        if self.T_A > self.T_B and self.Q <= 0:
            return (False, f"T_A > T_B but Q = {self.Q:g} (should be >0).")
        if self.T_A < self.T_B and self.Q >= 0:
            return (False, f"T_A < T_B but Q = {self.Q:g} (should be <0).")
        return (True, "Heat flow direction is physically correct.")

    # ------------- Modeling / Trace Tests -------------
    def test_modeling_q_from_integral(self) -> Tuple[bool, str]:
        """
        Re‑derive the heat flow rate using Fourier's law with variable cross‑section
        and compare with the reported Q.
        """
        # All givens must be present for this test.
        if any(v is None for v in [self.L, self.r_A, self.r_B, self.T_A, self.T_B, self.K]):
            return (False, "SKIP: Incomplete set of givens for Q derivation.")

        if self.Q is None:
            return (False, "Reported Q is missing; cannot compare.")

        # Derive Q from the physics:
        # Q = π * r_A * r_B * K * (T_A - T_B) / L
        expected_Q = math.pi * self.r_A * self.r_B * self.K * (self.T_A - self.T_B) / self.L

        if self._is_close(self.Q, expected_Q):
            return (True, f"Derived Q = {expected_Q:g}, reported Q = {self.Q:g}.")
        else:
            return (False, f"Derived Q = {expected_Q:g} differs from reported Q = {self.Q:g}.")

    def test_modeling_t_half_from_q(self) -> Tuple[bool, str]:
        """
        Derive T_half from the reported Q and givens using the temperature profile,
        then compare with the reported T_half.
        """
        # Need givens and Q.
        req = [self.L, self.r_A, self.r_B, self.T_A, self.T_B, self.K, self.Q]
        if any(v is None for v in req):
            return (False, "SKIP: Missing givens or Q for temperature derivation.")

        if self.T_half is None:
            return (False, "Reported T_half is missing; cannot compare.")

        # Compute r_half = r_A + (r_B - r_A) * (L/2) / L = (r_A + r_B)/2
        r_half = (self.r_A + self.r_B) / 2.0

        # Derive T_half from the integrated Fourier law:
        # integral from 0 to x: ∫ dx / (π r(x)^2)
        # Using known result: ∫_0^{L/2} dx/A(x) = L/(π(r_B - r_A)) * (1/r_A - 1/r_half)
        if self.r_B == self.r_A:
            # Degenerate constant radius case: area constant -> temperature linear.
            T_half_derived = self.T_A + (self.T_B - self.T_A) * 0.5
        else:
            integral_half = self.L / (math.pi * (self.r_B - self.r_A)) * (1.0/self.r_A - 1.0/r_half)
            # From Fourier: T_half - T_A = - Q/K * integral_half
            T_half_derived = self.T_A - (self.Q / self.K) * integral_half

        if self._is_close(self.T_half, T_half_derived):
            return (True, f"Derived T_half = {T_half_derived:g}, reported T_half = {self.T_half:g}.")
        else:
            return (False, f"Derived T_half = {T_half_derived:g} differs from reported T_half = {self.T_half:g}.")

    def test_modeling_temperature_profile_boundary_conditions(self) -> Tuple[bool, str]:
        """
        Use the derived relationship between T(x) and Q to check that
        the reported Q and T_half, together with the givens, produce a
        temperature profile that matches the boundary conditions T(0)=T_A and T(L)=T_B.
        """
        req = [self.L, self.r_A, self.r_B, self.T_A, self.T_B, self.K, self.Q, self.T_half]
        if any(v is None for v in req):
            return (False, "SKIP: Missing required data for boundary condition check.")

        # We'll use the same integration method as for T_half but for full length.
        # If the model is consistent, then using the same logic should give T_B at x=L.
        # For the full length, integral_0^L dx/A(x) = L/(π(r_B - r_A)) * (1/r_A - 1/r_B)
        if self.r_B == self.r_A:
            # Constant radius: linear temperature profile
            T_at_L_from_model = self.T_A + (self.T_B - self.T_A)  # should be T_B
        else:
            integral_full = self.L / (math.pi * (self.r_B - self.r_A)) * (1.0/self.r_A - 1.0/self.r_B)
            # T(x) = T_A - (Q/K) * integral_0^x
            T_at_L_from_model = self.T_A - (self.Q / self.K) * integral_full

        if self._is_close(T_at_L_from_model, self.T_B):
            return (True, f"Boundary condition satisfied: T(L) from model = {T_at_L_from_model:g}, target T_B = {self.T_B:g}.")
        else:
            return (False, f"Boundary condition violation: T(L) from model = {T_at_L_from_model:g} does not match T_B = {self.T_B:g}.")

    # ------------- Monotonicity Tests -------------
    def test_monotonicity_q_with_ra(self) -> Tuple[bool, str]:
        """
        Check that Q increases when r_A increases (proportional product).
        Perturb r_A by +10% and recompute Q from physics.
        """
        req = [self.L, self.r_A, self.r_B, self.T_A, self.T_B, self.K, self.Q]
        if any(v is None for v in req):
            return (False, "SKIP: Missing givens or Q for monotonicity test.")

        # Original Q derived from givens
        Q_orig = math.pi * self.r_A * self.r_B * self.K * (self.T_A - self.T_B) / self.L

        # Perturb r_A by +10%
        r_A_pert = self.r_A * 1.10
        Q_pert = math.pi * r_A_pert * self.r_B * self.K * (self.T_A - self.T_B) / self.L

        if Q_pert > Q_orig:
            return (True, "Monotonicity OK: Q increases with r_A.")
        else:
            return (False, f"Q decreased when r_A increased: original Q = {Q_orig:g}, perturbed Q = {Q_pert:g}.")

    def test_monotonicity_q_with_delta_T(self) -> Tuple[bool, str]:
        """
        Check that |Q| increases when |T_A - T_B| increases.
        Perturb T_B by +5°C (increase difference if T_A > T_B).
        """
        req = [self.L, self.r_A, self.r_B, self.T_A, self.T_B, self.K, self.Q]
        if any(v is None for v in req):
            return (False, "SKIP: Missing givens or Q for monotonicity test.")

        # Original magnitude
        Q_orig_mag = abs(self.Q)
        # Perturb T_B: increase by 5°C (make difference larger if T_A > T_B)
        T_B_pert = self.T_B + 5.0
        Q_pert_mag = abs(math.pi * self.r_A * self.r_B * self.K * (self.T_A - T_B_pert) / self.L)

        if Q_pert_mag >= Q_orig_mag:
            return (True, "Monotonicity OK: |Q| increases with |T_A - T_B|.")
        else:
            return (False, f"|Q| decreased when |ΔT| increased: orig |Q| = {Q_orig_mag:g}, pert |Q| = {Q_pert_mag:g}.")

    def test_monotonicity_t_half_with_T_B(self) -> Tuple[bool, str]:
        """
        Check that T_half increases when T_B increases (since T_half is between T_A and T_B).
        """
        req = [self.r_A, self.r_B, self.T_A, self.T_B, self.L, self.K, self.Q, self.T_half]
        if any(v is None for v in req):
            return (False, "SKIP: Missing data for T_half monotonicity.")

        # Derive T_half from (r_A, r_B, T_A, T_B, L, K, Q)
        r_half = (self.r_A + self.r_B) / 2.0
        if self.r_B == self.r_A:
            T_half_orig = self.T_A + (self.T_B - self.T_A) * 0.5
        else:
            integral_half = self.L / (math.pi * (self.r_B - self.r_A)) * (1.0/self.r_A - 1.0/r_half)
            T_half_orig = self.T_A - (self.Q / self.K) * integral_half

        # Perturb T_B by +10°C
        T_B_pert = self.T_B + 10.0
        # Recompute Q with new T_B
        Q_pert = math.pi * self.r_A * self.r_B * self.K * (self.T_A - T_B_pert) / self.L
        # Recompute T_half
        if self.r_B == self.r_A:
            T_half_pert = self.T_A + (T_B_pert - self.T_A) * 0.5
        else:
            integral_half = self.L / (math.pi * (self.r_B - self.r_A)) * (1.0/self.r_A - 1.0/r_half)
            T_half_pert = self.T_A - (Q_pert / self.K) * integral_half

        if T_half_pert > T_half_orig:
            return (True, "Monotonicity OK: T_half increases with T_B.")
        else:
            return (False, f"T_half decreased when T_B increased: orig T_half = {T_half_orig:g}, perturbed T_half = {T_half_pert:g}.")

    # ------------- Weak Answer Sanity Tests -------------
    def test_answer_sanity_finite(self) -> Tuple[bool, str]:
        """Weak sanity: Q and T_half should be finite (not NaN, not infinite)."""
        results = [self.Q, self.T_half]
        for name, val in zip(['Q', 'T_half'], results):
            if val is not None and (math.isnan(val) or math.isinf(val)):
                return (False, f"{name} is not finite: {val}.")
        return (True, "All available results are finite.")

    # ------------- Runner -------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
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