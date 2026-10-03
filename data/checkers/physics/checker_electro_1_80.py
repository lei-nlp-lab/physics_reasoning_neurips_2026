import math
import sympy as sp
import traceback

class PhysicsChecker:
    """Audits solutions for electrostatic potential due to a point charge near a grounded conducting sphere."""

    def __init__(self, solution_instance):
        # Extract results
        self.results = getattr(solution_instance, 'results', {})
        self.givens = getattr(solution_instance, 'givens', {})
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', {})

        # Safely extract required results
        self.V_vac = self.results.get('V_vac', None)
        self.V_diel = self.results.get('V_diel', None)

        # Safely extract given parameters
        self.a = None
        self.s = None
        self.q = None
        self.epsilon0 = None
        self.epsilon = None

        givens_dict = self.givens
        if isinstance(givens_dict, dict):
            self.a = givens_dict.get('a', None)
            self.s = givens_dict.get('s', None)
            self.q = givens_dict.get('q', None)
            self.epsilon0 = givens_dict.get('epsilon0', None)
            self.epsilon = givens_dict.get('epsilon', None)

        # Fallback to direct attribute if still None
        if self.a is None:
            self.a = getattr(solution_instance, 'a', None)
        if self.s is None:
            self.s = getattr(solution_instance, 's', None)
        if self.q is None:
            self.q = getattr(solution_instance, 'q', None)
        if self.epsilon0 is None:
            self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        if self.epsilon is None:
            self.epsilon = getattr(solution_instance, 'epsilon', None)

        # Convert s to sympy vector if possible
        self._s_vec = None
        if self.s is not None:
            try:
                self._s_vec = sp.Matrix(self.s) if not isinstance(self.s, sp.Matrix) else self.s
            except Exception:
                self._s_vec = None

        # Precompute s_norm (distance of charge from center)
        self._s_norm = None
        if self._s_vec is not None:
            try:
                self._s_norm = sp.sqrt(self._s_vec.dot(self._s_vec))
            except Exception:
                self._s_norm = None

    # ----------------------------------------------------------------------
    # Helper functions
    # ----------------------------------------------------------------------
    def _to_vector(self, v):
        """Convert to sympy Matrix (column vector)."""
        if isinstance(v, sp.Matrix):
            return v
        try:
            return sp.Matrix(v)
        except Exception:
            return None

    def _eval_potential(self, V, r_point):
        """
        Evaluate potential at a given point r_point (supporting callable or sympy expression).
        Returns None if evaluation fails.
        """
        if V is None:
            return None
        if callable(V):
            try:
                return V(r_point)
            except Exception:
                return None
        if hasattr(V, 'subs') and hasattr(V, 'free_symbols'):
            r_vec = self._to_vector(r_point)
            if r_vec is None:
                return None
            free_syms = V.free_symbols
            subs = {}
            for sym in free_syms:
                name = str(sym)
                if name == 'r_x' and r_vec.shape[0] >= 1:
                    subs[sym] = r_vec[0]
                elif name == 'r_y' and r_vec.shape[0] >= 2:
                    subs[sym] = r_vec[1]
                elif name == 'r_z' and r_vec.shape[0] >= 3:
                    subs[sym] = r_vec[2]
            givens_map = {'a': self.a, 'q': self.q, 'epsilon0': self.epsilon0, 'epsilon': self.epsilon}
            for name, val in givens_map.items():
                if val is not None:
                    sym = sp.Symbol(name)
                    if sym in free_syms:
                        subs[sym] = val
            if self._s_vec is not None:
                for i, sym_name in enumerate(['s_x', 's_y', 's_z']):
                    sym = sp.Symbol(sym_name)
                    if sym in free_syms and i < self._s_vec.shape[0]:
                        subs[sym] = self._s_vec[i]
            try:
                val = V.subs(subs).evalf()
                return float(val)
            except Exception:
                return None
        return None

    def _compute_expected_potential(self, r_point, epsilon_used):
        """
        Compute V_expected using the method of images formula:
        V = (q/(4π ε)) * (1/|r-s| - (a/s)/|r - (a^2/s^2) s|)
        """
        if any(x is None for x in [self.a, self._s_vec, self._s_norm, self.q, epsilon_used]):
            return None
        try:
            r_vec = self._to_vector(r_point)
            if r_vec is None:
                return None
            diff1 = r_vec - self._s_vec
            norm1 = sp.sqrt(diff1.dot(diff1))
            s_prime = (self.a**2 / (self._s_norm**2)) * self._s_vec
            diff2 = r_vec - s_prime
            norm2 = sp.sqrt(diff2.dot(diff2))
            val = (self.q / (4 * sp.pi * epsilon_used)) * (1/norm1 - (self.a/self._s_norm)/norm2)
            return float(val.evalf())
        except Exception:
            return None

    def _point_on_sphere(self):
        """Return a point on the sphere along the direction of s."""
        if self._s_vec is None or self._s_norm is None or self.a is None:
            return None
        try:
            unit = self._s_vec / self._s_norm
            return self.a * unit
        except Exception:
            return None

    def _point_far_away(self):
        """Return a point far from the sphere (100*a along s direction)."""
        if self._s_vec is None or self._s_norm is None or self.a is None:
            return None
        try:
            unit = self._s_vec / self._s_norm
            return 100 * self.a * unit
        except Exception:
            return None

    def _point_near_charge(self, offset_scale=0.01):
        """Return a point near the charge (s + small offset perpendicular)."""
        if self._s_vec is None or self.a is None:
            return None
        try:
            s_vec = self._s_vec
            if s_vec.shape[0] == 3:
                if abs(s_vec[2]) < 0.9:
                    perp = sp.Matrix([0,0,1]).cross(s_vec)
                else:
                    perp = sp.Matrix([1,0,0]).cross(s_vec)
            elif s_vec.shape[0] == 2:
                perp = sp.Matrix([-s_vec[1], s_vec[0]])
            else:
                perp = sp.Matrix([1])
            perp = perp / sp.sqrt(perp.dot(perp)) * self.a * offset_scale
            return s_vec + perp
        except Exception:
            return None

    # ----------------------------------------------------------------------
    # Test methods
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that required result keys exist."""
        if self.V_vac is None:
            return (False, "Missing required key 'V_vac'")
        if self.V_diel is None:
            return (False, "Missing required key 'V_diel'")
        return (True, "Both V_vac and V_diel are present")

    def test_interface_givens_available(self):
        """Check that essential givens are available."""
        missing = []
        for name, val in [('a', self.a), ('s', self.s), ('q', self.q), ('epsilon0', self.epsilon0), ('epsilon', self.epsilon)]:
            if val is None:
                missing.append(name)
        if missing:
            return (False, f"Missing given(s): {', '.join(missing)}")
        return (True, "All essential givens present")

    def test_invariant_boundary_condition_vacuum(self):
        """V_vac must be zero on the sphere surface."""
        if self.V_vac is None:
            return (False, "SKIP: V_vac not available")
        r_test = self._point_on_sphere()
        if r_test is None:
            return (False, "SKIP: could not compute point on sphere")
        val = self._eval_potential(self.V_vac, r_test)
        if val is None:
            return (False, "Failed to evaluate V_vac on sphere")
        if math.isclose(val, 0.0, abs_tol=1e-3):
            return (True, f"V_vac on sphere = {val:.4e}, close to 0")
        else:
            return (False, f"V_vac on sphere = {val:.4e}, expected 0")

    def test_invariant_boundary_condition_dielectric(self):
        """V_diel must be zero on the sphere surface."""
        if self.V_diel is None:
            return (False, "SKIP: V_diel not available")
        r_test = self._point_on_sphere()
        if r_test is None:
            return (False, "SKIP: could not compute point on sphere")
        val = self._eval_potential(self.V_diel, r_test)
        if val is None:
            return (False, "Failed to evaluate V_diel on sphere")
        if math.isclose(val, 0.0, abs_tol=1e-3):
            return (True, f"V_diel on sphere = {val:.4e}, close to 0")
        else:
            return (False, f"V_diel on sphere = {val:.4e}, expected 0")

    def test_invariant_infinity_vacuum(self):
        """V_vac should approach 0 at large distance."""
        if self.V_vac is None:
            return (False, "SKIP: V_vac not available")
        r_test = self._point_far_away()
        if r_test is None:
            return (False, "SKIP: could not compute far point")
        val = self._eval_potential(self.V_vac, r_test)
        if val is None:
            return (False, "Failed to evaluate V_vac at large distance")
        if math.isclose(val, 0.0, abs_tol=1e-6):
            return (True, f"V_vac at large distance = {val:.4e}, close to 0")
        else:
            return (False, f"V_vac at large distance = {val:.4e}, expected ~0")

    def test_invariant_infinity_dielectric(self):
        """V_diel should approach 0 at large distance."""
        if self.V_diel is None:
            return (False, "SKIP: V_diel not available")
        r_test = self._point_far_away()
        if r_test is None:
            return (False, "SKIP: could not compute far point")
        val = self._eval_potential(self.V_diel, r_test)
        if val is None:
            return (False, "Failed to evaluate V_diel at large distance")
        if math.isclose(val, 0.0, abs_tol=1e-6):
            return (True, f"V_diel at large distance = {val:.4e}, close to 0")
        else:
            return (False, f"V_diel at large distance = {val:.4e}, expected ~0")

    def test_modeling_vacuum_vs_dielectric(self):
        """V_diel should equal (epsilon0/epsilon)*V_vac for same observation point."""
        if any(x is None for x in [self.V_vac, self.V_diel, self.epsilon0, self.epsilon]):
            return (False, "SKIP: missing required data")
        r_test = self._point_near_charge(offset_scale=0.5)
        if r_test is None:
            return (False, "SKIP: could not compute test point")
        val_vac = self._eval_potential(self.V_vac, r_test)
        val_diel = self._eval_potential(self.V_diel, r_test)
        if val_vac is None or val_diel is None:
            return (False, "Failed to evaluate potentials")
        if val_diel == 0:
            return (False, "V_diel is zero, cannot compute ratio")
        expected_ratio = self.epsilon / self.epsilon0 if self.epsilon0 != 0 else None
        if expected_ratio is None:
            return (False, "epsilon0 is zero")
        actual_ratio = val_vac / val_diel
        if math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2):
            return (True, f"V_vac/V_diel ratio = {actual_ratio:.4f}, expected {expected_ratio:.4f}")
        else:
            return (False, f"V_vac/V_diel ratio = {actual_ratio:.4f}, expected {expected_ratio:.4f}")

    def test_modeling_image_formula_vacuum(self):
        """V_vac matches the exact method-of-images formula at a test point."""
        if any(x is None for x in [self.V_vac, self.a, self._s_vec, self._s_norm, self.q, self.epsilon0]):
            return (False, "SKIP: missing required data for formula")
        if self._s_vec is None or self._s_norm is None or self.a is None:
            return (False, "SKIP: cannot construct direction")
        unit = self._s_vec / self._s_norm
        r_test = 2.0 * self.a * unit
        val_actual = self._eval_potential(self.V_vac, r_test)
        if val_actual is None:
            return (False, "Failed to evaluate V_vac")
        val_expected = self._compute_expected_potential(r_test, self.epsilon0)
        if val_expected is None:
            return (False, "Failed to compute expected potential")
        if math.isclose(val_actual, val_expected, rel_tol=1e-2):
            return (True, f"V_vac = {val_actual:.4e}, expected {val_expected:.4e}")
        else:
            return (False, f"V_vac = {val_actual:.4e}, expected {val_expected:.4e}")

    def test_modeling_image_formula_dielectric(self):
        """V_diel matches the exact method-of-images formula with dielectric."""
        if any(x is None for x in [self.V_diel, self.a, self._s_vec, self._s_norm, self.q, self.epsilon]):
            return (False, "SKIP: missing required data for formula")
        if self._s_vec is None or self._s_norm is None or self.a is None:
            return (False, "SKIP: cannot construct direction")
        unit = self._s_vec / self._s_norm
        r_test = 2.0 * self.a * unit
        val_actual = self._eval_potential(self.V_diel, r_test)
        if val_actual is None:
            return (False, "Failed to evaluate V_diel")
        val_expected = self._compute_expected_potential(r_test, self.epsilon)
        if val_expected is None:
            return (False, "Failed to compute expected potential")
        if math.isclose(val_actual, val_expected, rel_tol=1e-2):
            return (True, f"V_diel = {val_actual:.4e}, expected {val_expected:.4e}")
        else:
            return (False, f"V_diel = {val_actual:.4e}, expected {val_expected:.4e}")

    def test_monotonicity_s_increase(self):
        """Potential magnitude at a far point should decrease when s increases."""
        if any(x is None for x in [self.a, self._s_vec, self._s_norm, self.q, self.epsilon0]):
            return (False, "SKIP: missing required data")
        if self._s_vec is None or self._s_norm is None or self.a is None:
            return (False, "SKIP: cannot construct direction")
        unit = self._s_vec / self._s_norm
        r_test = 10.0 * self.a * unit  # observation point far away

        # Save originals
        old_s_vec = self._s_vec
        old_s_norm = self._s_norm

        # Baseline with original s
        V_base = self._compute_expected_potential(r_test, self.epsilon0)

        # Perturb s by +10%
        self._s_vec = 1.1 * old_s_vec
        self._s_norm = 1.1 * old_s_norm
        V_pert = self._compute_expected_potential(r_test, self.epsilon0)

        # Restore originals
        self._s_vec = old_s_vec
        self._s_norm = old_s_norm

        if V_base is None or V_pert is None:
            return (False, "SKIP: could not compute potentials")

        if abs(V_pert) < abs(V_base) - 1e-12:
            return (True, f"|V_pert|={abs(V_pert):.4e} < |V_base|={abs(V_base):.4e}, monotonicity holds")
        else:
            return (False, f"|V_pert|={abs(V_pert):.4e} >= |V_base|={abs(V_base):.4e}, monotonicity fails")

    def test_monotonicity_epsilon_increase(self):
        """Potential magnitude at a fixed point should decrease when epsilon increases (same s)."""
        if any(x is None for x in [self.a, self._s_vec, self._s_norm, self.q, self.epsilon0, self.epsilon]):
            return (False, "SKIP: missing required data")
        if self._s_vec is None or self._s_norm is None or self.a is None or self.epsilon == 0:
            return (False, "SKIP: cannot compute (epsilon zero?)")
        # Use a test point
        unit = self._s_vec / self._s_norm
        r_test = 5.0 * self.a * unit

        # Baseline with epsilon0
        V_base = self._compute_expected_potential(r_test, self.epsilon0)
        # Perturb epsilon: use 10 * epsilon0 (larger)
        epsilon_large = 10.0 * self.epsilon0
        V_pert = self._compute_expected_potential(r_test, epsilon_large)

        if V_base is None or V_pert is None:
            return (False, "SKIP: could not compute potentials")

        if abs(V_pert) < abs(V_base) - 1e-12:
            return (True, f"|V_pert|={abs(V_pert):.4e} < |V_base|={abs(V_base):.4e} (epsilon increased), monotonicity holds")
        else:
            return (False, f"|V_pert|={abs(V_pert):.4e} >= |V_base|={abs(V_base):.4e}, monotonicity fails")

    def test_answer_sanity_finite_sign(self):
        """Both potentials should be finite and have consistent sign at a test point."""
        if any(x is None for x in [self.V_vac, self.V_diel, self.q]):
            return (False, "SKIP: missing required data")
        r_test = self._point_near_charge(offset_scale=2.0)
        if r_test is None:
            return (False, "SKIP: cannot get test point")
        v_vac = self._eval_potential(self.V_vac, r_test)
        v_diel = self._eval_potential(self.V_diel, r_test)
        if v_vac is None or v_diel is None:
            return (False, "Failed to evaluate one or both potentials")
        if not math.isfinite(v_vac) or not math.isfinite(v_diel):
            return (False, f"Non-finite values: V_vac={v_vac}, V_diel={v_diel}")
        # Verify they have same sign (or both zero)
        if v_vac * v_diel < 0:
            return (False, f"V_vac ({v_vac}) and V_diel ({v_diel}) have opposite signs")
        return (True, f"Both potentials finite and sign consistent: V_vac={v_vac:.4e}, V_diel={v_diel:.4e}")

    # ----------------------------------------------------------------------
    # Test runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)