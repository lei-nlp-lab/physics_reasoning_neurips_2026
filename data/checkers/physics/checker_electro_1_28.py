import math
import sympy as sp
import traceback

class PhysicsChecker:
    """
    Audits a solution for the Child-Langmuir space-charge limited current density
    between two plane-parallel electrodes.
    """
    def __init__(self, solution_instance):
        # safe extraction of givens
        givens = getattr(solution_instance, 'givens', {})
        self.d = givens.get('d', None)
        self.V0 = givens.get('V0', None)
        self.epsilon0 = givens.get('epsilon0', None)
        self.e = givens.get('e', None)
        self.m = givens.get('m', None)

        # safe extraction of results
        results = getattr(solution_instance, 'results', {})
        self.J = results.get('J', None)          # magnitude of current density
        self.j = results.get('j', None)          # vector current density

        # store the full solution instance for potential other uses
        self.sol = solution_instance

    # ------------------------------------------------------------------------
    #  interface tests
    # ------------------------------------------------------------------------
    def test_interface_required_keys_present(self):
        """
        Check that all required givens and output keys exist.
        """
        missing = []
        if self.d is None:
            missing.append('d')
        if self.V0 is None:
            missing.append('V0')
        if self.epsilon0 is None:
            missing.append('epsilon0')
        if self.e is None:
            missing.append('e')
        if self.m is None:
            missing.append('m')
        if self.J is None:
            missing.append('J')
        if self.j is None:
            missing.append('j')
        if missing:
            return (False, f"Missing keys: {', '.join(missing)}")
        return (True, "All required keys present.")

    def test_interface_numeric_types(self):
        """
        Verify that numeric givens and results are real numbers (int/float).
        SymPy expressions are allowed but must be convertible.
        """
        items = [('d', self.d), ('V0', self.V0), ('epsilon0', self.epsilon0),
                 ('e', self.e), ('m', self.m), ('J', self.J)]
        for name, val in items:
            if val is None:
                continue
            try:
                float(val)
            except (ValueError, TypeError):
                return (False, f"{name} is not numeric: {val}")
        return (True, "All required numeric fields are numeric.")

    # ------------------------------------------------------------------------
    #  invariant / theorem-based tests (use math.isclose)
    # ------------------------------------------------------------------------
    def test_invariant_child_langmuir_law(self):
        """
        Verify that the magnitude J obeys Child-Langmuir law:
        J = (4*ε0/9) * sqrt(2e/m) * V0**(3/2) / d**2.
        """
        if any(v is None for v in [self.d, self.V0, self.epsilon0, self.e, self.m, self.J]):
            return (False, "SKIP: missing required givens or J")

        try:
            d = float(self.d)
            V0 = float(self.V0)
            eps0 = float(self.epsilon0)
            e = float(self.e)
            m = float(self.m)
            J = float(self.J)
        except (ValueError, TypeError) as exc:
            return (False, f"Cannot convert to float: {exc}")

        # compute expected J
        expected_J = (4 * eps0 / 9.0) * math.sqrt(2.0 * e / m) * (V0 ** 1.5) / (d ** 2)

        if math.isclose(J, expected_J, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"J={J} is consistent with Child-Langmuir law (expected {expected_J:.6e}).")
        else:
            return (False, f"J={J} deviates from Child-Langmuir law (expected {expected_J:.6e}).")

    def test_invariant_poisson_energy_consistency(self):
        """
        Weak sanity: Using the derived relation J = (4ε0/9) sqrt(2e/m) V0^(3/2)/d²,
        we can check that V(d) = V0 when integrating from 0. This is a derived
        consistency check; we recompute V0 from the given J.
        """
        if any(v is None for v in [self.d, self.epsilon0, self.e, self.m, self.J]):
            return (False, "SKIP: missing required givens or J")

        try:
            d = float(self.d)
            eps0 = float(self.epsilon0)
            e = float(self.e)
            m = float(self.m)
            J = float(self.J)
        except (ValueError, TypeError) as exc:
            return (False, f"Cannot convert to float: {exc}")

        # From the derivation: (4/3) V0^(3/4) = 2 * sqrt( J / (eps0 * sqrt(2e/m)) ) * d
        A = J / (eps0 * math.sqrt(2.0 * e / m))
        # dV/dx at x=0 is 0 -> integration constant zero.
        # V0 should satisfy: V0^(3/4) = (3/2) * sqrt(A) * d
        derived_V0 = ((3.0/2.0) * math.sqrt(A) * d) ** (4.0/3.0)

        if math.isclose(derived_V0, float(self.V0), rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Consistency check passed: derived V0 = {derived_V0:.6e} matches given.")
        else:
            return (False, f"Consistency check failed: derived V0 = {derived_V0:.6e} does not match given {self.V0}.")

    def test_invariant_current_direction(self):
        """
        Check that the vector j has the correct sign: electrons move from lower to
        higher potential (x-direction), so j_x should be negative (current flows
        opposite to electron motion).  J is positive magnitude.
        """
        if self.j is None or self.J is None:
            return (False, "SKIP: j or J missing")

        try:
            J_mag = float(self.J)
            # j may be given as a sympy vector or a string "-J e_x"
            # We try to extract x-component if possible.
            # For simplicity, we just check that j_x is negative if J>0.
            if hasattr(self.j, '__getitem__'):   # list/tuple/Vector
                j_x = float(self.j[0])
            elif isinstance(self.j, sp.Basic):   # sympy expression
                # Try to get the coefficient of a unit vector e_x
                # simpler: evaluate at x=1 and extract numeric part
                # But we only need sign: assume the expression is -J * e_x
                # Evaluate at some point? We'll just check that j is negative scalar if J>0.
                j_eval = complex(self.j.subs({sp.Symbol('e_x'): 1}).evalf())
                j_x = j_eval.real
            else:
                # fallback: try float conversion (may fail)
                j_x = float(self.j)

            if j_x < 0 and J_mag > 0:
                return (True, f"j_x = {j_x} is negative as expected (electrons flow opposite to current).")
            else:
                return (False, f"j_x = {j_x} should be negative for J={J_mag}.")
        except Exception as exc:
            return (False, f"Cannot determine direction: {exc}")

    # ------------------------------------------------------------------------
    #  monotonicity tests (using perturbation)
    # ------------------------------------------------------------------------
    def test_monotonicity_current_increases_with_voltage(self):
        """
        When V0 increases by 10%, J should increase (by factor ~ (1.1)^(3/2) ≈ 1.153).
        We derive perturbed J using Child-Langmuir law.
        """
        if any(v is None for v in [self.d, self.V0, self.epsilon0, self.e, self.m, self.J]):
            return (False, "SKIP: missing required givens or J")

        try:
            d = float(self.d)
            V0 = float(self.V0)
            eps0 = float(self.epsilon0)
            e = float(self.e)
            m = float(self.m)
        except (ValueError, TypeError) as exc:
            return (False, f"Cannot convert to float: {exc}")

        # perturb V0 by +10%
        V0_perturbed = V0 * 1.10
        expected_J_pert = (4 * eps0 / 9.0) * math.sqrt(2.0 * e / m) * (V0_perturbed ** 1.5) / (d ** 2)

        # baseline J from given
        J_baseline = float(self.J)
        # predicted perturbed J from Child-Langmuir
        J_pert_predicted = J_baseline * (1.10 ** 1.5)  # relative scaling

        # Check that J_pert_predicted matches expected_J_pert within tolerance
        if math.isclose(expected_J_pert, J_pert_predicted, rel_tol=1e-4):
            return (True, f"Monotonicity check: J increases by factor {1.10**1.5:.4f} when V0 increased by 10%.")
        else:
            return (False, f"Expected J_pert={expected_J_pert:.6e}, but predicted J_pert={J_pert_predicted:.6e}.")

    def test_monotonicity_current_decreases_with_distance(self):
        """
        When d increases by 10%, J should decrease by factor 1/(1.1^2) ≈ 0.826.
        """
        if any(v is None for v in [self.d, self.V0, self.epsilon0, self.e, self.m, self.J]):
            return (False, "SKIP: missing required givens or J")

        try:
            d = float(self.d)
            V0 = float(self.V0)
            eps0 = float(self.epsilon0)
            e = float(self.e)
            m = float(self.m)
        except (ValueError, TypeError) as exc:
            return (False, f"Cannot convert to float: {exc}")

        d_perturbed = d * 1.10
        expected_J_pert = (4 * eps0 / 9.0) * math.sqrt(2.0 * e / m) * (V0 ** 1.5) / (d_perturbed ** 2)

        J_baseline = float(self.J)
        J_pert_predicted = J_baseline / (1.10 ** 2)

        if math.isclose(expected_J_pert, J_pert_predicted, rel_tol=1e-4):
            return (True, f"Monotonicity check: J decreases by factor {1/(1.1**2):.4f} when d increased by 10%.")
        else:
            return (False, f"Expected J_pert={expected_J_pert:.6e}, but predicted J_pert={J_pert_predicted:.6e}.")

    # ------------------------------------------------------------------------
    #  modeling/trace quality tests
    # ------------------------------------------------------------------------
    def test_modeling_poisson_equation_used(self):
        """
        Check that the explanation steps include Poisson's equation, energy conservation,
        and the space-charge boundary condition.
        """
        steps = getattr(self.sol, 'explanation_steps', None)
        if steps is None or len(steps) == 0:
            return (False, "SKIP: no explanation_steps available to analyze modeling quality.")

        all_text = " ".join(str(s) for s in steps).lower()

        checks = {
            "poisson equation": "poisson" in all_text,
            "energy conservation": "energy" in all_text and ("½" in all_text or "1/2" in all_text),
            "dv/dx(0)=0": "dv/dx" in all_text and "0" in all_text,
        }
        missing = [k for k, v in checks.items() if not v]
        if missing:
            return (False, f"Missing key modeling elements: {', '.join(missing)}")
        return (True, "Explanation steps include Poisson, energy conservation, and space-charge boundary condition.")

    def test_modeling_current_continuity(self):
        """
        Check that the explanation steps note that current density is constant (dJ/dx=0).
        """
        steps = getattr(self.sol, 'explanation_steps', None)
        if steps is None or len(steps) == 0:
            return (False, "SKIP: no explanation_steps available.")

        all_text = " ".join(str(s) for s in steps).lower()

        # look for keywords indicating constant current
        has_steady = "steady" in all_text and "state" in all_text
        has_dj_zero = "dj" in all_text and "dx" in all_text and "0" in all_text
        has_j_constant = "j" in all_text and "constant" in all_text

        if has_steady or has_dj_zero or has_j_constant:
            return (True, "Explanation steps acknowledge constant current density (dJ/dx=0).")
        else:
            return (False, "Missing explicit mention of steady-state constant current density.")

    def test_trace_boundary_conditions(self):
        """
        Check that the explanation steps mention the three boundary conditions:
        V(0)=0, V(d)=V0, and dV/dx(0)=0.
        """
        steps = getattr(self.sol, 'explanation_steps', None)
        if steps is None or len(steps) == 0:
            return (False, "SKIP: no explanation_steps available.")

        all_text = " ".join(str(s) for s in steps).lower()

        checks = {
            "V(0)=0": ("v(0)=0" in all_text) or ("v=0" in all_text and "x=0" in all_text),
            "V(d)=V0": ("v(d)=v0" in all_text) or ("v=v0" in all_text and "x=d" in all_text),
            "dV/dx(0)=0": ("dv/dx(0)=0" in all_text) or ("dv/dx=0" in all_text and "x=0" in all_text),
        }
        missing = [k for k, v in checks.items() if not v]
        if missing:
            return (False, f"Missing boundary condition details: {', '.join(missing)}")
        return (True, "All three boundary conditions (V(0)=0, V(d)=V0, dV/dx(0)=0) are referenced in steps.")

    # ------------------------------------------------------------------------
    #  weak final-answer sanity (no isclose, no re-derivation)
    # ------------------------------------------------------------------------
    def test_answer_sanity_basic(self):
        """
        Weak sanity: J magnitude should be positive and finite.
        """
        if self.J is None:
            return (False, "SKIP: J missing")
        try:
            J_mag = float(self.J)
        except (ValueError, TypeError):
            return (False, "J cannot be converted to float")

        if math.isfinite(J_mag) and J_mag > 0:
            return (True, f"J = {J_mag:.6e} is positive and finite.")
        else:
            return (False, f"J = {J_mag} is not positive finite.")

    # ------------------------------------------------------------------------
    #  test runner
    # ------------------------------------------------------------------------
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
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)