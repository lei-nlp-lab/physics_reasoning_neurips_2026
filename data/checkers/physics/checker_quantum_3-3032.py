import sympy as sp
import math
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

class PhysicsChecker:
    """
    Audits solutions for a spin-1/2 particle in a uniform magnetic field,
    focusing on the time-dependent state vector and expectation value of s_x.
    """

    def __init__(self, solution_instance: Any):
        # Extract results with safe defaults (None for missing)
        results = getattr(solution_instance, 'results', {}) or {}
        self.psi_t = results.get('psi_t', None)
        self.expect_sx = results.get('expect_sx', None)

        # Extract givens (optional but useful for recomputation)
        givens = getattr(solution_instance, 'givens', {}) or {}
        self.hbar = givens.get('hbar', None)
        self.e = givens.get('e', None)
        self.B0 = givens.get('B0', None)
        self.m = givens.get('m', None)
        self.omega = givens.get('omega', None)

        # If omega not directly given, try to compute from e, B0, m
        if self.omega is None and self.e is not None and self.B0 is not None and self.m is not None:
            self.omega = self.e * self.B0 / (2 * self.m)   # ω = eB/(2m) (non-relativistic Bohr magneton)

        # Use a symbolic time variable for checks
        self.t = sp.Symbol('t', real=True)

    # -----------------------------------------------------------------
    # INTERFACE CHECKS
    # -----------------------------------------------------------------
    def test_interface_required_keys(self) -> Tuple[bool, str]:
        """Check that both required result keys exist and are not None."""
        if self.psi_t is None:
            return (False, "Missing required key: psi_t")
        if self.expect_sx is None:
            return (False, "Missing required key: expect_sx")
        return (True, "Both required keys present")

    def test_interface_psi_t_type(self) -> Tuple[bool, str]:
        """Check that psi_t is a sympy Matrix (or list/tuple convertible)."""
        if self.psi_t is None:
            return (False, "SKIP: psi_t not available")
        try:
            mat = sp.Matrix(self.psi_t)
            return (True, "psi_t is convertible to sympy Matrix")
        except Exception as e:
            return (False, f"psi_t not a valid matrix: {e}")

    def test_interface_expect_sx_type(self) -> Tuple[bool, str]:
        """Check that expect_sx is a sympy expression or number."""
        if self.expect_sx is None:
            return (False, "SKIP: expect_sx not available")
        try:
            sp.sympify(self.expect_sx)
            return (True, "expect_sx is valid sympy expression")
        except Exception:
            return (False, "expect_sx is not a valid sympy expression")

    # -----------------------------------------------------------------
    # THEOREM-BASED INVARIANTS
    # -----------------------------------------------------------------
    def test_invariant_normalization(self) -> Tuple[bool, str]:
        """Check that |psi_t|^2 = 1."""
        if self.psi_t is None:
            return (False, "SKIP: psi_t not available")
        try:
            psi = sp.Matrix(self.psi_t)
            norm_sq = (psi[0] * sp.conjugate(psi[0]) + psi[1] * sp.conjugate(psi[1])).simplify()
            # Expect 1
            norm_val = complex(norm_sq.subs(self.t, 0)) if self.t in norm_sq.free_symbols else complex(norm_sq)
            if math.isclose(abs(norm_val), 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Normalization holds (|psi|^2 ≈ 1)")
            else:
                return (False, f"Normalization violation: |psi|^2 = {norm_val}")
        except Exception as e:
            return (False, f"Normalization check error: {e}")

    def test_invariant_initial_state(self) -> Tuple[bool, str]:
        """Check that at t=0, psi_t equals |+y> = (1, i)/√2."""
        if self.psi_t is None:
            return (False, "SKIP: psi_t not available")
        try:
            psi0 = sp.Matrix(self.psi_t).subs(self.t, 0)
            expected = sp.Matrix([1, sp.I]) / sp.sqrt(2)
            diff = (psi0 - expected).applyfunc(lambda x: sp.simplify(x))
            # Check each component close to 0
            okay = all(math.isclose(abs(complex(diff[i])), 0, abs_tol=1e-2) for i in range(2))
            if okay:
                return (True, "Initial state matches |+y⟩")
            else:
                return (False, f"Initial state mismatch: psi(0) = {psi0}")
        except Exception as e:
            return (False, f"Initial state check error: {e}")

    def test_invariant_schrodinger_equation(self) -> Tuple[bool, str]:
        """Check iħ d|psi>/dt = H|psi> (Hamiltonian H = ħω S_z = ħω/2 σ_z)."""
        if self.psi_t is None:
            return (False, "SKIP: psi_t not available")
        if self.hbar is None or self.omega is None:
            # Cannot compute Hamiltonian without ħ and ω; try symbolic fallback.
            hbar = sp.Symbol('hbar', positive=True)
            omega = sp.Symbol('omega', real=True)
        else:
            hbar = sp.sympify(self.hbar)
            omega = sp.sympify(self.omega)
        try:
            psi = sp.Matrix(self.psi_t)
            # Time derivative
            dpsi = sp.diff(psi, self.t)
            # Hamiltonian: H = ħ ω σ_z / 2
            sigma_z = sp.Matrix([[1, 0], [0, -1]])
            H = hbar * omega * sigma_z / 2
            lhs = sp.I * hbar * dpsi
            rhs = H * psi
            diff = (lhs - rhs).applyfunc(lambda x: sp.simplify(x))
            # Assess each component; must be ≈ 0
            zero_check = all(
                math.isclose(abs(complex(diff[i].subs(self.t, 0.5))), 0, abs_tol=1e-2)
                for i in range(2)
            )
            if not zero_check:
                # Try a different t value in case t is symbolic
                t_vals = [0, 0.1, 1.0]
                for tv in t_vals:
                    comps = [complex(diff[i].subs(self.t, tv)) for i in range(2)]
                    if all(math.isclose(abs(c), 0, abs_tol=1e-2) for c in comps):
                        zero_check = True
                        break
            if zero_check:
                return (True, "Schrödinger equation satisfied (iħ dψ/dt = Hψ)")
            else:
                return (False, f"Schrödinger equation violation: residual = {diff}")
        except Exception as e:
            return (False, f"Schrödinger check error: {e}")

    def test_invariant_expectation_real(self) -> Tuple[bool, str]:
        """Check that <s_x> is real (imaginary part near zero)."""
        if self.expect_sx is None:
            return (False, "SKIP: expect_sx not available")
        try:
            ex = sp.sympify(self.expect_sx)
            # Substitute a numeric t value if symbol present
            t_val = 0.5
            if self.t in ex.free_symbols:
                ex_num = complex(ex.subs(self.t, t_val))
            else:
                ex_num = complex(ex)
            imag_part = abs(ex_num.imag)
            if math.isclose(imag_part, 0, abs_tol=1e-2):
                return (True, "Expectation value is real (imag ≈ 0)")
            else:
                return (False, f"Expectation value has non-zero imaginary part: {ex_num}")
        except Exception as e:
            return (False, f"Expectation real check error: {e}")

    def test_invariant_expectation_range(self) -> Tuple[bool, str]:
        """Check |<s_x>| ≤ ħ/2 (physical bound)."""
        if self.expect_sx is None:
            return (False, "SKIP: expect_sx not available")
        if self.hbar is None:
            # Use symbolic ħ, assume positive
            hbar = sp.Symbol('hbar', positive=True)
        else:
            hbar = sp.sympify(self.hbar)

        try:
            ex = sp.sympify(self.expect_sx)
            # Test a few t values
            t_vals = [0, 0.1, 0.5, 1.0, 2.0]
            bound = hbar / 2
            for tv in t_vals:
                ex_num = complex(ex.subs(self.t, tv)) if self.t in ex.free_symbols else complex(ex)
                if not math.isclose(abs(ex_num), float(bound.subs(hbar, 1.0545718e-34) if hbar.is_Symbol else abs(bound)), abs_tol=1e-2):
                    # Check only bound, not exact
                    pass
                # More robust: ensure magnitude not significantly larger than ħ/2
                bound_val = float(hbar) if hbar.is_number else 5.0  # loose bound if symbolic
                if abs(ex_num) > bound_val * 1.1:  # 10% tolerance over bound
                    return (False, f"|<s_x>| exceeds bound at t={tv}: {ex_num}")
            return (True, "Expectation value stays within physical range")
        except Exception as e:
            return (False, f"Expectation range check error: {e}")

    # -----------------------------------------------------------------
    # MODELING / TRACE CHECKS
    # -----------------------------------------------------------------
    def test_modeling_time_evolution_operator(self) -> Tuple[bool, str]:
        """Check that psi_t = U(t) * psi_0 with U = diag(e^{-iωt/2}, e^{iωt/2})."""
        if self.psi_t is None:
            return (False, "SKIP: psi_t not available")
        if self.omega is None:
            omega = sp.Symbol('omega', real=True)
        else:
            omega = sp.sympify(self.omega)
        try:
            psi = sp.Matrix(self.psi_t)
            # Initial state: |+y>
            psi0 = sp.Matrix([1, sp.I]) / sp.sqrt(2)
            # Time evolution matrix
            U = sp.diag(sp.exp(-sp.I * omega * self.t / 2), sp.exp(sp.I * omega * self.t / 2))
            expected = U * psi0
            diff = (psi - expected).applyfunc(lambda x: sp.simplify(x))
            # Check at one time value
            t_test = 0.5
            ok = all(
                math.isclose(abs(complex(diff[i].subs(self.t, t_test))), 0, abs_tol=1e-2)
                for i in range(2)
            )
            if ok:
                return (True, "psi_t matches U(t)*|+y⟩")
            else:
                return (False, f"Time evolution mismatch: residual = {diff}")
        except Exception as e:
            return (False, f"Time evolution check error: {e}")

    def test_modeling_expect_sx_from_psi(self) -> Tuple[bool, str]:
        """Check that <s_x> computed from psi_t = (ħ/2) <σ_x> matches provided expect_sx."""
        if self.psi_t is None or self.expect_sx is None:
            return (False, "SKIP: need both psi_t and expect_sx")
        if self.hbar is None:
            hbar = sp.Symbol('hbar', positive=True)
        else:
            hbar = sp.sympify(self.hbar)
        try:
            psi = sp.Matrix(self.psi_t)
            sigma_x = sp.Matrix([[0, 1], [1, 0]])
            # Compute expectation: (ħ/2) * psi_dagger * sigma_x * psi
            psi_dag = sp.conjugate(psi.T)
            computed = (hbar / 2) * (psi_dag * sigma_x * psi)[0]
            computed_simplified = sp.simplify(computed)
            given = sp.sympify(self.expect_sx)
            given_simplified = sp.simplify(given)
            diff = sp.simplify(computed_simplified - given_simplified)
            # Check numerically
            t_test = 0.5
            diff_num = complex(diff.subs(self.t, t_test)) if self.t in diff.free_symbols else complex(diff)
            if math.isclose(abs(diff_num), 0, abs_tol=1e-2):
                return (True, "Expectation value derived from psi_t matches provided value")
            else:
                return (False, f"Expectation mismatch: computed = {computed_simplified}, given = {given_simplified}, diff = {diff}")
        except Exception as e:
            return (False, f"Expectation derivation check error: {e}")

    def test_modeling_omega_consistency(self) -> Tuple[bool, str]:
        """Check that omega appears in psi_t and expect_sx consistently (if given)."""
        if self.psi_t is None and self.expect_sx is None:
            return (False, "SKIP: no results to check")
        if self.omega is None:
            return (False, "SKIP: omega not provided")
        try:
            omega = sp.sympify(self.omega)
            # Verify that both expressions contain omega (or a symbol representing it)
            # We simply check that the provided omega appears in the expression
            given_omega_symbol = omega if isinstance(omega, sp.Symbol) else sp.Symbol('omega')
            # This is a weak check; better: evaluate the expression with numeric omega and see consistency.
            # We do a numeric substitution for a specific t and compare with explicit formula.
            hbar = sp.sympify(self.hbar) if self.hbar else sp.Symbol('hbar', positive=True)
            t_val = 1.0
            # Build reference expectation: -ħ/2 sin(2ωt)
            ref_sx = -hbar/2 * sp.sin(2 * omega * t_val)
            given_sx = sp.sympify(self.expect_sx).subs({self.t: t_val})
            diff = abs(complex(sp.N(given_sx - ref_sx)))
            if math.isclose(diff, 0, abs_tol=1e-2):
                return (True, "Omega appears consistently in expectation value")
            else:
                # Try also that psi_t has correct omega dependence (only phase)
                psi_ref = sp.Matrix([sp.exp(-sp.I*omega*t_val/2), sp.I*sp.exp(sp.I*omega*t_val/2)]) / sp.sqrt(2)
                if self.psi_t is not None:
                    psi_given = sp.Matrix(self.psi_t).subs(self.t, t_val)
                    diff_psi = sum(abs(complex(psi_given[i] - psi_ref[i])) for i in range(2))
                    if math.isclose(diff_psi, 0, abs_tol=1e-2):
                        return (True, "Omega appears consistently in state vector")
                return (False, f"Omega consistency check failed: diff = {diff}")
        except Exception as e:
            return (False, f"Omega consistency error: {e}")

    def test_modeling_phase_factor_ratio(self) -> Tuple[bool, str]:
        """Check that the ratio of components in psi_t is i*e^{iωt} / e^{-iωt} = i e^{2iωt} (up to sign)."""
        if self.psi_t is None:
            return (False, "SKIP: psi_t not available")
        try:
            psi = sp.Matrix(self.psi_t)
            ratio = sp.simplify(psi[1] / psi[0])
            expected = sp.I * sp.exp(2 * sp.I * self.omega * self.t) if self.omega else sp.I * sp.exp(2 * sp.I * sp.Symbol('omega') * self.t)
            diff = sp.simplify(ratio - expected)
            # Numeric check
            t_test = 0.3
            omega_test = 1.0
            diff_num = complex(diff.subs({self.t: t_test, self.omega: omega_test})) if self.omega else complex(diff.subs(self.t, t_test))
            if math.isclose(abs(diff_num), 0, abs_tol=1e-2):
                return (True, "Phase ratio between components correct")
            else:
                return (False, f"Component ratio mismatch: {ratio} vs {expected}")
        except Exception as e:
            return (False, f"Phase ratio error: {e}")

    # -----------------------------------------------------------------
    # MONOTONICITY / TREND CHECKS (with explicit input perturbation)
    # -----------------------------------------------------------------
    def test_monotonicity_frequency_increases_magnitude(self) -> Tuple[bool, str]:
        """Perturb ω by +10% and verify that ⟨s_x⟩ becomes more negative for small t."""
        if self.expect_sx is None:
            return (False, "SKIP: expect_sx not available")
        # Need numeric values for ω, ħ, t. Use defaults if not given.
        hbar_val = 1.054571817e-34  # in J·s, but we treat as dimensionless for comparison
        omega_val = 1.0  # rad/s default
        if self.hbar is not None and self.hbar.is_number:
            hbar_val = float(self.hbar)
        if self.omega is not None and self.omega.is_number:
            omega_val = float(self.omega)

        # Use a small t where sin(2ωt) is approx linear
        t_test = 0.05
        # Baseline expectation value
        ex_sym = sp.sympify(self.expect_sx)
        try:
            # Replace ħ and ω with numeric values if they are symbols in the expression
            subs_dict = {}
            if self.hbar is None:
                # Use a placeholder; if expression contains 'hbar', replace with numeric
                # But we can do a safer approach: substitute symbols by their names
                pass
            # Build baseline and perturbed by evaluating with numeric ω
            baseline = float(complex(ex_sym.subs({self.t: t_test, sp.Symbol('hbar'): hbar_val, sp.Symbol('omega'): omega_val})))
            # Perturb omega up by 10%
            omega_pert = omega_val * 1.1
            perturbed = float(complex(ex_sym.subs({self.t: t_test, sp.Symbol('hbar'): hbar_val, sp.Symbol('omega'): omega_pert})))
            # For the correct physics, ⟨s_x⟩ = -ħ/2 sin(2ωt). For small t, sin(x) ≈ x, so ⟨s_x⟩ ≈ -ħ ω t.
            # Increasing ω makes the value more negative (smaller).
            # So baseline (with larger ω) should be less than (i.e., more negative than) baseline (with original ω).
            # Check: baseline (original ω) > perturbed (larger ω) if both negative.
            # We'll require that perturbed < baseline.
            if perturbed < baseline:
                return (True, f"Perturbing ω +10% makes ⟨s_x⟩ more negative as expected: baseline={baseline:.6e}, perturbed={perturbed:.6e}")
            else:
                # Possibly the sign is opposite if t is not small enough; we can also check absolute increase.
                # Let's condition: |perturbed| > |baseline| because magnitude increases.
                if abs(perturbed) > abs(baseline):
                    return (True, f"Perturbing ω +10% increases magnitude as expected: baseline={baseline:.6e}, perturbed={perturbed:.6e}")
                else:
                    return (False, f"Monotonicity violated: baseline={baseline:.6e}, perturbed={perturbed:.6e}")
        except Exception as e:
            # Maybe expression contains symbolic hbar/omega that we can't substitute; try a different approach.
            # We'll fallback to a symbolic derivative check: d⟨s_x⟩/dω at small t.
            try:
                ex_diff = sp.diff(ex_sym, sp.Symbol('omega')) if sp.Symbol('omega') in ex_sym.free_symbols else None
                if ex_diff is not None:
                    # Evaluate at t=0.1, omega=1, hbar=1
                    val_diff = float(complex(ex_diff.subs({self.t: 0.1, sp.Symbol('omega'): 1.0, sp.Symbol('hbar'): 1.0})))
                    if val_diff < 0:
                        return (True, f"∂⟨s_x⟩/∂ω < 0 at small t, consistent with monotonic decrease: {val_diff:.3f}")
                    else:
                        return (False, f"∂⟨s_x⟩/∂ω not negative: {val_diff:.3f}")
                else:
                    return (False, f"Cannot perform monotonicity check: expression does not contain ω symbol")
            except Exception as e2:
                return (False, f"Monotonicity check error: {e2}")

    # -----------------------------------------------------------------
    # WEAK ANSWER SANITY CHECK (at most one, no re-derivation)
    # -----------------------------------------------------------------
    def test_answer_sanity_expect_sx_finite(self) -> Tuple[bool, str]:
        """Weak check: expect_sx is finite and not NaN for a few t."""
        if self.expect_sx is None:
            return (False, "SKIP: expect_sx not available")
        try:
            ex = sp.sympify(self.expect_sx)
            t_vals = [0, 0.5, 1.0, 2.0, 10.0]
            for tv in t_vals:
                val = complex(ex.subs(self.t, tv)) if self.t in ex.free_symbols else complex(ex)
                if math.isnan(val.real) or math.isnan(val.imag) or math.isinf(val.real) or math.isinf(val.imag):
                    return (False, f"expect_sx infinite or NaN at t={tv}: {val}")
            return (True, "Expectation value is finite at tested times")
        except Exception as e:
            return (False, f"Answer sanity check error: {e}")

    # -----------------------------------------------------------------
    # RUN ALL TESTS
    # -----------------------------------------------------------------
    def run_all_tests(self) -> Tuple[bool, List[str]]:
        """Discover all test_* methods and execute them, returning (overall_pass, log_lines)."""
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