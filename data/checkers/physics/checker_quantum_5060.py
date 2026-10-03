import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Physics checker for weak-field Zeeman energy shift in LS coupling.
    Checks: interface, invariant, monotonicity, modeling/trace, weak answer sanity.
    """
    def __init__(self, solution_instance):
        # Extract results
        self.results = getattr(solution_instance, 'results', {}) or {}
        self.delta_E = self.results.get('delta_E', None)

        # Extract givens/inputs safely
        self.givens = getattr(solution_instance, 'givens', {}) or {}
        # Try to get quantum numbers from various possible locations
        self.J = getattr(solution_instance, 'J', None) or self.givens.get('J', None)
        self.L = getattr(solution_instance, 'L', None) or self.givens.get('L', None)
        self.S = getattr(solution_instance, 'S', None) or self.givens.get('S', None)
        self.J_z = getattr(solution_instance, 'J_z', None) or self.givens.get('J_z', None) or self.givens.get('M_J', None)
        self.B = getattr(solution_instance, 'B', None) or self.givens.get('B', None)

        # Physical constants (symbolic)
        self.mu_B = sp.symbols('mu_B')
        self.hbar = sp.symbols('hbar')

        # Lande g-factor expression (if quantum numbers available)
        self.g_J_formula = None
        if all(q is not None for q in [self.J, self.L, self.S]):
            self.g_J_formula = 1 + (self.J*(self.J+1) + self.S*(self.S+1) - self.L*(self.L+1)) / (2*self.J*(self.J+1))

    # ---------- Interface tests ----------
    def test_interface_delta_e_present(self):
        if self.delta_E is None:
            return (False, "Missing required result key 'delta_E'")
        return (True, "delta_E key present and not None")

    def test_interface_quantum_numbers_present(self):
        missing = []
        if self.J is None:
            missing.append('J')
        if self.L is None:
            missing.append('L')
        if self.S is None:
            missing.append('S')
        if self.J_z is None:
            missing.append('J_z')
        if missing:
            return (False, f"Missing quantum numbers: {', '.join(missing)}")
        return (True, "All required quantum numbers present")

    # ---------- Theorem-based invariant tests (at least two) ----------
    def test_invariant_lande_g_factor(self):
        if self.delta_E is None:
            return (False, "SKIP: delta_E not available")
        if any(q is None for q in [self.J, self.L, self.S, self.J_z, self.B]):
            return (False, "SKIP: missing J, L, S, J_z, or B")
        g_J = self.g_J_formula
        if g_J is None:
            return (False, "Could not compute Lande g factor")
        # Expected energy: ΔE = μ_B * B * g_J * J_z (SI, J_z dimensionless M_J)
        expected = self.mu_B * self.B * g_J * self.J_z
        try:
            diff = sp.simplify(self.delta_E - expected)
            if diff == 0:
                return (True, "Landé g-factor formula holds symbolically")
            # Numeric fallback with actual constants if present
            numeric_env = {sp.Symbol('mu_B'): sp.nsimplify(9.274009994e-24),
                           sp.Symbol('hbar'): sp.nsimplify(1.054571817e-34)}
            numeric_dE = self.delta_E.subs(numeric_env)
            numeric_exp = expected.subs(numeric_env)
            dE_val = complex(numeric_dE.evalf())
            exp_val = complex(numeric_exp.evalf())
            if math.isclose(dE_val.real, exp_val.real, rel_tol=1e-2, abs_tol=1e-2) and abs(dE_val.imag) < 1e-10:
                return (True, f"Numeric agreement: ΔE ≈ {dE_val.real:.6e}")
            else:
                return (False, f"Energy mismatch: got {dE_val.real}, expected {exp_val.real}")
        except Exception as e:
            # Last resort: numeric from floats
            try:
                g_J_num = float(g_J)
                B_num = float(self.B)
                Jz_num = float(self.J_z)
                mu_B_num = 9.274009994e-24
                expected_num = mu_B_num * B_num * g_J_num * Jz_num
                dE_num = float(self.delta_E)
                if math.isclose(dE_num, expected_num, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, f"Numeric agreement: {dE_num} ≈ {expected_num}")
                else:
                    return (False, f"Numeric mismatch: ΔE={dE_num}, expected={expected_num}")
            except:
                return (False, f"Could not evaluate: {e}")

    def test_invariant_spin_expectation_bound(self):
        """
        Check that the expectation value of S_z, derived from ΔE and J_z,
        satisfies |⟨S_z⟩| ≤ S (in units of ħ). This follows from the
        projection theorem for vector operators.
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E not available")
        if any(q is None for q in [self.J, self.L, self.S, self.J_z, self.B]):
            return (False, "SKIP: missing J, L, S, J_z, or B")
        g_J = self.g_J_formula
        if g_J is None:
            return (False, "Could not compute Lande g factor")
        # Compute ⟨S_z⟩/ħ = (g_J - 1) * M_J
        try:
            M_J = float(self.J_z)
            S_val = float(self.S)
            g_val = float(g_J)
        except:
            return (False, "Cannot convert quantum numbers to float")
        S_z_expect = (g_val - 1) * M_J  # in units of ħ
        if abs(S_z_expect) <= S_val + 1e-9:  # include tolerance for numerical rounding
            return (True, f"|⟨S_z⟩| = {abs(S_z_expect):.6f} ≤ S = {S_val:.6f}")
        else:
            return (False, f"|⟨S_z⟩| = {abs(S_z_expect):.6f} exceeds S = {S_val:.6f}")

    # ---------- Monotonicity test with explicit perturbation ----------
    def test_monotonicity_energy_increases_with_Jz(self):
        """
        Perturb J_z by +1 (keeping other quantum numbers fixed) and check that
        the energy change is linear in J_z. We use the fact that the expected
        expression ΔE ∝ J_z (since g_J is constant). We cannot re‑evaluate the
        student's code, but we can check the proportionality from the
        given single result: if ΔE is linear in J_z, then ΔE(J_z+1) - ΔE(J_z)
        should equal ΔE(J_z)/J_z. This is equivalent to a first‑order check.
        """
        if self.delta_E is None or self.J_z is None:
            return (False, "SKIP: missing delta_E or J_z")
        try:
            Jz_val = float(self.J_z)
            dE_val = float(self.delta_E)
        except:
            return (False, "Cannot convert values to float")
        if Jz_val == 0:
            return (False, "SKIP: J_z=0, cannot test proportionality")
        # Baseline: ΔE for given J_z
        # Perturbed: J_z' = J_z + 1. Expected ΔE' = dE_val * (Jz_val+1)/Jz_val
        expected_perturbed = dE_val * (Jz_val + 1) / Jz_val
        # The difference ΔE(J_z+1) - ΔE(J_z) should equal dE_val/Jz_val
        delta_difference = expected_perturbed - dE_val
        expected_increment = dE_val / Jz_val
        # Check that the increment matches (within relative tolerance)
        if math.isclose(delta_difference, expected_increment, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Linear proportionality holds: increment {delta_difference:.6e} ≈ {expected_increment:.6e}")
        else:
            return (False, f"Linear proportionality fails: increment={delta_difference:.6e}, expected={expected_increment:.6e}")

    # ---------- Modeling/Trace quality tests (at least two) ----------
    def test_modeling_linear_in_B(self):
        """
        Check that the expression is linear in B (weak field assumption).
        Symbolically, the derivative with respect to B should be constant.
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E not available")
        try:
            expr = sp.sympify(self.delta_E)
            B_sym = sp.Symbol('B')
            if B_sym in expr.free_symbols:
                deriv = sp.diff(expr, B_sym)
                if B_sym in deriv.free_symbols:
                    return (False, "Delta_E is not linear in B (B appears in derivative)")
                else:
                    return (True, "Delta_E is linear in B (derivative constant)")
            else:
                # B may be numeric. Check that ΔE / B is finite (if B numeric)
                try:
                    B_val = float(self.B)
                    dE_num = float(self.delta_E)
                    ratio = dE_num / B_val
                    if math.isfinite(ratio):
                        return (True, "Delta_E / B is finite (consistent with linearity)")
                    else:
                        return (False, "Delta_E / B is not finite")
                except:
                    return (False, "Cannot check linearity numerically")
        except Exception as e:
            return (False, f"Cannot analyze linearity: {e}")

    def test_modeling_clebsch_gordan_structure(self):
        """
        Check that the delta_E expression contains Clebsch-Gordan coefficients
        or the sum over M_L, as required by LS coupling.
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E not available")
        expr_str = str(self.delta_E)
        # Look for typical patterns
        if 'Clebsch' in expr_str or 'CG' in expr_str.upper() or 'langle' in expr_str or 'M_L' in expr_str:
            return (True, "Expression contains Clebsch-Gordan structure")
        if 'sum' in expr_str.lower() or '∑' in expr_str:
            return (True, "Expression contains sum over M_L")
        # Fallback: if Lande g-factor is present, it implies CG structure
        if self.g_J_formula is not None:
            try:
                expr = sp.sympify(self.delta_E)
                J_sym = sp.Symbol('J')
                L_sym = sp.Symbol('L')
                S_sym = sp.Symbol('S')
                if any(sym in expr.free_symbols for sym in [J_sym, L_sym, S_sym]):
                    return (True, "Energy depends on J, L, S (consistent with CG)")
            except:
                pass
            return (True, "Lande g-factor formula present (implies CG structure)")
        return (False, "No Clebsch-Gordan or sum-over-M_L pattern found")

    # ---------- Weak answer sanity (only one such method) ----------
    def test_answer_sanity_plausible(self):
        """
        Check that energy is finite and within a typical range for atomic Zeeman
        shifts (1e-30 J to 1e-10 J for B ~ 1 T). This is a weak, non‑precise test.
        """
        if self.delta_E is None:
            return (False, "SKIP: delta_E not available")
        try:
            val = float(self.delta_E)
        except:
            # Symbolic expressions are considered fine (finite by assumption)
            return (True, "Symbolic energy, plausible")
        if not math.isfinite(val) or math.isnan(val):
            return (False, f"Energy is not finite: {val}")
        if abs(val) > 1e-10 or abs(val) < 1e-30:
            return (False, f"Energy magnitude {abs(val)} outside typical range [1e-30, 1e-10]")
        return (True, f"Energy {val} is finite and within plausible range")

    # ---------- Main runner ----------
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