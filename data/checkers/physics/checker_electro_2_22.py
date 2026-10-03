import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Physics auditor for coaxial cable problem (magnetic field, electric field, energy, L, C).
    """

    # Required result keys
    REQUIRED_KEYS = ['B_phi', 'E_r', 'U_m_per_length', 'U_e_per_length',
                     'L_per_length', 'C_per_length']

    def __init__(self, solution_instance):
        # Extract givens – do not assign numeric defaults, use None for missing.
        self.R1 = getattr(solution_instance, 'R1', None)
        self.R2 = getattr(solution_instance, 'R2', None)
        self.i = getattr(solution_instance, 'i', None)
        self.V = getattr(solution_instance, 'V', None)
        self.epsilon0 = getattr(solution_instance, 'epsilon0', None)
        self.mu0 = getattr(solution_instance, 'mu0', None)

        # Results dictionary – the main output from the model.
        raw_results = getattr(solution_instance, 'results', None)
        if raw_results is None:
            raw_results = {}
        self.results = raw_results

        # Pre-extract each result safely (None if missing)
        self.B_phi = self.results.get('B_phi', None)
        self.E_r = self.results.get('E_r', None)
        self.U_m = self.results.get('U_m_per_length', None)
        self.U_e = self.results.get('U_e_per_length', None)
        self.L = self.results.get('L_per_length', None)
        self.C = self.results.get('C_per_length', None)

    # ------------------------------------------------------------------
    # Helper: convert to sympy expression
    # ------------------------------------------------------------------
    @staticmethod
    def _to_sympy(expr):
        if isinstance(expr, sp.Expr):
            return expr
        if isinstance(expr, (int, float)):
            return sp.N(expr)
        try:
            return sp.sympify(expr)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Helper: evaluate sympy expression numerically given substitution dict
    # ------------------------------------------------------------------
    @staticmethod
    def _eval_expr(expr, subs_dict):
        try:
            val = expr.subs(subs_dict).evalf()
            return float(val)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # INTERFACE TESTS
    # ------------------------------------------------------------------
    def test_interface_required_keys_present(self):
        if self.results is None:
            return (False, "results dictionary is None")
        missing = [k for k in self.REQUIRED_KEYS if k not in self.results]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_results_not_none(self):
        none_keys = [k for k in self.REQUIRED_KEYS if self.results.get(k) is None]
        if none_keys:
            return (False, f"Keys with None value: {none_keys}")
        return (True, "No None values in results")

    # ------------------------------------------------------------------
    # INVARIANT TESTS (theorem-based physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_ampere_law(self):
        """B_phi * 2πr = μ₀ i (Ampere's law)."""
        B = self._to_sympy(self.B_phi)
        if B is None:
            return (False, "SKIP: B_phi not available or not convertible to sympy")
        r_sp = sp.Symbol('r')
        try:
            lhs = sp.simplify(B * 2 * sp.pi * r_sp)
        except Exception:
            return (False, "SKIP: cannot simplify B_phi expression")
        if self.mu0 is not None and self.i is not None:
            mu0_sp = sp.N(self.mu0) if isinstance(self.mu0, (int, float)) else sp.Symbol('mu0')
            i_sp = sp.N(self.i) if isinstance(self.i, (int, float)) else sp.Symbol('i')
            try:
                rhs = mu0_sp * i_sp
            except Exception:
                return (False, "SKIP: cannot construct RHS")
            diff = sp.simplify(lhs - rhs)
            if diff == 0:
                return (True, "Ampere's law satisfied (symbolic)")
            # Numeric fallback
            if self.R1 is not None and self.R2 is not None and self.R1 < self.R2:
                r_test = (self.R1 + self.R2) / 2.0
                subs = {r_sp: r_test}
                lhs_num = self._eval_expr(lhs, subs)
                rhs_num = self._eval_expr(rhs, {})
                if lhs_num is not None and rhs_num is not None and math.isclose(lhs_num, rhs_num, rel_tol=1e-2):
                    return (True, "Ampere's law satisfied (numerical fallback)")
                else:
                    return (False, f"Ampere's law failed: LHS={lhs_num:.6e}, RHS={rhs_num:.6e}")
            else:
                return (False, "Cannot verify Ampere's law: R1 or R2 missing, and symbolic mismatch")
        else:
            return (False, "SKIP: mu0 or i missing")

    def test_invariant_gauss_law_er(self):
        """E_r * r * ln(R2/R1) = V (from Gauss & potential)."""
        Er = self._to_sympy(self.E_r)
        if Er is None:
            return (False, "SKIP: E_r not available or not convertible")
        r_sp = sp.Symbol('r')
        R1_sp = sp.Symbol('R1')
        R2_sp = sp.Symbol('R2')
        try:
            lhs = sp.simplify(Er * r_sp * sp.log(R2_sp / R1_sp))
        except Exception:
            return (False, "SKIP: cannot simplify E_r expression")
        if self.R1 is not None and self.R2 is not None and self.V is not None:
            r_test = (self.R1 + self.R2) / 2.0
            subs = {R1_sp: self.R1, R2_sp: self.R2, r_sp: r_test}
            lhs_num = self._eval_expr(lhs, subs)
            if lhs_num is not None and math.isclose(lhs_num, self.V, rel_tol=1e-2, abs_tol=1e-2):
                return (True, "Gauss law relation from potential satisfied (numerical)")
            else:
                return (False, f"E_r relation failed: lhs_num={lhs_num:.6e}, V={self.V:.6e}")
        else:
            return (False, "SKIP: missing givens (R1, R2, V) to verify E_r relation")

    def test_invariant_magnetic_energy_formula(self):
        """U_m_per_length = (mu0 * i^2) / (4π) * ln(R2/R1)."""
        Um = self._to_sympy(self.U_m)
        if Um is None:
            return (False, "SKIP: U_m_per_length not available")
        if self.mu0 is not None and self.i is not None and self.R1 is not None and self.R2 is not None:
            mu0_sp = sp.N(self.mu0) if isinstance(self.mu0, (int, float)) else sp.Symbol('mu0')
            i_sp = sp.N(self.i) if isinstance(self.i, (int, float)) else sp.Symbol('i')
            try:
                expected = (mu0_sp * i_sp**2) / (4 * sp.pi) * sp.log(self.R2 / self.R1)
            except Exception:
                return (False, "SKIP: error constructing expected expression")
            diff = sp.simplify(Um - expected)
            if diff == 0:
                return (True, "Magnetic energy formula satisfied (symbolic)")
            diff_num = self._eval_expr(diff, {})
            if diff_num is not None and math.isclose(diff_num, 0.0, abs_tol=1e-2):
                return (True, "Magnetic energy formula satisfied (numerical)")
            else:
                return (False, f"Magnetic energy formula failed, diff={diff_num:.6e}")
        else:
            return (False, "SKIP: missing givens for magnetic energy check")

    def test_invariant_electric_energy_formula(self):
        """U_e_per_length = (π * epsilon0 * V^2) / ln(R2/R1)."""
        Ue = self._to_sympy(self.U_e)
        if Ue is None:
            return (False, "SKIP: U_e_per_length not available")
        if self.epsilon0 is not None and self.V is not None and self.R1 is not None and self.R2 is not None:
            try:
                expected = (sp.pi * sp.N(self.epsilon0) * sp.N(self.V)**2) / sp.log(self.R2 / self.R1)
            except Exception:
                return (False, "SKIP: error constructing expected expression")
            diff = sp.simplify(Ue - expected)
            if diff == 0:
                return (True, "Electric energy formula satisfied (symbolic)")
            diff_num = self._eval_expr(diff, {})
            if diff_num is not None and math.isclose(diff_num, 0.0, abs_tol=1e-2):
                return (True, "Electric energy formula satisfied (numerical)")
            else:
                return (False, f"Electric energy formula failed, diff={diff_num:.6e}")
        else:
            return (False, "SKIP: missing givens for electric energy check")

    def test_invariant_inductance_formula(self):
        """L_per_length = μ0/(2π) * ln(R2/R1)."""
        L = self._to_sympy(self.L)
        if L is None:
            return (False, "SKIP: L_per_length not available")
        if self.mu0 is not None and self.R1 is not None and self.R2 is not None:
            expected = (sp.N(self.mu0) / (2 * sp.pi)) * sp.log(self.R2 / self.R1)
            diff = sp.simplify(L - expected)
            if diff == 0:
                return (True, "Inductance formula satisfied (symbolic)")
            diff_num = self._eval_expr(diff, {})
            if diff_num is not None and math.isclose(diff_num, 0.0, abs_tol=1e-2):
                return (True, "Inductance formula satisfied (numerical)")
            else:
                return (False, f"Inductance formula failed, diff={diff_num:.6e}")
        else:
            return (False, "SKIP: missing givens for inductance check")

    def test_invariant_capacitance_formula(self):
        """C_per_length = 2π ε0 / ln(R2/R1)."""
        C = self._to_sympy(self.C)
        if C is None:
            return (False, "SKIP: C_per_length not available")
        if self.epsilon0 is not None and self.R1 is not None and self.R2 is not None:
            expected = (2 * sp.pi * sp.N(self.epsilon0)) / sp.log(self.R2 / self.R1)
            diff = sp.simplify(C - expected)
            if diff == 0:
                return (True, "Capacitance formula satisfied (symbolic)")
            diff_num = self._eval_expr(diff, {})
            if diff_num is not None and math.isclose(diff_num, 0.0, abs_tol=1e-2):
                return (True, "Capacitance formula satisfied (numerical)")
            else:
                return (False, f"Capacitance formula failed, diff={diff_num:.6e}")
        else:
            return (False, "SKIP: missing givens for capacitance check")

    # ------------------------------------------------------------------
    # MODELING / TRACE TESTS
    # ------------------------------------------------------------------
    def test_modeling_energy_from_fields_consistency(self):
        """Magnetic energy per length = 0.5 * L * i^2,
           Electric energy per length = 0.5 * C * V^2."""
        # Magnetic side
        if self.U_m is not None and self.L is not None and self.i is not None:
            Um = self._to_sympy(self.U_m)
            L_sym = self._to_sympy(self.L)
            i_sym = sp.N(self.i) if isinstance(self.i, (int, float)) else sp.Symbol('i')
            expected_Um = sp.Rational(1,2) * L_sym * i_sym**2
            diff = sp.simplify(Um - expected_Um)
            if diff == 0:
                mag_ok = True
            else:
                diff_num = self._eval_expr(diff, {})
                mag_ok = diff_num is not None and math.isclose(diff_num, 0.0, abs_tol=1e-2)
        else:
            mag_ok = None

        # Electric side
        if self.U_e is not None and self.C is not None and self.V is not None:
            Ue = self._to_sympy(self.U_e)
            C_sym = self._to_sympy(self.C)
            V_sym = sp.N(self.V) if isinstance(self.V, (int, float)) else sp.Symbol('V')
            expected_Ue = sp.Rational(1,2) * C_sym * V_sym**2
            diff = sp.simplify(Ue - expected_Ue)
            if diff == 0:
                elec_ok = True
            else:
                diff_num = self._eval_expr(diff, {})
                elec_ok = diff_num is not None and math.isclose(diff_num, 0.0, abs_tol=1e-2)
        else:
            elec_ok = None

        if mag_ok is None and elec_ok is None:
            return (False, "SKIP: insufficient data for energy relations")
        msgs = []
        if mag_ok is not None:
            msgs.append(f"Magnetic: {'ok' if mag_ok else 'FAIL'}")
        if elec_ok is not None:
            msgs.append(f"Electric: {'ok' if elec_ok else 'FAIL'}")
        all_ok = (mag_ok is None or mag_ok) and (elec_ok is None or elec_ok)
        return (all_ok, "; ".join(msgs))

    def test_modeling_capacitance_from_charge_relations(self):
        """C_per_length = λ/V, with λ = 2π ε0 V / ln(R2/R1)."""
        C = self._to_sympy(self.C)
        if C is None or self.V is None or self.epsilon0 is None or self.R1 is None or self.R2 is None:
            return (False, "SKIP: missing required variables")
        expected_C = (2 * sp.pi * sp.N(self.epsilon0)) / sp.log(self.R2 / self.R1)
        diff = sp.simplify(C - expected_C)
        if diff == 0:
            return (True, "C = λ/V satisfied (symbolic)")
        diff_num = self._eval_expr(diff, {})
        if diff_num is not None and math.isclose(diff_num, 0.0, abs_tol=1e-2):
            return (True, "C = λ/V satisfied (numerical)")
        else:
            return (False, f"C = λ/V failed, diff={diff_num:.6e}")

    # ------------------------------------------------------------------
    # MONOTONICITY TESTS (explicit perturbation of an input)
    # ------------------------------------------------------------------
    def test_monotonicity_Um_increases_with_i_squared(self):
        """Perturb current i by +10% and check that magnetic energy U_m increases
           (approximately as i^2, so about 21% increase)."""
        Um = self._to_sympy(self.U_m)
        if Um is None or self.i is None:
            return (False, "SKIP: missing U_m or i")
        i_baseline = self.i
        i_perturbed = i_baseline * 1.10  # +10%
        # Since U_m expression may be symbolic, we substitute new i
        # We need to know which symbol represents i in the expression.
        # Usually the expression uses symbol 'i'. We'll try that.
        i_sym = sp.Symbol('i')
        subs_baseline = {i_sym: i_baseline}
        subs_perturbed = {i_sym: i_perturbed}
        # Also need other symbols like mu0, R1, R2. They should be in expression but we can provide fallback numeric values.
        # For evaluation, we need all symbols present. We'll try to evaluate; if missing, use defaults.
        # We'll extract free symbols and provide numeric if possible.
        free_syms = Um.free_symbols
        # Build substitution dict for baseline and perturbed
        subs_base = {}
        subs_pert = {}
        for sym in free_syms:
            name = str(sym)
            if name == 'i':
                subs_base[sym] = i_baseline
                subs_pert[sym] = i_perturbed
            elif name == 'mu0':
                subs_base[sym] = self.mu0 if self.mu0 is not None else 4*math.pi*1e-7
                subs_pert[sym] = subs_base[sym]
            elif name == 'R1':
                subs_base[sym] = self.R1 if self.R1 is not None else 1e-3
                subs_pert[sym] = subs_base[sym]
            elif name == 'R2':
                subs_base[sym] = self.R2 if self.R2 is not None else 2e-3
                subs_pert[sym] = subs_base[sym]
            else:
                # Unknown symbol, try to use numeric if possible
                # If it's a number already, it's free? Actually numeric won't be free.
                # For safety, skip and hope expression can still evaluate.
                pass
        # Evaluate baseline and perturbed
        Um_base = self._eval_expr(Um, subs_base)
        Um_pert = self._eval_expr(Um, subs_pert)
        if Um_base is None or Um_pert is None:
            return (False, "Cannot evaluate U_m for baseline/perturbed")
        # Check that perturbed > baseline (positive change)
        if Um_pert > Um_base:
            # Optional: check roughly i^2 scaling: ratio should be about 1.21
            ratio = Um_pert / Um_base
            expected_ratio = (i_perturbed / i_baseline)**2  # = 1.21
            if math.isclose(ratio, expected_ratio, rel_tol=0.1):
                return (True, f"U_m increases from {Um_base:.6e} to {Um_pert:.6e}, ratio={ratio:.4f} (expected ~1.21)")
            else:
                # Still a monotonic increase, just not exact square scale; accept as monotonic
                return (True, f"U_m increases (monotonic): {Um_base:.6e} -> {Um_pert:.6e}")
        else:
            return (False, f"U_m did not increase with i: base={Um_base:.6e}, pert={Um_pert:.6e}")

    def test_monotonicity_L_increases_with_R2(self):
        """Perturb outer radius R2 by +5% and check that inductance L increases."""
        L = self._to_sympy(self.L)
        if L is None or self.R2 is None:
            return (False, "SKIP: missing L or R2")
        # We need R1 and mu0 for evaluation, but expression may contain them.
        # Determine symbols
        free_syms = L.free_symbols
        subs_base = {}
        subs_pert = {}
        # Identify symbols
        for sym in free_syms:
            name = str(sym)
            if name == 'R2':
                subs_base[sym] = self.R2
                subs_pert[sym] = self.R2 * 1.05  # +5%
            elif name == 'R1':
                subs_base[sym] = self.R1 if self.R1 is not None else 1e-3
                subs_pert[sym] = subs_base[sym]  # unchanged
            elif name == 'mu0':
                subs_base[sym] = self.mu0 if self.mu0 is not None else 4*math.pi*1e-7
                subs_pert[sym] = subs_base[sym]
            else:
                # other symbols; try to assign numeric if possible, else skip
                pass
        L_base = self._eval_expr(L, subs_base)
        L_pert = self._eval_expr(L, subs_pert)
        if L_base is None or L_pert is None:
            return (False, "Cannot evaluate L for baseline/perturbed")
        if L_pert > L_base:
            return (True, f"L increases from {L_base:.6e} to {L_pert:.6e} when R2 increased by 5%")
        else:
            return (False, f"L did not increase: base={L_base:.6e}, pert={L_pert:.6e}")

    # ------------------------------------------------------------------
    # WEAK ANSWER SANITY (at most one)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_and_sign(self):
        """All results should be finite and have physically expected sign/range."""
        checks = []
        # B_phi should be positive (using typical values)
        B = self._to_sympy(self.B_phi)
        if B is not None and self.R1 is not None and self.R2 is not None:
            r_test = (self.R1 + self.R2) / 2.0
            # Provide fallback for mu0 and i if missing
            subs = {sp.Symbol('r'): r_test,
                    sp.Symbol('mu0'): self.mu0 if self.mu0 is not None else 4*math.pi*1e-7,
                    sp.Symbol('i'): self.i if self.i is not None else 1.0}
            val = self._eval_expr(B, subs)
            if val is not None:
                checks.append(("B_phi positive", val > 0))
        # E_r positive
        Er = self._to_sympy(self.E_r)
        if Er is not None and self.R1 is not None and self.R2 is not None and self.V is not None:
            r_test = (self.R1 + self.R2) / 2.0
            subs = {sp.Symbol('r'): r_test,
                    sp.Symbol('V'): self.V}
            val = self._eval_expr(Er, subs)
            if val is not None:
                checks.append(("E_r positive", val > 0))
        # Energies positive
        for name, expr in [("U_m", self.U_m), ("U_e", self.U_e), ("L", self.L), ("C", self.C)]:
            if expr is not None:
                v = self._to_sympy(expr)
                if v is not None:
                    # Try to evaluate with plausible defaults if needed
                    free = v.free_symbols
                    subs = {}
                    for sym in free:
                        sname = str(sym)
                        if sname == 'i' and self.i is not None:
                            subs[sym] = self.i
                        elif sname == 'V' and self.V is not None:
                            subs[sym] = self.V
                        elif sname == 'R1' and self.R1 is not None:
                            subs[sym] = self.R1
                        elif sname == 'R2' and self.R2 is not None:
                            subs[sym] = self.R2
                        elif sname == 'mu0' and self.mu0 is not None:
                            subs[sym] = self.mu0
                        elif sname == 'epsilon0' and self.epsilon0 is not None:
                            subs[sym] = self.epsilon0
                        else:
                            # missing, use a typical number (not defaulting to 0!)
                            # but we can still evaluate partially; if fails, skip
                            pass
                    num_val = self._eval_expr(v, subs)
                    if num_val is not None and math.isfinite(num_val):
                        checks.append((f"{name} positive", num_val > 0))

        all_ok = all(ok for _, ok in checks)
        msg = "; ".join(f"{n}: {'PASS' if ok else 'FAIL'}" for n, ok in checks) if checks else "No checks performed"
        return (all_ok, msg)

    # ------------------------------------------------------------------
    # MAIN RUNNER
    # ------------------------------------------------------------------
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
                passed, msg = False, f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)