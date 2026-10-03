import math
from typing import Any, Dict, List, Tuple, Optional

import sympy as sp
import traceback

class PhysicsChecker:
    """
    Auditor for a time-varying capacitor with a dielectric.
    Checks physical consistency and modeling quality based on provided givens and results.
    """

    def __init__(self, solution_instance: Any):
        # Safe extraction of givens
        givens = getattr(solution_instance, 'givens', {})
        if givens is None:
            givens = {}
        self.V0 = givens.get('V0', None)
        self.omega = givens.get('omega', None)
        self.R = givens.get('R', None)
        self.d = givens.get('d', None)
        self.k_e = givens.get('k_e', None)
        self.epsilon0 = givens.get('epsilon0', None)
        self.mu0 = givens.get('mu0', None)

        # Safe extraction of results
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        self.E_z = results.get('E_z', None)
        self.sigma_f = results.get('sigma_f', None)
        self.B_phi = results.get('B_phi', None)
        self.flux = results.get('flux', None)

        # Explanation steps (optional for trace checks)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ---------------------------------------------------------------
    # Helper to parse symbolic expressions
    # ---------------------------------------------------------------
    def _parse_sympy(self, expr):
        if isinstance(expr, str):
            try:
                return sp.sympify(expr)
            except:
                return None
        elif isinstance(expr, sp.Basic):
            return expr
        else:
            return None  # unknown type

    # ---------------------------------------------------------------
    # Interface tests: check existence of required outputs
    # ---------------------------------------------------------------
    def test_interface_electric_field(self) -> Tuple[bool, str]:
        if self.E_z is None:
            return False, "SKIP: E_z is missing"
        return True, "E_z present"

    def test_interface_free_charge_density(self) -> Tuple[bool, str]:
        if self.sigma_f is None:
            return False, "SKIP: sigma_f is missing"
        return True, "sigma_f present"

    def test_interface_magnetic_field(self) -> Tuple[bool, str]:
        if self.B_phi is None:
            return False, "SKIP: B_phi is missing"
        return True, "B_phi present"

    def test_interface_poynting_flux(self) -> Tuple[bool, str]:
        if self.flux is None:
            return False, "SKIP: flux is missing"
        return True, "flux present"

    # ---------------------------------------------------------------
    # Invariant tests: theorem-based relations (use isclose)
    # ---------------------------------------------------------------
    def test_invariant_electric_field_relation(self) -> Tuple[bool, str]:
        """Check E_z = V0/d * cos(omega*t) at t=0 and t=pi/(2*omega)"""
        if any(v is None for v in (self.V0, self.d, self.omega, self.E_z)):
            return False, "SKIP: Missing data for E_z invariant"
        t = sp.symbols('t', real=True)
        expected_expr = self.V0 / self.d * sp.cos(self.omega * t)
        E_expr = self._parse_sympy(self.E_z)
        if E_expr is None:
            return False, "SKIP: Cannot parse E_z expression"
        # evaluate at t=0
        try:
            val0 = float(E_expr.subs(t, 0))
            exp0 = float(expected_expr.subs(t, 0))
        except Exception:
            return False, "SKIP: Cannot evaluate at t=0"
        if not math.isclose(val0, exp0, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"At t=0: E_z={val0}, expected={exp0}"
        # evaluate at t=π/(2ω)
        try:
            t1 = math.pi / (2 * self.omega)
            val1 = float(E_expr.subs(t, t1))
            exp1 = float(expected_expr.subs(t, t1))
        except Exception:
            return False, "SKIP: Cannot evaluate at t=π/(2ω)"
        if not math.isclose(val1, exp1, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"At t=π/(2ω): E_z={val1}, expected={exp1}"
        return True, "E_z matches V0/d * cos(omega*t)"

    def test_invariant_free_charge_density(self) -> Tuple[bool, str]:
        """Check sigma_f = epsilon0 * k_e * E_z at two times"""
        if any(v is None for v in (self.epsilon0, self.k_e, self.E_z, self.sigma_f)):
            return False, "SKIP: Missing data for sigma_f invariant"
        t = sp.symbols('t', real=True)
        E_expr = self._parse_sympy(self.E_z)
        sig_expr = self._parse_sympy(self.sigma_f)
        if E_expr is None or sig_expr is None:
            return False, "SKIP: Cannot parse symbolic expressions"
        expected = self.epsilon0 * self.k_e * E_expr
        # check at t=0 and t=π/(2ω)
        if self.omega is None:
            return False, "SKIP: omega missing"
        test_times = [0, math.pi/(2*self.omega)]
        for t_val in test_times:
            try:
                v_sig = float(sig_expr.subs(t, t_val))
                v_exp = float(expected.subs(t, t_val))
            except Exception:
                return False, f"SKIP: Cannot evaluate at t={t_val}"
            if not math.isclose(v_sig, v_exp, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"At t={t_val}: sigma_f={v_sig}, expected={v_exp}"
        return True, "sigma_f = epsilon0 * k_e * E_z"

    def test_invariant_ampere_maxwell(self) -> Tuple[bool, str]:
        """Check Ampere-Maxwell: ∮B·dl = μ0 ε0 k_e dΦ_E/dt for a loop of radius r"""
        if any(v is None for v in (self.mu0, self.epsilon0, self.k_e, self.omega, self.V0, self.d, self.B_phi, self.R)):
            return False, "SKIP: Missing physical constants or B_phi"
        # Choose test radius = R/2 and time = π/(4ω) (sin non-zero)
        test_r = self.R / 2.0
        t_test = math.pi / (4.0 * self.omega)  # sin=√2/2
        r = sp.symbols('r', positive=True)
        t = sp.symbols('t', real=True)
        B_expr = self._parse_sympy(self.B_phi)
        if B_expr is None:
            return False, "SKIP: Cannot parse B_phi expression"
        # evaluate B_phi at test point
        try:
            B_val = float(B_expr.subs({r: test_r, t: t_test}))
        except Exception:
            return False, "SKIP: Cannot evaluate B_phi at test point"
        # left side: B * 2πr
        left = B_val * 2 * math.pi * test_r
        # right side: μ0 ε0 k_e πr^2 dE_z/dt
        dE_dt = -self.V0 / self.d * self.omega * math.sin(self.omega * t_test)
        right = self.mu0 * self.epsilon0 * self.k_e * math.pi * test_r**2 * dE_dt
        if not math.isclose(left, right, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Ampere-Maxwell mismatch: left={left}, right={right}"
        return True, "B_phi satisfies Ampere-Maxwell law"

    def test_invariant_poynting_flux(self) -> Tuple[bool, str]:
        """Check flux from S·dA matches given flux at a test time."""
        if any(v is None for v in (self.mu0, self.E_z, self.B_phi, self.flux, self.R, self.d, self.omega)):
            return False, "SKIP: Missing data for Poynting flux check"
        t_test = math.pi / (4.0 * self.omega)
        t = sp.symbols('t', real=True)
        r = sp.symbols('r', positive=True)
        E_expr = self._parse_sympy(self.E_z)
        B_expr = self._parse_sympy(self.B_phi)
        flux_expr = self._parse_sympy(self.flux)
        if any(e is None for e in (E_expr, B_expr, flux_expr)):
            return False, "SKIP: Cannot parse one of the expressions"
        try:
            E_val = float(E_expr.subs(t, t_test))
            B_val = float(B_expr.subs({r: self.R, t: t_test}))
        except Exception:
            return False, "SKIP: Cannot evaluate E or B at test point"
        # S_r = -E_z B_phi / μ0  (since ẑ×φ̂ = -r̂)
        S_r = -(E_val * B_val) / self.mu0
        # Flux = S_r * area = S_r * 2πR d
        expected_flux = S_r * 2 * math.pi * self.R * self.d
        try:
            given_flux = float(flux_expr.subs(t, t_test))
        except Exception:
            return False, "SKIP: Cannot evaluate flux expression"
        if not math.isclose(given_flux, expected_flux, rel_tol=2e-2, abs_tol=1e-2):
            return False, f"Flux mismatch at t={t_test}: given={given_flux}, expected={expected_flux}"
        return True, "Flux from S·dA is consistent"

    # ---------------------------------------------------------------
    # Modeling / Trace tests: sign conventions, dimensional analysis
    # ---------------------------------------------------------------
    def test_modeling_electric_field_sign(self) -> Tuple[bool, str]:
        """E_z and sigma_f should have the same sign (since sigma = ε0 k_e E)."""
        if any(v is None for v in (self.E_z, self.sigma_f, self.omega)):
            return False, "SKIP: Missing data for sign check"
        t = sp.symbols('t', real=True)
        E_expr = self._parse_sympy(self.E_z)
        sig_expr = self._parse_sympy(self.sigma_f)
        if E_expr is None or sig_expr is None:
            return False, "SKIP: Cannot parse expressions"
        t_test = 0.0  # cos=1
        try:
            E_val = float(E_expr.subs(t, t_test))
            sig_val = float(sig_expr.subs(t, t_test))
        except Exception:
            return False, "SKIP: Cannot evaluate at t=0"
        if (E_val > 0 and sig_val < 0) or (E_val < 0 and sig_val > 0):
            return False, f"Signs disagree at t=0: E_z={E_val}, sigma_f={sig_val}"
        return True, "E_z and sigma_f have consistent sign"

    def test_modeling_b_field_sign(self) -> Tuple[bool, str]:
        """B_phi sign should match dE_z/dt (positive dE/dt -> positive B_phi)."""
        if any(v is None for v in (self.B_phi, self.E_z, self.omega, self.V0, self.d)):
            return False, "SKIP: Missing data for B sign check"
        # choose t = 3π/(2ω) so dE/dt = V0 ω/d > 0
        t_test = 3*math.pi/(2*self.omega)
        r = sp.symbols('r', positive=True)
        t = sp.symbols('t', real=True)
        B_expr = self._parse_sympy(self.B_phi)
        if B_expr is None:
            return False, "SKIP: Cannot parse B_phi"
        test_r = self.R/2 if self.R else 0.1
        try:
            B_val = float(B_expr.subs({r: test_r, t: t_test}))
        except Exception:
            return False, "SKIP: Cannot evaluate B_phi"
        dE_dt = -self.V0/self.d * self.omega * math.sin(self.omega*t_test)  # positive
        if (B_val > 0 and dE_dt < 0) or (B_val < 0 and dE_dt > 0):
            return False, f"B_phi sign ({B_val}) inconsistent with dE/dt ({dE_dt})"
        return True, "B_phi sign consistent with dE_z/dt"

    def test_modeling_flux_time_dependence(self) -> Tuple[bool, str]:
        """Flux should contain sin(2ωt) or sin*cos factor."""
        if self.flux is None or self.omega is None:
            return False, "SKIP: Missing flux or omega"
        flux_expr = self._parse_sympy(self.flux)
        if flux_expr is None:
            return False, "SKIP: Cannot parse flux expression"
        t = sp.symbols('t', real=True)
        expanded = sp.expand(flux_expr)
        has_sin2 = expanded.has(sp.sin(2*self.omega*t))
        has_product = expanded.has(sp.sin(self.omega*t)*sp.cos(self.omega*t))
        if not (has_sin2 or has_product):
            return False, "Flux does not show sin(2ωt) or sin*cos dependence"
        return True, "Flux time dependence consistent with sin(2ωt)"

    # ---------------------------------------------------------------
    # Monotonicity tests: explicitly perturb an input and compare
    # ---------------------------------------------------------------
    def test_monotonicity_b_field_with_d(self) -> Tuple[bool, str]:
        """
        Perturb d by +10% and check that B_phi scales as 1/d.
        Evaluate B_phi at same r and t, with original d and perturbed d.
        """
        if any(v is None for v in (self.B_phi, self.d, self.R, self.omega, self.V0, self.mu0, self.epsilon0, self.k_e)):
            return False, "SKIP: Missing data for d monotonicity"
        # Parse B_phi expression; we need to replace numeric d with a symbol for substitution
        B_expr = self._parse_sympy(self.B_phi)
        if B_expr is None:
            return False, "SKIP: Cannot parse B_phi expression"
        # create a symbolic variable for d and attempt to substitute the numeric value
        d_sym = sp.Symbol('d')
        # If the expression contains the numeric value of d, replace it with the symbol
        # (this is safe even if the numeric value doesn't appear exactly, then substitution does nothing)
        B_expr_sym = B_expr.subs(self.d, d_sym)
        # define test r and t
        t_sym = sp.Symbol('t', real=True)
        r_sym = sp.Symbol('r', positive=True)
        test_r = self.R / 2.0
        t_test = math.pi / (4.0 * self.omega)
        # compute baseline B at original d
        try:
            B_orig = float(B_expr_sym.subs({d_sym: self.d, r_sym: test_r, t_sym: t_test}))
        except Exception:
            return False, "SKIP: Cannot evaluate B_orig"
        # perturb d: increase by 10%
        d_pert = 1.1 * self.d
        try:
            B_pert = float(B_expr_sym.subs({d_sym: d_pert, r_sym: test_r, t_sym: t_test}))
        except Exception:
            return False, "SKIP: Cannot evaluate B_pert"
        # expected ratio: B_pert / B_orig = d_orig / d_pert (since B ∝ 1/d)
        if B_orig == 0.0:
            return False, "B_orig is zero, cannot compute ratio"
        actual_ratio = B_pert / B_orig
        expected_ratio = self.d / d_pert  # ≈ 0.90909
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"d monotonicity: ratio={actual_ratio}, expected~{expected_ratio}"
        return True, "B_phi scaling with d is consistent (∝1/d)"

    def test_monotonicity_flux_with_R(self) -> Tuple[bool, str]:
        """
        Perturb R by factor and check flux scales as R^2.
        Use the flux expression, evaluate at R*0.5 and R*1.5, compare ratio.
        """
        if any(v is None for v in (self.flux, self.R, self.omega)):
            return False, "SKIP: Missing data for R monotonicity"
        flux_expr = self._parse_sympy(self.flux)
        if flux_expr is None:
            return False, "SKIP: Cannot parse flux expression"
        # Replace numeric R with symbolic R for perturbation
        R_sym = sp.Symbol('R')
        flux_expr_sym = flux_expr.subs(self.R, R_sym)
        t = sp.Symbol('t', real=True)
        t_test = math.pi / (4.0 * self.omega)
        # evaluate at R1 = 0.5*R and R2 = 1.5*R
        R1 = 0.5 * self.R
        R2 = 1.5 * self.R
        try:
            flux1 = float(flux_expr_sym.subs({R_sym: R1, t: t_test}))
            flux2 = float(flux_expr_sym.subs({R_sym: R2, t: t_test}))
        except Exception:
            return False, "SKIP: Cannot evaluate flux at perturbed R"
        if flux1 == 0.0:
            return False, "flux1 is zero, cannot compute ratio"
        actual_ratio = flux2 / flux1
        expected_ratio = (R2/R1)**2  # 9.0
        if not math.isclose(actual_ratio, expected_ratio, rel_tol=1e-2, abs_tol=0.2):
            return False, f"R monotonicity: ratio={actual_ratio}, expected~{expected_ratio}"
        return True, "Flux scales as R^2"

    # ---------------------------------------------------------------
    # Weak answer sanity (at most one)
    # ---------------------------------------------------------------
    def test_answer_sanity_finite_signs(self) -> Tuple[bool, str]:
        """
        Check that all answers are finite and evaluate without NaN/inf.
        Do not compute expected values.
        """
        t = sp.symbols('t', real=True)
        r = sp.symbols('r', positive=True)
        t_test = math.pi/(4*self.omega) if self.omega else 0
        test_r = self.R/2 if self.R else 0.1
        anomalies = []
        # Check E_z
        if self.E_z is not None:
            E_expr = self._parse_sympy(self.E_z)
            if E_expr is not None:
                try:
                    val = float(E_expr.subs(t, t_test))
                    if math.isnan(val) or math.isinf(val):
                        anomalies.append("E_z is NaN or inf")
                except:
                    anomalies.append("E_z evaluation error")
        else:
            anomalies.append("E_z missing")
        # Check sigma_f
        if self.sigma_f is not None:
            sig_expr = self._parse_sympy(self.sigma_f)
            if sig_expr is not None:
                try:
                    val = float(sig_expr.subs(t, t_test))
                    if math.isnan(val) or math.isinf(val):
                        anomalies.append("sigma_f is NaN or inf")
                except:
                    anomalies.append("sigma_f evaluation error")
        # Check B_phi
        if self.B_phi is not None:
            B_expr = self._parse_sympy(self.B_phi)
            if B_expr is not None:
                try:
                    val = float(B_expr.subs({r: test_r, t: t_test}))
                    if math.isnan(val) or math.isinf(val):
                        anomalies.append("B_phi is NaN or inf")
                except:
                    anomalies.append("B_phi evaluation error")
        # Check flux
        if self.flux is not None:
            flux_expr = self._parse_sympy(self.flux)
            if flux_expr is not None:
                try:
                    val = float(flux_expr.subs(t, t_test))
                    if math.isnan(val) or math.isinf(val):
                        anomalies.append("flux is NaN or inf")
                except:
                    anomalies.append("flux evaluation error")
        if anomalies:
            return False, " ; ".join(anomalies)
        return True, "All answers are finite and evaluable"

    # ---------------------------------------------------------------
    # Main runner
    # ---------------------------------------------------------------
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
                passed, msg = False, f"CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)