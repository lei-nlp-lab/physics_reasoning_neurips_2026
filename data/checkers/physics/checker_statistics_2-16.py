import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        res = getattr(solution_instance, 'results', {})
        self.n_He = res.get('n_He', None)
        self.lambda_He = res.get('lambda_He', None)
        self.delta_He = res.get('delta_He', None)
        self.lambda_e_Cu = res.get('lambda_e_Cu', None)
        self.delta_e_Cu = res.get('delta_e_Cu', None)
        self.N_c = res.get('N_c', None)
        self.N_v = res.get('N_v', None)
        self.n_i = res.get('n_i', None)
        self.delta_e_Ge = res.get('delta_e_Ge', None)

        givens = getattr(solution_instance, 'givens', {})
        self.T = givens.get('T', None)
        self.P = givens.get('P', None)
        self.n_e_Cu = givens.get('n_e_Cu', None)
        self.E_g = givens.get('E_g', None)
        self.m_He_u = givens.get('m_He_u', None)
        self.m_e_eff_factor = givens.get('m_e_eff', None)
        self.m_h_eff_factor = givens.get('m_h_eff', None)

        self.solution_instance = solution_instance

        # Physical constants
        self.h = 6.62607015e-34
        self.kB = 1.380649e-23
        self.m_e = 9.1093837e-31
        self.amu = 1.66053906660e-27
        self.eV = 1.602176634e-19

        # Derived constants
        self.m_He = 4.0026 * self.amu
        self.m_e_eff = 0.12 * self.m_e if self.m_e_eff_factor is None else self.m_e_eff_factor * self.m_e
        self.m_h_eff = 0.3 * self.m_e if self.m_h_eff_factor is None else self.m_h_eff_factor * self.m_e
        self.Eg_J = 1.0 * self.eV if self.E_g is None else self.E_g * self.eV

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def _get_T(self):
        return self.T if self.T is not None else 300.0

    def _get_P(self):
        return self.P if self.P is not None else 101325.0

    def _get_n_e_Cu(self):
        return self.n_e_Cu if self.n_e_Cu is not None else 8.5e28

    def _get_m_He(self):
        return self.m_He

    def _get_m_e_eff(self):
        return self.m_e_eff

    def _get_m_h_eff(self):
        return self.m_h_eff

    def _get_Eg_J(self):
        return self.Eg_J

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_n_He(self):
        if self.n_He is None:
            return (False, "n_He missing")
        if not isinstance(self.n_He, (float, int)):
            return (False, f"n_He type {type(self.n_He)}")
        return (True, "n_He present and numeric")

    def test_interface_lambda_He(self):
        if self.lambda_He is None:
            return (False, "lambda_He missing")
        if not isinstance(self.lambda_He, (float, int)):
            return (False, f"lambda_He type {type(self.lambda_He)}")
        return (True, "lambda_He present and numeric")

    def test_interface_delta_He(self):
        if self.delta_He is None:
            return (False, "delta_He missing")
        if not isinstance(self.delta_He, (float, int)):
            return (False, f"delta_He type {type(self.delta_He)}")
        return (True, "delta_He present and numeric")

    def test_interface_lambda_e_Cu(self):
        if self.lambda_e_Cu is None:
            return (False, "lambda_e_Cu missing")
        if not isinstance(self.lambda_e_Cu, (float, int)):
            return (False, f"lambda_e_Cu type {type(self.lambda_e_Cu)}")
        return (True, "lambda_e_Cu present and numeric")

    def test_interface_delta_e_Cu(self):
        if self.delta_e_Cu is None:
            return (False, "delta_e_Cu missing")
        if not isinstance(self.delta_e_Cu, (float, int)):
            return (False, f"delta_e_Cu type {type(self.delta_e_Cu)}")
        return (True, "delta_e_Cu present and numeric")

    def test_interface_N_c(self):
        if self.N_c is None:
            return (False, "N_c missing")
        if not isinstance(self.N_c, (float, int)):
            return (False, f"N_c type {type(self.N_c)}")
        return (True, "N_c present and numeric")

    def test_interface_N_v(self):
        if self.N_v is None:
            return (False, "N_v missing")
        if not isinstance(self.N_v, (float, int)):
            return (False, f"N_v type {type(self.N_v)}")
        return (True, "N_v present and numeric")

    def test_interface_n_i(self):
        if self.n_i is None:
            return (False, "n_i missing")
        if not isinstance(self.n_i, (float, int)):
            return (False, f"n_i type {type(self.n_i)}")
        return (True, "n_i present and numeric")

    def test_interface_delta_e_Ge(self):
        if self.delta_e_Ge is None:
            return (False, "delta_e_Ge missing")
        if not isinstance(self.delta_e_Ge, (float, int)):
            return (False, f"delta_e_Ge type {type(self.delta_e_Ge)}")
        return (True, "delta_e_Ge present and numeric")

    # ------------------------------------------------------------------
    # Invariant tests (fuzzy numerical checks)
    # ------------------------------------------------------------------
    def test_invariant_ideal_gas_n_He(self):
        if self.n_He is None:
            return (False, "n_He missing")
        T = self._get_T()
        P = self._get_P()
        expected = P / (self.kB * T)
        if math.isclose(self.n_He, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"n_He ≈ P/(kT) = {expected:.3e}")
        return (False, f"n_He = {self.n_He:.3e}, expected {expected:.3e}")

    def test_invariant_lambda_He(self):
        if self.lambda_He is None:
            return (False, "lambda_He missing")
        T = self._get_T()
        m = self._get_m_He()
        expected = self.h / math.sqrt(2 * math.pi * m * self.kB * T)
        if math.isclose(self.lambda_He, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"lambda_He ≈ h/√(2πmkT) = {expected:.3e}")
        return (False, f"lambda_He = {self.lambda_He:.3e}, expected {expected:.3e}")

    def test_invariant_delta_He_product(self):
        if None in (self.delta_He, self.n_He, self.lambda_He):
            return (False, "delta_He, n_He or lambda_He missing")
        computed = self.n_He * self.lambda_He**3
        if math.isclose(self.delta_He, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_He = n_He * lambda_He^3 = {computed:.3e}")
        return (False, f"delta_He = {self.delta_He:.3e}, computed {computed:.3e}")

    def test_invariant_lambda_e_Cu(self):
        if self.lambda_e_Cu is None:
            return (False, "lambda_e_Cu missing")
        T = self._get_T()
        expected = self.h / math.sqrt(2 * math.pi * self.m_e * self.kB * T)
        if math.isclose(self.lambda_e_Cu, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"lambda_e_Cu ≈ h/√(2π m_e kT) = {expected:.3e}")
        return (False, f"lambda_e_Cu = {self.lambda_e_Cu:.3e}, expected {expected:.3e}")

    def test_invariant_delta_e_Cu_product(self):
        if None in (self.delta_e_Cu, self.lambda_e_Cu):
            return (False, "delta_e_Cu or lambda_e_Cu missing")
        n = self._get_n_e_Cu()
        computed = n * self.lambda_e_Cu**3
        if math.isclose(self.delta_e_Cu, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_e_Cu = n_e_Cu * lambda_e_Cu^3 = {computed:.3e}")
        return (False, f"delta_e_Cu = {self.delta_e_Cu:.3e}, computed {computed:.3e}")

    def test_invariant_semiconductor_N_c(self):
        if self.N_c is None:
            return (False, "N_c missing")
        T = self._get_T()
        m_eff = self._get_m_e_eff()
        expected = 2 * (2 * math.pi * m_eff * self.kB * T / self.h**2)**(3/2)
        if math.isclose(self.N_c, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_c ≈ 2(2π m_e* kT/h^2)^{3/2} = {expected:.3e}")
        return (False, f"N_c = {self.N_c:.3e}, expected {expected:.3e}")

    def test_invariant_semiconductor_N_v(self):
        if self.N_v is None:
            return (False, "N_v missing")
        T = self._get_T()
        m_eff = self._get_m_h_eff()
        expected = 2 * (2 * math.pi * m_eff * self.kB * T / self.h**2)**(3/2)
        if math.isclose(self.N_v, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"N_v ≈ 2(2π m_h* kT/h^2)^{3/2} = {expected:.3e}")
        return (False, f"N_v = {self.N_v:.3e}, expected {expected:.3e}")

    def test_invariant_semiconductor_n_i(self):
        if None in (self.n_i, self.N_c, self.N_v):
            return (False, "n_i, N_c or N_v missing")
        T = self._get_T()
        Eg = self._get_Eg_J()
        expected = math.sqrt(self.N_c * self.N_v) * math.exp(-Eg / (2 * self.kB * T))
        if math.isclose(self.n_i, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"n_i ≈ sqrt(N_c N_v) exp(-Eg/(2kT)) = {expected:.3e}")
        return (False, f"n_i = {self.n_i:.3e}, expected {expected:.3e}")

    def test_invariant_delta_e_Ge_product(self):
        if None in (self.delta_e_Ge, self.n_i):
            return (False, "delta_e_Ge or n_i missing")
        T = self._get_T()
        m_eff = self._get_m_e_eff()
        lam = self.h / math.sqrt(2 * math.pi * m_eff * self.kB * T)
        computed = self.n_i * lam**3
        if math.isclose(self.delta_e_Ge, computed, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"delta_e_Ge = n_i * lambda_e_Ge^3 = {computed:.3e}")
        return (False, f"delta_e_Ge = {self.delta_e_Ge:.3e}, computed {computed:.3e}")

    # ------------------------------------------------------------------
    # Modeling tests (quality of approximations)
    # ------------------------------------------------------------------
    def test_modeling_He_classical(self):
        if self.delta_He is None:
            return (False, "delta_He missing")
        if self.delta_He < 1e-5:
            return (True, f"delta_He = {self.delta_He:.3e} << 1, classical OK")
        return (False, f"delta_He = {self.delta_He:.3e}, may not be classical")

    def test_modeling_Cu_degenerate(self):
        if self.delta_e_Cu is None:
            return (False, "delta_e_Cu missing")
        if self.delta_e_Cu > 100:
            return (True, f"delta_e_Cu = {self.delta_e_Cu:.3e} >> 1, degenerate")
        return (False, f"delta_e_Cu = {self.delta_e_Cu:.3e}, not clearly degenerate")

    def test_modeling_Ge_nondegenerate(self):
        if self.delta_e_Ge is None:
            return (False, "delta_e_Ge missing")
        if self.delta_e_Ge < 1:
            return (True, f"delta_e_Ge = {self.delta_e_Ge:.3e} < 1, non-degenerate")
        return (False, f"delta_e_Ge = {self.delta_e_Ge:.3e} >= 1, possibly degenerate")

    def test_modeling_Ge_n_i_vs_Nc_Nv(self):
        if None in (self.n_i, self.N_c, self.N_v):
            return (False, "n_i, N_c or N_v missing")
        threshold = 0.1  # n_i should be much less
        if self.n_i < threshold * self.N_c and self.n_i < threshold * self.N_v:
            return (True, f"n_i = {self.n_i:.3e} << N_c,N_v")
        return (False, f"n_i = {self.n_i:.3e}, N_c = {self.N_c:.3e}, N_v = {self.N_v:.3e}")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb T and observe delta)
    # ------------------------------------------------------------------
    def test_monotonicity_He_delta_vs_T(self):
        T = self._get_T()
        P = self._get_P()
        m = self._get_m_He()
        # baseline at T
        n0 = P / (self.kB * T)
        lam0 = self.h / math.sqrt(2 * math.pi * m * self.kB * T)
        delta0 = n0 * lam0**3

        T_pert = T * 1.1
        n1 = P / (self.kB * T_pert)
        lam1 = self.h / math.sqrt(2 * math.pi * m * self.kB * T_pert)
        delta1 = n1 * lam1**3

        if delta1 < delta0:
            return (True, f"T↑ {T_pert:.1f}K -> delta ↓ {delta1:.3e} < {delta0:.3e}")
        return (False, f"Monotonicity broken: baseline {delta0:.3e}, perturbed {delta1:.3e}")

    def test_monotonicity_Cu_delta_vs_T(self):
        T = self._get_T()
        n = self._get_n_e_Cu()
        lam0 = self.h / math.sqrt(2 * math.pi * self.m_e * self.kB * T)
        delta0 = n * lam0**3

        T_pert = T * 1.1
        lam1 = self.h / math.sqrt(2 * math.pi * self.m_e * self.kB * T_pert)
        delta1 = n * lam1**3

        if delta1 < delta0:
            return (True, f"T↑ {T_pert:.1f}K -> delta ↓ {delta1:.3e} < {delta0:.3e}")
        return (False, f"Monotonicity broken: baseline {delta0:.3e}, perturbed {delta1:.3e}")

    def test_monotonicity_Ge_delta_vs_T(self):
        """delta_e_Ge should increase with T because n_i increases exponentially."""
        T = self._get_T()
        Eg = self._get_Eg_J()
        m_e_eff = self._get_m_e_eff()
        m_h_eff = self._get_m_h_eff()

        def compute_delta(T_val):
            Nc = 2 * (2 * math.pi * m_e_eff * self.kB * T_val / self.h**2)**(3/2)
            Nv = 2 * (2 * math.pi * m_h_eff * self.kB * T_val / self.h**2)**(3/2)
            n_i = math.sqrt(Nc * Nv) * math.exp(-Eg / (2 * self.kB * T_val))
            lam = self.h / math.sqrt(2 * math.pi * m_e_eff * self.kB * T_val)
            return n_i * lam**3

        delta0 = compute_delta(T)
        T_pert = T * 1.1
        delta1 = compute_delta(T_pert)

        if delta1 > delta0:
            return (True, f"T↑ {T_pert:.1f}K -> delta ↑ {delta1:.3e} > {delta0:.3e}")
        return (False, f"Monotonicity broken: baseline {delta0:.3e}, perturbed {delta1:.3e}")

    # ------------------------------------------------------------------
    # Weak answer sanity test (one only)
    # ------------------------------------------------------------------
    def test_answer_sanity_delta_sign(self):
        """All deltas should be positive (finite, real)."""
        deltas = [self.delta_He, self.delta_e_Cu, self.delta_e_Ge]
        names = ["delta_He", "delta_e_Cu", "delta_e_Ge"]
        for d, name in zip(deltas, names):
            if d is not None:
                if not (isinstance(d, (float, int)) and d > 0):
                    return (False, f"{name} = {d} not positive")
        return (True, "All provided deltas are positive")

    # ------------------------------------------------------------------
    # Main runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)