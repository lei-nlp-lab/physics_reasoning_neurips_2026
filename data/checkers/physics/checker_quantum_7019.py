import math
import sympy as sp
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # extract givens and results safely
        self.givens = getattr(solution_instance, 'givens', {})
        self.results = getattr(solution_instance, 'results', {})

        # extract required quantities (use None if missing)
        self.r = self.givens.get('r', None)          # expected in Å or meters
        self.N = self.givens.get('N', None)          # number of electrons (18)
        self.hbar = self.givens.get('hbar', None)
        self.m_e = self.givens.get('m_e', None)
        self.c = self.givens.get('c', None)
        self.h = self.givens.get('h', None)

        # results emitted by solver
        self.eigenfunction = self.results.get('eigenfunction', None)
        self.occupancy = self.results.get('occupancy', None)
        self.Delta_E = self.results.get('Delta_E', None)
        self.lambda_ = self.results.get('lambda', None)
        self.lambda_num = self.results.get('lambda_num', None)

        # physical constants (used when givens missing)
        self._hbar_SI = 1.054571817e-34
        self._m_e_SI = 9.1093837015e-31
        self._c_SI = 2.99792458e8
        self._h_SI = 6.62607015e-34
        self._pi = math.pi

    # -----------------------------------------------------------------
    # Interface tests
    # -----------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys exist and are not None."""
        required = ['eigenfunction', 'occupancy', 'Delta_E', 'lambda', 'lambda_num']
        missing = [k for k in required if getattr(self, k, None) is None]
        if missing:
            return (False, f"Missing keys: {missing}")
        return (True, "All required keys present")

    def test_interface_occupancy_type(self):
        """Occupancy must be a dict mapping int or numeric to int."""
        if self.occupancy is None:
            return (False, "SKIP: occupancy not available")
        if not isinstance(self.occupancy, dict):
            return (False, f"occupancy is not a dict: {type(self.occupancy)}")
        for k, v in self.occupancy.items():
            # keys can be int or sympy Integer
            if not isinstance(k, (int, float, sp.Integer, sp.Float)):
                return (False, f"key {k} has invalid type {type(k)}")
            if not isinstance(v, (int, sp.Integer)):
                return (False, f"value {v} for key {k} is not int")
        return (True, "occupancy dict is properly typed")

    def test_interface_eigenfunction_type(self):
        """Eigenfunction should be a string or sympy expression."""
        if self.eigenfunction is None:
            return (False, "SKIP: eigenfunction not available")
        if not isinstance(self.eigenfunction, (str, sp.Expr)):
            return (False, f"eigenfunction type {type(self.eigenfunction)} not acceptable")
        return (True, "eigenfunction is string or sympy expression")

    # -----------------------------------------------------------------
    # Invariant / theorem-based tests
    # -----------------------------------------------------------------
    def test_invariant_eigenfunction_normalization(self):
        """Check that eigenfunction (if sympy) is normalized on [0,2π]."""
        if self.eigenfunction is None:
            return (False, "SKIP: eigenfunction not available")
        # only check if it's a sympy expression
        if not isinstance(self.eigenfunction, sp.Expr):
            return (False, "SKIP: eigenfunction not sympy, cannot integrate")
        theta = sp.Symbol('theta', real=True)
        try:
            psi = self.eigenfunction
            # compute integral of |psi|^2 from 0 to 2π
            integrand = sp.conjugate(psi) * psi
            integral = sp.integrate(integrand, (theta, 0, 2*sp.pi))
            # should be 1 (real)
            if integral.is_number:
                if math.isclose(float(integral.evalf()), 1.0, rel_tol=1e-2, abs_tol=1e-2):
                    return (True, "Normalization integral ≈ 1")
                else:
                    return (False, f"Normalization integral = {integral}, not 1")
            else:
                # symbolic result: try to simplify to 1
                if sp.simplify(integral - 1) == 0:
                    return (True, "Normalization symbolic = 1")
                else:
                    return (False, f"Normalization integral = {integral}, cannot verify exactly 1")
        except Exception as e:
            return (False, f"Normalization test crashed: {e}")

    def test_invariant_energy_formula(self):
        """Check that Delta_E matches theoretical formula ΔE = 9 ħ²/(2 m_e r²)."""
        if self.Delta_E is None:
            return (False, "SKIP: Delta_E not available")
        r_val = self._get_r_in_meters()
        if r_val is None:
            return (False, "SKIP: cannot determine radius for energy check")
        hbar = self.hbar if self.hbar is not None else self._hbar_SI
        m_e = self.m_e if self.m_e is not None else self._m_e_SI
        expected = 9 * hbar**2 / (2 * m_e * r_val**2)
        Delta_E_val = self._to_float(self.Delta_E)
        if Delta_E_val is None:
            return (False, "SKIP: Delta_E cannot be converted to float")
        if math.isclose(Delta_E_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ΔE ≈ {expected:.2e} J (expected 9ħ²/(2m_er²))")
        else:
            return (False, f"ΔE = {Delta_E_val:.2e} J, expected {expected:.2e} J")

    def test_invariant_wavelength_formula(self):
        """Check that lambda equation matches λ = hc/ΔE."""
        if self.lambda_ is None or self.Delta_E is None:
            return (False, "SKIP: lambda or Delta_E not available")
        h = self.h if self.h is not None else self._h_SI
        c = self.c if self.c is not None else self._c_SI
        Delta_E_val = self._to_float(self.Delta_E)
        if Delta_E_val is None or Delta_E_val == 0:
            return (False, "SKIP: Delta_E cannot be converted to float or zero")
        expected_lambda = h * c / Delta_E_val
        lambda_val = self._to_float(self.lambda_)
        if lambda_val is None:
            return (False, "SKIP: lambda cannot be converted to float")
        if math.isclose(lambda_val, expected_lambda, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"λ ≈ {lambda_val:.3e} m matches hc/ΔE")
        else:
            return (False, f"λ = {lambda_val:.3e} m, expected {expected_lambda:.3e} m from hc/ΔE")

    # -----------------------------------------------------------------
    # Modeling / trace tests (based on explanation steps)
    # -----------------------------------------------------------------
    def test_modeling_occupancy_total_electrons(self):
        """Total electrons in occupancy dict should be 18."""
        if self.occupancy is None:
            return (False, "SKIP: occupancy not available")
        total = sum(self.occupancy.values())
        if total != 18:
            return (False, f"Total electrons = {total}, expected 18")
        return (True, "Total occupancy sums to 18")

    def test_modeling_occupancy_pauli_fill(self):
        """Check that occupancy follows Pauli: m=0 has 2, each |m|>0 has ≤4."""
        if self.occupancy is None:
            return (False, "SKIP: occupancy not available")
        for m, count in self.occupancy.items():
            m_int = int(m)  # assume integer
            if m_int == 0:
                if count != 2:
                    return (False, f"m=0 occupancy = {count}, should be 2")
            else:
                if count not in [0, 2, 4]:
                    return (False, f"m={m} occupancy = {count}, should be 0,2,4")
        # additionally, highest occupied should be m=4 with 4 electrons
        max_occupied = max(self.occupancy.keys())
        if max_occupied != 4 or self.occupancy.get(4, 0) != 4:
            return (False, f"Highest occupied should be m=4 with 4, got {max_occupied} with {self.occupancy.get(4,0)}")
        return (True, "Occupancy follows Pauli principle")

    def test_trace_excitation_transition(self):
        """Check that Delta_E corresponds to m=4 → m=5 transition."""
        if self.Delta_E is None:
            return (False, "SKIP: Delta_E not available")
        r_val = self._get_r_in_meters()
        if r_val is None:
            return (False, "SKIP: radius needed for transition check")
        hbar = self.hbar if self.hbar is not None else self._hbar_SI
        m_e = self.m_e if self.m_e is not None else self._m_e_SI
        E_4 = hbar**2 * 16 / (2 * m_e * r_val**2)
        E_5 = hbar**2 * 25 / (2 * m_e * r_val**2)
        expected = E_5 - E_4
        Delta_E_val = self._to_float(self.Delta_E)
        if Delta_E_val is None:
            return (False, "SKIP: Delta_E cannot be converted")
        if math.isclose(Delta_E_val, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"ΔE corresponds to |m|=4→5 transition")
        else:
            return (False, f"ΔE = {Delta_E_val:.2e}, expected {expected:.2e} from 4→5")

    # -----------------------------------------------------------------
    # Monotonicity tests
    # -----------------------------------------------------------------
    def test_monotonicity_wavelength_vs_radius(self):
        """Increasing radius should increase wavelength (λ ∝ r²)."""
        if self.lambda_num is None:
            # fallback to lambda_ if available
            if self.lambda_ is None:
                return (False, "SKIP: no wavelength available for monotonicity")
            baseline = self._to_float(self.lambda_)
        else:
            baseline = self._to_float(self.lambda_num)
        if baseline is None or baseline <= 0:
            return (False, "SKIP: baseline wavelength not positive float")

        # perturb radius by +2% and estimate new wavelength
        r_val = self._get_r_in_meters()
        if r_val is None:
            return (False, "SKIP: radius needed for perturbation")

        # compute expected scaling: λ_new ≈ baseline * (r_new/r_baseline)^2
        r_perturbed = r_val * 1.02
        # We need to recalculate Delta_E and lambda using formulas
        hbar = self.hbar if self.hbar is not None else self._hbar_SI
        m_e = self.m_e if self.m_e is not None else self._m_e_SI
        h = self.h if self.h is not None else self._h_SI
        c = self.c if self.c is not None else self._c_SI

        Delta_E_new = 9 * hbar**2 / (2 * m_e * r_perturbed**2)
        lambda_new = h * c / Delta_E_new

        if not math.isclose(lambda_new / baseline, 1.02**2, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"Wavelength scaling deviates: λ_new/λ_base = {lambda_new/baseline:.4f}, expected {1.02**2:.4f}")
        return (True, f"Wavelength increases with radius as expected (λ∝r²)")

    # -----------------------------------------------------------------
    # Answer sanity (weak, secondary)
    # -----------------------------------------------------------------
    def test_answer_sanity_wavelength_range(self):
        """Check that lambda_num is positive and roughly in visible range (4000-7000 Å)."""
        if self.lambda_num is None:
            return (False, "SKIP: lambda_num not available")
        lam = self._to_float(self.lambda_num)
        if lam is None:
            return (False, "SKIP: lambda_num cannot be converted to float")
        if lam <= 0:
            return (False, f"lambda_num = {lam} <= 0")
        # convert to Angstrom if needed (assume meters or Angstrom)
        if lam > 1e-6:  # likely in meters
            lam_A = lam * 1e10
        else:
            lam_A = lam
        if 4000 <= lam_A <= 7000:
            return (True, f"λ ≈ {lam_A:.0f} Å in visible range")
        else:
            return (False, f"λ ≈ {lam_A:.0f} Å outside visible range (4000-7000 Å)")

    # -----------------------------------------------------------------
    # Helper methods
    # -----------------------------------------------------------------
    def _get_r_in_meters(self):
        """Return radius in meters, converting if necessary."""
        if self.r is None:
            return None
        r_val = self._to_float(self.r)
        if r_val is None:
            return None
        # assume it's in Å if value is around 4 (4 A) or meters if ~4e-10
        if r_val > 1e-9:  # > 1 nm => likely Å
            return r_val * 1e-10
        else:
            return r_val

    def _to_float(self, val):
        """Convert sympy number or numeric to float, else None."""
        if val is None:
            return None
        try:
            # if sympy, use evalf()
            if hasattr(val, 'evalf'):
                return float(val.evalf())
            return float(val)
        except (TypeError, ValueError):
            return None

    # -----------------------------------------------------------------
    # Main runner
    # -----------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        # discover all test methods
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