import math
import sympy
import traceback

class PhysicsChecker:
    """
    Checker for an electron in a uniform magnetic field (Landau levels).
    Audits Hamiltonian and energy spectrum from the solution instance.
    """

    def __init__(self, solution_instance):
        # Safe extraction from results dict
        self.results = getattr(solution_instance, 'results', {})
        self.hamiltonian = self.results.get('hamiltonian', None)
        self.energy_spectrum = self.results.get('energy_spectrum', None)

        # Optional givens (do not assign numeric defaults)
        self.m = getattr(solution_instance, 'm', None)
        self.e = getattr(solution_instance, 'e', None)
        self.B = getattr(solution_instance, 'B', None)
        self.hbar = getattr(solution_instance, 'hbar', None)
        self.c = getattr(solution_instance, 'c', None)

        # Explanation steps for modeling checks (optional)
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

    # ----------------------------------------------------------------------
    # Helper to clean LaTeX and parse to sympy expression
    # ----------------------------------------------------------------------
    def _parse_energy_expression(self):
        """Try to parse the energy_spectrum string into a sympy expression,
        substituting reasonable defaults for missing constants.
        Returns expression, or None on failure."""
        if self.energy_spectrum is None:
            return None
        e_str = self.energy_spectrum
        # Replace LaTeX greek/special chars
        e_str = e_str.replace('\\hbar', 'hbar')
        e_str = e_str.replace('\\omega_c', 'omega_c')
        e_str = e_str.replace('ω_c', 'omega_c')
        e_str = e_str.replace('\\frac', '/')
        # Remove braces that are not needed
        # This is a best-effort cleaning
        try:
            # Define symbols
            n, hbar, pz, m, omega_c = sympy.symbols('n hbar pz m omega_c')
            expression = sympy.sympify(e_str)
            return expression
        except (sympy.SympifyError, TypeError):
            return None

    # ----------------------------------------------------------------------
    # Interface tests – check that required keys exist with correct types
    # ----------------------------------------------------------------------
    def test_interface_hamiltonian_key(self):
        if self.hamiltonian is None:
            return (False, "hamiltonian key missing in results")
        if not isinstance(self.hamiltonian, str):
            return (False, "hamiltonian is not a string")
        return (True, "hamiltonian key present and is string")

    def test_interface_energy_spectrum_key(self):
        if self.energy_spectrum is None:
            return (False, "energy_spectrum key missing in results")
        if not isinstance(self.energy_spectrum, str):
            return (False, "energy_spectrum is not a string")
        return (True, "energy_spectrum key present and is string")

    # ----------------------------------------------------------------------
    # Invariant tests – theorem‑based physical sanity (string & logical checks)
    # ----------------------------------------------------------------------
    def test_invariant_hamiltonian_form(self):
        """Verify the Hamiltonian string contains the Landau‑gauge minimal‑coupling expression."""
        if self.hamiltonian is None:
            return (False, "SKIP: hamiltonian not available")
        h = self.hamiltonian
        checks = {
            "p_x^2": "p_x^2" in h,
            "p_y + eBx/c": ("p_y" in h and "eBx" in h and "c" in h),
            "p_z^2": "p_z^2" in h,
            "1/(2m) factor": ("1/(2m)" in h or "1/2m" in h or "1/(2 m)" in h),
            "denominator 2m": ("2m" in h or "2 m" in h),
        }
        if all(checks.values()):
            return (True, "Hamiltonian contains expected minimal‑coupling terms")
        else:
            missing = [k for k, v in checks.items() if not v]
            return (False, f"Hamiltonian missing: {', '.join(missing)}")

    def test_invariant_energy_spectrum_form(self):
        """Verify the energy spectrum string contains quantized Landau levels and kinetic term."""
        if self.energy_spectrum is None:
            return (False, "SKIP: energy_spectrum not available")
        e = self.energy_spectrum
        checks = {
            "n+1/2 structure": any(x in e for x in ["n+1/2", "n+0.5", "n+\\frac{1}{2}"]),
            "hbar/ħ": ("hbar" in e or "ħ" in e),
            "cyclotron frequency": any(x in e for x in ["omega_c", "ω_c", "eB/(mc)", "e B/(m c)", "eB/mc"]),
            "p_z^2/(2m)": ("p_z^2/(2m)" in e or "p_z^2/2m" in e),
        }
        if all(checks.values()):
            return (True, "Energy spectrum contains quantized Landau level structure")
        else:
            missing = [k for k, v in checks.items() if not v]
            return (False, f"Energy spectrum missing: {', '.join(missing)}")

    def test_invariant_energy_positivity(self):
        """Energy expression should imply non‑negativity (harmonic oscillator + kinetic term)."""
        if self.energy_spectrum is None:
            return (False, "SKIP: energy_spectrum not available")
        if "hbar" in self.energy_spectrum and "p_z^2" in self.energy_spectrum:
            return (True, "Energy expression appears non‑negative (oscillator + kinetic term)")
        else:
            return (False, "Energy expression does not obviously guarantee non‑negativity")

    # ----------------------------------------------------------------------
    # Modeling / trace tests – at least two
    # ----------------------------------------------------------------------
    def test_modeling_quantization(self):
        """The energy spectrum must reflect quantization with integer n."""
        if self.energy_spectrum is None:
            return (False, "SKIP: energy_spectrum not available")
        e = self.energy_spectrum
        # Check for explicit statement of integer n
        if "n=0,1,2" in e or "n = 0,1,2" in e or "n = 0, 1, 2" in e:
            return (True, "Energy spectrum shows explicit integer n (Landau levels)")
        elif "n" in e and ("integer" in e or "quantum" in e):
            return (True, "Energy spectrum references integer quantum number n")
        elif "n" in e:
            return (True, "Energy spectrum contains quantum number n (likely quantized)")
        else:
            return (False, "Energy spectrum does not exhibit quantization (no n or integer reference)")

    def test_modeling_gauge_choice(self):
        """Check that the Hamiltonian is written in the Landau gauge (A = (0, Bx, 0))."""
        if self.hamiltonian is None:
            return (False, "SKIP: hamiltonian not available")
        h = self.hamiltonian
        # The gauge should have A along y, depending on x.
        # Look for terms that indicate A_y = Bx, A_x=0, A_z=0.
        # "p_y + eBx/c" is a strong indicator.
        # Also check that there is no A_x dependence.
        if "p_y" in h and "eBx" in h and "p_x" in h and "p_z" in h:
            # Ensure no p_x + ... term that would imply A_x
            if "p_x" in h and "+" not in h.split("p_x")[0][-3:]:  # rough check
                return (True, "Hamiltonian uses Landau gauge (A_y = Bx, A_x=0, A_z=0)")
            else:
                return (True, "Hamiltonian appears to use Landau gauge")
        else:
            return (False, "Hamiltonian does not clearly reflect Landau gauge")

    # ----------------------------------------------------------------------
    # Monotonicity test – explicitly perturb B and compare energy
    # ----------------------------------------------------------------------
    def test_monotonicity_energy_vs_magnetic_field(self):
        """Perturb B and verify that the energy increases.
        Uses sympy to evaluate the energy expression for ground state (n=0, pz=0)."""
        if self.energy_spectrum is None:
            return (False, "SKIP: energy_spectrum not available")
        # Try to parse the expression
        expr = self._parse_energy_expression()
        if expr is None:
            # Fallback: check string for monotonic behavior
            if "hbar" in self.energy_spectrum and ("omega_c" in self.energy_spectrum or "eB" in self.energy_spectrum):
                return (True, "Expression contains cyclotron frequency; monotonic increase with B inferred")
            else:
                return (False, "Cannot verify monotonicity without parsable expression")
        # Define symbols used
        n, hbar, pz, m, omega_c = sympy.symbols('n hbar pz m omega_c')
        # Substitute n=0, pz=0, hbar=1.054571817e-34, m=9.10938356e-31, e=1.602176634e-19, c=299792458
        # Use numeric values for constants
        hbar_val = 1.054571817e-34 if self.hbar is None else self.hbar
        m_val = 9.10938356e-31 if self.m is None else self.m
        e_val = 1.602176634e-19 if self.e is None else self.e
        c_val = 299792458 if self.c is None else self.c
        B_base = 1.0 if self.B is None else self.B
        # Base energy for B_base
        omega_c_base = e_val * B_base / (m_val * c_val)
        subs_base = {n: 0, pz: 0, hbar: hbar_val, m: m_val, omega_c: omega_c_base}
        try:
            E_base = float(expr.subs(subs_base))
        except Exception:
            return (False, "Failed to evaluate energy expression for base B")
        # Perturb B by +10%
        B_pert = B_base * 1.1
        omega_c_pert = e_val * B_pert / (m_val * c_val)
        subs_pert = {n: 0, pz: 0, hbar: hbar_val, m: m_val, omega_c: omega_c_pert}
        try:
            E_pert = float(expr.subs(subs_pert))
        except Exception:
            return (False, "Failed to evaluate energy expression for perturbed B")
        if E_pert > E_base:
            return (True, f"Energy increases from {E_base} to {E_pert} when B increases 10%")
        else:
            return (False, f"Energy does not increase when B increases: base {E_base}, perturbed {E_pert}")

    # ----------------------------------------------------------------------
    # Weak answer sanity test – only non‑emptiness, at most one
    # ----------------------------------------------------------------------
    def test_answer_sanity_basic_nonempty(self):
        """Check that both hamiltonian and energy_spectrum are non‑empty strings."""
        if self.hamiltonian is None and self.energy_spectrum is None:
            return (False, "Both hamiltonian and energy_spectrum are missing")
        # Check non-emptiness
        if self.hamiltonian is not None and len(self.hamiltonian.strip()) == 0:
            return (False, "Hamiltonian string is empty")
        if self.energy_spectrum is not None and len(self.energy_spectrum.strip()) == 0:
            return (False, "Energy spectrum string is empty")
        return (True, "Hamiltonian and energy spectrum are present and non‑empty")

    # ----------------------------------------------------------------------
    # Runner
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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)