import sympy as sp
import math
import traceback

class PhysicsChecker:
    """
    Audits a solution for the quantization experiments (photoelectric effect and blackbody radiation).
    """

    def __init__(self, solution_instance):
        # Safely extract main attributes (no numeric defaults)
        self.results = getattr(solution_instance, 'results', None) or {}
        self.givens = getattr(solution_instance, 'givens', None)
        if self.givens is None:
            self.givens = {}
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)

        # Extract important result quantities
        self.K_max = self.results.get('K_max', None)
        self.hf = self.results.get('hf', None)
        self.phi = self.results.get('phi', None)
        self.u_f_T = self.results.get('u_f_T', None)

        # Extract typical givens (may be missing)
        self.h = self.givens.get('h', None) if isinstance(self.givens, dict) else None
        self.c = self.givens.get('c', None) if isinstance(self.givens, dict) else None
        self.k = self.givens.get('k', None) if isinstance(self.givens, dict) else None
        self.T = self.givens.get('T', None) if isinstance(self.givens, dict) else None
        self.f = self.givens.get('f', None) if isinstance(self.givens, dict) else None

    def _to_float(self, value):
        """Try to convert a value to a float; return None on failure."""
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all four must‑emit keys exist in results."""
        missing = []
        for key in ['K_max', 'hf', 'phi', 'u_f_T']:
            if key not in self.results:
                missing.append(key)
        if missing:
            return (False, f"Missing result keys: {missing}")
        return (True, "All required keys present.")

    # ------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # ------------------------------------------------------------------
    def test_invariant_Kmax_equation(self):
        """Verify Einstein’s photoelectric equation: K_max = hf - phi."""
        K = self._to_float(self.K_max)
        hf = self._to_float(self.hf)
        phi = self._to_float(self.phi)
        if K is None or hf is None or phi is None:
            return (False, "SKIP: missing K_max, hf, or phi raw values")
        expected = hf - phi
        if math.isclose(K, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"K_max={K} matches hf - phi = {expected}")
        else:
            return (False, f"K_max={K} differs from hf - phi = {expected}")

    def test_invariant_u_f_T_positive(self):
        """Spectral energy density must be positive for T>0 and f>0."""
        u = self._to_float(self.u_f_T)
        if u is None:
            return (False, "SKIP: u(f,T) not available")
        if u > 0:
            return (True, f"u(f,T) = {u} > 0")
        else:
            return (False, f"u(f,T) = {u} is not positive")

    def test_invariant_low_f_limit(self):
        """At low frequency (hf << kT), u ≈ (8π f^2 / c^3) k T."""
        u = self._to_float(self.u_f_T)
        h = self._to_float(self.h)
        c = self._to_float(self.c)
        k = self._to_float(self.k)
        T = self._to_float(self.T)
        f = self._to_float(self.f)
        if any(x is None for x in [u, h, c, k, T, f]):
            return (False, "SKIP: missing u, h, c, k, T, or f")
        # Check low‑frequency condition: hf << kT
        if f <= 0 or T <= 0:
            return (False, "SKIP: non‑positive f or T")
        exponent = h * f / (k * T)
        if exponent > 0.1:  # not in the low‑frequency regime
            return (False, f"SKIP: hf/(kT) = {exponent} > 0.1, not low frequency")
        classical = (8.0 * math.pi * f**2 * k * T) / (c**3)
        if math.isclose(u, classical, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"u={u} ≈ classical Rayleigh‑Jeans {classical}")
        else:
            return (False, f"u={u} differs from classical limit {classical}")

    def test_invariant_threshold_frequency(self):
        """If f is exactly the threshold frequency, K_max should be zero."""
        h = self._to_float(self.h)
        phi = self._to_float(self.phi)
        f = self._to_float(self.f)
        K = self._to_float(self.K_max)
        if any(x is None for x in [h, phi, f, K]):
            return (False, "SKIP: missing h, phi, f, or K_max")
        if h <= 0:
            return (False, "SKIP: non‑positive h")
        f0 = phi / h
        if math.isclose(f, f0, rel_tol=1e-2, abs_tol=1e-2):
            if math.isclose(K, 0.0, abs_tol=1e-2):
                return (True, f"At threshold f={f}, K_max={K} ≈ 0")
            else:
                return (False, f"At threshold f={f}, K_max={K} should be ≈0")
        else:
            return (False, f"SKIP: given f={f} is not the threshold frequency {f0}")

    # ------------------------------------------------------------------
    # Modeling tests (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_photoelectric_explanation(self):
        """Check that explanation_steps mentions Einstein’s equation or photon quantization."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        steps_text = str(self.explanation_steps).lower()
        keywords = ['einstein', 'photon', 'hν', 'hf', 'work function', 'kinetic energy']
        found = [kw for kw in keywords if kw in steps_text]
        if found:
            return (True, f"Photoelectric effect key concepts found: {found}")
        else:
            return (False, "No photoelectric key terms found in explanation_steps")

    def test_modeling_blackbody_explanation(self):
        """Check that explanation_steps mentions Planck’s law and quantization."""
        if self.explanation_steps is None:
            return (False, "SKIP: no explanation_steps provided")
        steps_text = str(self.explanation_steps).lower()
        keywords = ['planck', 'quantization', 'hν', 'hf', 'oscillator', 'blackbody', 'ultraviolet catastrophe']
        found = [kw for kw in keywords if kw in steps_text]
        if found:
            return (True, f"Blackbody key concepts found: {found}")
        else:
            return (False, "No blackbody key terms found in explanation_steps")

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb an input and compare output)
    # ------------------------------------------------------------------
    def test_monotonicity_Kmax_increases_with_f(self):
        """Increasing frequency should increase K_max (hf - phi)."""
        K = self._to_float(self.K_max)
        h = self._to_float(self.h)
        phi = self._to_float(self.phi)
        f = self._to_float(self.f)
        if any(x is None for x in [K, h, phi, f]):
            return (False, "SKIP: missing K_max, h, phi, or f")
        if h <= 0 or f <= 0:
            return (False, "SKIP: non‑positive h or f")
        # Perturb frequency upward by 5%
        f_pert = f * 1.05
        K_pert = h * f_pert - phi
        if K_pert > K:
            return (True, f"K_max increases from {K} to {K_pert} when f↑5%")
        else:
            return (False, f"K_max does not increase: base={K}, perturbed={K_pert}")

    def test_monotonicity_uv_falls_high_f(self):
        """At high frequency (hf >> kT), u(f,T) should decrease (exponential cut‑off)."""
        u = self._to_float(self.u_f_T)
        h = self._to_float(self.h)
        c = self._to_float(self.c)
        k = self._to_float(self.k)
        T = self._to_float(self.T)
        f = self._to_float(self.f)
        if any(x is None for x in [u, h, c, k, T, f]):
            return (False, "SKIP: missing u, h, c, k, T, or f")
        if T <= 0 or f <= 0:
            return (False, "SKIP: non‑positive T or f")
        # Check that we are in the high‑frequency regime (hf >> kT)
        exponent = h * f / (k * T)
        if exponent < 5:
            return (False, f"SKIP: hf/(kT) = {exponent} is not >>1 (need >5)")
        # Perturb frequency upward by 10% and recompute u
        f_pert = f * 1.10
        u_pert = (8 * math.pi * h * f_pert**3 / c**3) / (math.exp(h * f_pert / (k * T)) - 1)
        if u_pert < u:
            return (True, f"u decreases from {u} to {u_pert} when f↑10% (high‑f regime)")
        else:
            return (False, f"u does not decrease: base={u}, perturbed={u_pert}")

    # ------------------------------------------------------------------
    # Weak answer sanity (only one, no isclose, no rederivation)
    # ------------------------------------------------------------------
    def test_answer_sanity_values(self):
        """Weak sanity: all output values should be finite and non‑negative where applicable."""
        checks = []
        # K_max should be non‑negative or could be negative if below threshold; still finite.
        K = self._to_float(self.K_max)
        if K is not None:
            if not math.isfinite(K):
                checks.append("K_max is not finite")
            # No sign requirement – allow negative for no emission scenario
        # u_f_T must be positive if both T and f are present
        u = self._to_float(self.u_f_T)
        if u is not None:
            if not math.isfinite(u):
                checks.append("u(f,T) is not finite")
            if u < 0:
                checks.append("u(f,T) is negative")
        # hf and phi should be finite
        hf = self._to_float(self.hf)
        if hf is not None and not math.isfinite(hf):
            checks.append("hf is not finite")
        phi = self._to_float(self.phi)
        if phi is not None and not math.isfinite(phi):
            checks.append("phi is not finite")
        if checks:
            return (False, "; ".join(checks))
        return (True, "All tested answer values are finite and meet basic sanity criteria")

    # ------------------------------------------------------------------
    # Main test runner
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
                passed, msg = False, f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)