import math
import sympy as sp
import cmath

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens safely
        givens = getattr(solution_instance, 'givens', None) or {}
        self.N = getattr(givens, 'N', None)
        self.E0 = getattr(givens, 'E0', None)
        self.W = getattr(givens, 'W', None)
        
        # Extract results safely
        results = getattr(solution_instance, 'results', None) or {}
        self.E_j = results.get('E_j', None)          # expected list of length N or None
        self.eigenstate_coeffs = results.get('eigenstate_coeffs', None)  # list of N lists (coeffs per eigenstate)
        
        # Auxiliary: explanation_steps for modeling guidance
        self.explanation_steps = getattr(solution_instance, 'explanation_steps', None)
        
    # ------------------------------------------------------------------
    # Interface tests
    # ------------------------------------------------------------------
    def test_interface_required_parameters(self):
        """Check that N, E0, W are present and numeric."""
        if self.N is None:
            return False, "Missing parameter: N"
        if not isinstance(self.N, (int, float)):
            return False, f"N should be int or float, got {type(self.N)}"
        if self.E0 is None:
            return False, "Missing parameter: E0"
        if not isinstance(self.E0, (int, float, complex)):
            return False, f"E0 should be numeric, got {type(self.E0)}"
        if self.W is None:
            return False, "Missing parameter: W"
        if not isinstance(self.W, (int, float, complex)):
            return False, f"W should be numeric, got {type(self.W)}"
        return True, "All required parameters present and numeric"

    def test_interface_E_j_list(self):
        """E_j must be a list of length N."""
        if self.E_j is None:
            return False, "Missing result: E_j"
        if not isinstance(self.E_j, (list, tuple)):
            return False, f"E_j should be a list/tuple, got {type(self.E_j)}"
        if self.N is not None and len(self.E_j) != self.N:
            return False, f"Expected {self.N} eigenvalues, got {len(self.E_j)}"
        return True, f"E_j is a list of length {len(self.E_j)}"

    def test_interface_eigenstate_coeffs_list(self):
        """eigenstate_coeffs must be a list of N lists (each of length N)."""
        if self.eigenstate_coeffs is None:
            return False, "Missing result: eigenstate_coeffs"
        if not isinstance(self.eigenstate_coeffs, (list, tuple)):
            return False, f"eigenstate_coeffs should be a list/tuple, got {type(self.eigenstate_coeffs)}"
        if self.N is not None and len(self.eigenstate_coeffs) != self.N:
            return False, f"Expected {self.N} eigenstates, got {len(self.eigenstate_coeffs)}"
        # Check each eigenstate coefficients list length
        for idx, coeffs in enumerate(self.eigenstate_coeffs):
            if not isinstance(coeffs, (list, tuple)):
                return False, f"Eigenstate {idx} coeffs is not a list/tuple"
            if self.N is not None and len(coeffs) != self.N:
                return False, f"Eigenstate {idx} has {len(coeffs)} coefficients, expected {self.N}"
        return True, f"eigenstate_coeffs is a list of {len(self.eigenstate_coeffs)} eigenstates, each of length {self.N}"

    # ------------------------------------------------------------------
    # Invariant tests (physical sanity)
    # ------------------------------------------------------------------
    def test_invariant_dispersion_relation(self):
        """Check that E_j = E0 + 2W cos(2π j/N) for all j (fuzzy)."""
        if self.N is None or self.E0 is None or self.W is None or self.E_j is None:
            return False, "SKIP: missing required data"
        for j, E_j in enumerate(self.E_j):
            expected = self.E0 + 2 * self.W * math.cos(2 * math.pi * j / self.N)
            if not math.isclose(E_j, expected, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"E_{j} = {E_j}, expected {expected}"
        return True, f"All {self.N} eigenvalues satisfy dispersion relation within tolerance"

    def test_invariant_normalization(self):
        """For each eigenstate, sum of |c_n|^2 = 1 (fuzzy)."""
        if self.eigenstate_coeffs is None or self.N is None:
            return False, "SKIP: missing eigenstate_coeffs or N"
        for idx, coeffs in enumerate(self.eigenstate_coeffs):
            norm_sq = sum(abs(c)**2 for c in coeffs)
            if not math.isclose(norm_sq, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                return False, f"Eigenstate {idx} norm^2 = {norm_sq}, expected 1.0"
        return True, f"All {len(self.eigenstate_coeffs)} eigenstates have unit norm"

    def test_invariant_orthogonality(self):
        """Check that different eigenstates are orthogonal: <ψ_j|ψ_k> ≈ 0 for j≠k (fuzzy)."""
        if self.eigenstate_coeffs is None or self.N is None:
            return False, "SKIP: missing eigenstate_coeffs or N"
        if len(self.eigenstate_coeffs) < 2:
            return True, "Only one eigenstate, orthogonality trivially satisfied"
        for j in range(len(self.eigenstate_coeffs)):
            for k in range(j+1, len(self.eigenstate_coeffs)):
                overlap = sum(cj.conjugate()*ck for cj, ck in zip(self.eigenstate_coeffs[j], self.eigenstate_coeffs[k]))
                # For real inner product we use abs(overlap)
                if not math.isclose(abs(overlap), 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    return False, f"Overlap between eigenstates {j} and {k} = {abs(overlap)}, expected 0"
        return True, "All eigenstates orthogonal within tolerance"

    def test_invariant_energy_real(self):
        """Eigenvalues must be real (periodic boundaries give real from cos)."""
        if self.E_j is None:
            return False, "SKIP: no E_j"
        for j, val in enumerate(self.E_j):
            if abs(val.imag) > 1e-8:
                return False, f"E_{j} has imaginary part {val.imag}"
        return True, "All eigenvalues are real"

    # ------------------------------------------------------------------
    # Modeling / trace tests (based on explanation steps and physics)
    # ------------------------------------------------------------------
    def test_modeling_bloch_phase_progression(self):
        """Check that for each eigenstate, coefficient ratio c_{n+1}/c_n is constant = e^{i k_j}."""
        if self.eigenstate_coeffs is None or self.N is None or self.E_j is None:
            return False, "SKIP: missing required data"
        for j in range(len(self.eigenstate_coeffs)):
            coeffs = self.eigenstate_coeffs[j]
            # Compute k_j from dispersion? Better: from derived phase.
            # We'll compute ratio for consecutive n (0-based index)
            if len(coeffs) < 2:
                continue  # N=1 trivial
            # Ratio should be exp(i*k_j) where k_j = 2πj/N
            k_j = 2 * math.pi * j / self.N
            expected_ratio = cmath.exp(1j * k_j)
            for n in range(len(coeffs)-1):
                if abs(coeffs[n]) < 1e-12 and abs(coeffs[n+1]) < 1e-12:
                    continue  # degenerate zero coefficients
                # Avoid division by zero
                if abs(coeffs[n]) < 1e-12 or abs(coeffs[n+1]) < 1e-12:
                    return False, f"Eigenstate {j}, n={n}: coefficient(s) near zero, cannot verify phase"
                ratio = coeffs[n+1] / coeffs[n]
                # Closeness of complex numbers up to overall phase? The ratio should be exactly expected_ratio
                # But normalization may introduce phase. Actually we can check arg difference.
                if not math.isclose(ratio.real, expected_ratio.real, rel_tol=1e-2, abs_tol=1e-2) or \
                   not math.isclose(ratio.imag, expected_ratio.imag, rel_tol=1e-2, abs_tol=1e-2):
                    return False, f"Eigenstate {j}, n={n}: ratio {ratio}, expected {expected_ratio}"
        return True, "All eigenstates show correct Bloch phase progression"

    def test_modeling_uniform_amplitude(self):
        """For each eigenstate, all |c_n| should be equal (1/sqrt(N)) - (fuzzy)."""
        if self.eigenstate_coeffs is None or self.N is None:
            return False, "SKIP: missing data"
        for j, coeffs in enumerate(self.eigenstate_coeffs):
            amplitudes = [abs(c) for c in coeffs]
            A0 = amplitudes[0]
            for n, A in enumerate(amplitudes):
                if not math.isclose(A, A0, rel_tol=1e-2, abs_tol=1e-2):
                    return False, f"Eigenstate {j}, site {n}: amplitude {A}, expected {A0} (uniform)"
        return True, "All eigenstates have uniform amplitude distribution"

    # ------------------------------------------------------------------
    # Monotonicity tests (perturb input and check trend)
    # ------------------------------------------------------------------
    def test_monotonicity_energy_spread_increases_with_W(self):
        """If W > 0, increasing W should increase the energy bandwidth (max E - min E)."""
        # Can't perturb actual inputs because we have only one fixed solution_instance.
        # But we can compute expected trend: For given N, bandwidth = 4|W| (since cos ranges -1 to 1).
        # This tests that the reported E_j are consistent with the formula.
        if self.W is None or self.E_j is None or self.N is None:
            return True, "SKIP: missing data, cannot test monotonicity"
        if abs(self.W) < 1e-12:
            return True, "W=0, bandwidth zero, monotonicity trivial"
        bandwidth = max(self.E_j) - min(self.E_j)
        # Expected bandwidth: 4|W|
        expected_bandwidth = 4 * abs(self.W)
        if not math.isclose(bandwidth, expected_bandwidth, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"Bandwidth {bandwidth}, expected {expected_bandwidth} (4|W|)"
        return True, f"Bandwidth {bandwidth} matches 4|W| = {expected_bandwidth}"

    # ------------------------------------------------------------------
    # Weak answer sanity (only one such test allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_finite_real(self):
        """All energies should be finite real numbers."""
        if self.E_j is None:
            return False, "SKIP: no E_j"
        for j, val in enumerate(self.E_j):
            if not isinstance(val, (int, float, complex)):
                return False, f"E_{j} is not numeric"
            if not math.isfinite(val.real) or not math.isfinite(val.imag):
                return False, f"E_{j} is not finite"
        return True, "All energies finite"

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
                passed, msg = False, f"{name} CRASHED: {e}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return (all_pass, logs)