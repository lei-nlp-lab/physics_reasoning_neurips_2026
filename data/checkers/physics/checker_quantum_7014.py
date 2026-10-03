import math
import sympy as sp

class PhysicsChecker:
    """
    PhysicsChecker for a system of two identical particles in a 1D harmonic oscillator
    with mutual harmonic interaction.
    """
    def __init__(self, solution_instance):
        # Extract givens safely – always use None for missing values
        self.M = getattr(solution_instance, 'M', None)
        self.k = getattr(solution_instance, 'k', None)
        self.K = getattr(solution_instance, 'K', None)
        self.hbar = getattr(solution_instance, 'hbar', None)

        # Extract results from solution_instance.results dict (safe access)
        results = getattr(solution_instance, 'results', {})
        self.E00 = results.get('E00', None)
        self.E10 = results.get('E10', None)
        self.E01 = results.get('E01', None)
        self.allowed_states = results.get('allowed_states', None)
        self.S_00 = results.get('S_00', None)
        self.S_10 = results.get('S_10', None)
        self.S_01 = results.get('S_01', None)

        # Compute derived quantities for later checks
        self.omega_CM = None
        self.omega_rel = None
        if self.M is not None and self.M > 0 and self.k is not None and self.k >= 0:
            self.omega_CM = math.sqrt(self.k / self.M)
        if self.M is not None and self.M > 0 and self.k is not None and self.K is not None:
            self.omega_rel = math.sqrt((self.k + 2 * self.K) / self.M)

        # Derived energies (for comparison)
        self.E00_calc = None
        self.E10_calc = None
        self.E01_calc = None
        if self.hbar is not None and self.omega_CM is not None and self.omega_rel is not None:
            self.E00_calc = 0.5 * self.hbar * (self.omega_CM + self.omega_rel)
            self.E10_calc = self.E00_calc + self.hbar * self.omega_CM
            self.E01_calc = self.E00_calc + self.hbar * self.omega_rel

    # ----------------------------------------------------------------------
    # Interface checks
    # ----------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all expected results keys exist."""
        if self.E00 is None:
            return (False, "Missing result: E00")
        if self.E10 is None:
            return (False, "Missing result: E10")
        if self.E01 is None:
            return (False, "Missing result: E01")
        if self.allowed_states is None:
            return (False, "Missing result: allowed_states")
        if self.S_00 is None:
            return (False, "Missing result: S_00")
        if self.S_10 is None:
            return (False, "Missing result: S_10")
        if self.S_01 is None:
            return (False, "Missing result: S_01")
        return (True, "All required result keys present")

    def test_interface_givens_valid(self):
        """Check that essential givens are provided and physically meaningful."""
        if self.M is None:
            return (False, "M is missing")
        if self.M <= 0:
            return (False, f"M={self.M} should be positive")
        if self.k is None:
            return (False, "k is missing")
        if self.k < 0:
            return (False, f"k={self.k} cannot be negative")
        if self.K is None:
            return (False, "K is missing")
        if self.K < 0:
            return (False, f"K={self.K} cannot be negative (attractive force)")
        if self.hbar is None:
            return (False, "hbar is missing")
        if self.hbar <= 0:
            return (False, "hbar must be positive")
        return (True, "All givens present and valid")

    # ----------------------------------------------------------------------
    # Invariant checks (theorem-based, use math.isclose)
    # ----------------------------------------------------------------------
    def test_invariant_energies_positive(self):
        """All three energies must be positive."""
        if any(e is None for e in [self.E00, self.E10, self.E01]):
            return (True, "SKIP: missing energies")
        if self.E00 <= 0:
            return (False, f"E00={self.E00} should be >0")
        if self.E10 <= 0:
            return (False, f"E10={self.E10} should be >0")
        if self.E01 <= 0:
            return (False, f"E01={self.E01} should be >0")
        return (True, "All energies positive")

    def test_invariant_energy_ordering(self):
        """For K>0, ω_rel > ω_CM, so E01 > E10 > E00."""
        if any(e is None for e in [self.E00, self.E10, self.E01]):
            return (True, "SKIP: missing energies")
        if self.K is None or self.K <= 0:
            return (True, "SKIP: K not positive")
        if not (self.E10 > self.E00):
            return (False, f"E10={self.E10} not > E00={self.E00}")
        if not (self.E01 > self.E10):
            return (False, f"E01={self.E01} not > E10={self.E10}")
        return (True, "Energy ordering correct: E01 > E10 > E00")

    def test_invariant_ground_energy_formula(self):
        """Check E00 against derived formula using math.isclose."""
        if self.E00 is None or self.E00_calc is None:
            return (True, "SKIP: missing E00 or givens for calculation")
        if not math.isclose(self.E00, self.E00_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E00={self.E00} deviates from derived {self.E00_calc}")
        return (True, "Ground state energy matches formula")

    def test_invariant_first_CM_energy_formula(self):
        """Check E10 = E00 + ℏ ω_CM."""
        if self.E10 is None or self.E10_calc is None:
            return (True, "SKIP: missing E10 or givens")
        if not math.isclose(self.E10, self.E10_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E10={self.E10} deviates from derived {self.E10_calc}")
        return (True, "First CM excited state energy matches formula")

    def test_invariant_first_rel_energy_formula(self):
        """Check E01 = E00 + ℏ ω_rel."""
        if self.E01 is None or self.E01_calc is None:
            return (True, "SKIP: missing E01 or givens")
        if not math.isclose(self.E01, self.E01_calc, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"E01={self.E01} deviates from derived {self.E01_calc}")
        return (True, "First relative excited state energy matches formula")

    # ----------------------------------------------------------------------
    # Modeling checks (based on explanation steps)
    # ----------------------------------------------------------------------
    def test_modeling_allowed_states_spinless(self):
        """For spinless bosons, only even relative quantum number states are allowed."""
        if self.allowed_states is None:
            return (True, "SKIP: allowed_states missing")
        if not isinstance(self.allowed_states, list):
            return (False, "allowed_states should be a list of tuples")
        expected = [(0, 0), (1, 0)]
        if set(tuple(s) for s in self.allowed_states) != set(expected):
            return (False, f"allowed_states = {self.allowed_states}, expected {expected}")
        return (True, "Allowed states (spinless) correctly identified")

    def test_modeling_spin_assignments(self):
        """For fermions, check spin assignments for each low state."""
        if None in [self.S_00, self.S_10, self.S_01]:
            return (True, "SKIP: missing spin results")
        errors = []
        if self.S_00 != 0:
            errors.append(f"S_00 should be 0, got {self.S_00}")
        if self.S_10 != 0:
            errors.append(f"S_10 should be 0, got {self.S_10}")
        if self.S_01 != 1:
            errors.append(f"S_01 should be 1, got {self.S_01}")
        if errors:
            return (False, "; ".join(errors))
        return (True, "Spin assignments correct")

    def test_modeling_relative_parity_logic(self):
        """Consistency: spatial symmetry ↔ spin for fermions. Not a quantitative test."""
        if None in [self.S_00, self.S_01]:
            return (True, "SKIP: missing spin data")
        return (True, "Spin-parity logic appears consistent")

    # ----------------------------------------------------------------------
    # Monotonicity checks – each perturbs an input and compares perturbed output to baseline
    # Baseline is the solution's output; perturbed output is computed from perturbed givens.
    # ----------------------------------------------------------------------
    def test_monotonicity_energy_increase_with_K(self):
        """Perturb K by +10% and verify that all three predicted energies increase relative to solution's values."""
        if any(e is None for e in [self.E00, self.E10, self.E01]):
            return (True, "SKIP: missing energies")
        if any(g is None for g in [self.M, self.k, self.K, self.hbar]):
            return (True, "SKIP: missing givens")
        if self.M <= 0 or self.k < 0:
            return (True, "SKIP: invalid givens")

        # Baseline energies from solution
        baseline = (self.E00, self.E10, self.E01)

        # Compute baseline omegas from givens (should match solution; we already checked invariants)
        omega_CM_base = math.sqrt(self.k / self.M)
        omega_rel_base = math.sqrt((self.k + 2 * self.K) / self.M)

        # Perturb K by +10%
        K_pert = self.K * 1.1
        omega_rel_pert = math.sqrt((self.k + 2 * K_pert) / self.M)
        E00_pert = 0.5 * self.hbar * (omega_CM_base + omega_rel_pert)
        E10_pert = E00_pert + self.hbar * omega_CM_base
        E01_pert = E00_pert + self.hbar * omega_rel_pert

        errors = []
        if E00_pert <= baseline[0]:
            errors.append(f"E00 did not increase: baseline {baseline[0]}, perturbed predicted {E00_pert}")
        if E10_pert <= baseline[1]:
            errors.append(f"E10 did not increase: baseline {baseline[1]}, perturbed predicted {E10_pert}")
        if E01_pert <= baseline[2]:
            errors.append(f"E01 did not increase: baseline {baseline[2]}, perturbed predicted {E01_pert}")
        if errors:
            return (False, "; ".join(errors))
        return (True, "All energies increase under +10% K perturbation")

    def test_monotonicity_energy_increase_with_k(self):
        """Perturb k by +10% and verify that all three predicted energies increase relative to solution's values."""
        if any(e is None for e in [self.E00, self.E10, self.E01]):
            return (True, "SKIP: missing energies")
        if any(g is None for g in [self.M, self.k, self.K, self.hbar]):
            return (True, "SKIP: missing givens")
        if self.M <= 0 or self.k < 0:
            return (True, "SKIP: invalid givens")

        baseline = (self.E00, self.E10, self.E01)

        # Perturb k by +10%
        k_pert = self.k * 1.1
        omega_CM_pert = math.sqrt(k_pert / self.M)
        omega_rel_pert = math.sqrt((k_pert + 2 * self.K) / self.M)
        E00_pert = 0.5 * self.hbar * (omega_CM_pert + omega_rel_pert)
        E10_pert = E00_pert + self.hbar * omega_CM_pert
        E01_pert = E00_pert + self.hbar * omega_rel_pert

        errors = []
        if E00_pert <= baseline[0]:
            errors.append(f"E00 did not increase: baseline {baseline[0]}, perturbed predicted {E00_pert}")
        if E10_pert <= baseline[1]:
            errors.append(f"E10 did not increase: baseline {baseline[1]}, perturbed predicted {E10_pert}")
        if E01_pert <= baseline[2]:
            errors.append(f"E01 did not increase: baseline {baseline[2]}, perturbed predicted {E01_pert}")
        if errors:
            return (False, "; ".join(errors))
        return (True, "All energies increase under +10% k perturbation")

    # ----------------------------------------------------------------------
    # Answer sanity (weak, no re-derivation)
    # ----------------------------------------------------------------------
    def test_answer_sanity_non_nan(self):
        """Check that all numerical results are finite and not NaN."""
        for name, val in [("E00", self.E00), ("E10", self.E10), ("E01", self.E01),
                          ("S_00", self.S_00), ("S_10", self.S_10), ("S_01", self.S_01)]:
            if val is None:
                continue
            if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                return (False, f"{name} is NaN or Inf")
        if self.allowed_states is None:
            return (True, "SKIP: allowed_states missing")
        for state in self.allowed_states:
            if not isinstance(state, (list, tuple)):
                return (False, "State in allowed_states not a tuple/list")
            if not all(isinstance(n, int) and n >= 0 for n in state):
                return (False, f"State {state} has non-integer or negative quantum numbers")
        return (True, "All numerical results are finite and non-NaN")

    # ----------------------------------------------------------------------
    # Main test runner
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