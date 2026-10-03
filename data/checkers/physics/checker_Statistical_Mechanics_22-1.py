import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract results dictionary or use empty
        results = getattr(solution_instance, 'results', {})
        if results is None:
            results = {}
        
        # Extract results safely
        self.Q = results.get('Q', None)
        self.S_dot_water = results.get('S_dot_water', None)
        self.S_dot_ins = results.get('S_dot_ins', None)
        self.S_dot_room = results.get('S_dot_room', None)
        self.S_dot_univ = results.get('S_dot_univ', None)
        
        # Extract givens: try from results, then from solution_instance attributes
        self.T_h = results.get('T_h', None) or getattr(solution_instance, 'T_h', None)
        self.T_c = results.get('T_c', None) or getattr(solution_instance, 'T_c', None)
        self.r1 = results.get('r1', None) or getattr(solution_instance, 'r1', None)
        self.r2 = results.get('r2', None) or getattr(solution_instance, 'r2', None)
        self.L = results.get('L', None) or getattr(solution_instance, 'L', None)
        self.k = results.get('k', None) or getattr(solution_instance, 'k', None)
        
        # Also try from givens dictionary if present
        givens = getattr(solution_instance, 'givens', None)
        if givens is not None:
            self.T_h = self.T_h or givens.get('T_h')
            self.T_c = self.T_c or givens.get('T_c')
            self.r1 = self.r1 or givens.get('r1')
            self.r2 = self.r2 or givens.get('r2')
            self.L = self.L or givens.get('L')
            self.k = self.k or givens.get('k')
        
        # Fallback to None if not found
        if self.T_h is None:
            self.T_h = None
        if self.T_c is None:
            self.T_c = None
        if self.r1 is None:
            self.r1 = None
        if self.r2 is None:
            self.r2 = None
        if self.L is None:
            self.L = None
        if self.k is None:
            self.k = None

    # Helper to compute Q from givens
    def _compute_Q(self, T_h, T_c, r1, r2, L, k):
        """Return Q using the radial conduction formula."""
        return 2 * math.pi * k * L * (T_h - T_c) / math.log(r2 / r1)

    # -------------------------------------------------------------------
    # Interface tests
    # -------------------------------------------------------------------
    def test_interface_required_keys(self):
        """Check that all required result keys exist and have correct types."""
        required = ['Q', 'S_dot_water', 'S_dot_ins', 'S_dot_room', 'S_dot_univ']
        for key in required:
            val = getattr(self, key, None)
            if val is None:
                return (False, f"Missing key {key}")
            if not isinstance(val, (int, float)):
                return (False, f"Key {key} has non-numeric type: {type(val)}")
        return (True, "All required keys present and numeric")

    def test_interface_givens_present(self):
        """Check that given physical parameters are present and nonzero where required."""
        givens = {'T_h': self.T_h, 'T_c': self.T_c, 'r1': self.r1,
                  'r2': self.r2, 'L': self.L, 'k': self.k}
        missing = []
        for name, val in givens.items():
            if val is None:
                missing.append(name)
            elif not isinstance(val, (int, float)):
                missing.append(name)
        if missing:
            return (False, f"Missing givens: {missing}")
        # Additional checks
        if self.r1 is not None and self.r1 <= 0:
            return (False, f"r1 must be positive, got {self.r1}")
        if self.r2 is not None and self.r2 <= self.r1:
            return (False, f"r2 must be > r1, got r2={self.r2}, r1={self.r1}")
        if self.L is not None and self.L <= 0:
            return (False, f"L must be positive, got {self.L}")
        if self.k is not None and self.k <= 0:
            return (False, f"k must be positive, got {self.k}")
        if self.T_h is not None and self.T_c is not None and self.T_h <= self.T_c:
            return (False, f"T_h must be > T_c, got T_h={self.T_h}, T_c={self.T_c}")
        return (True, "All givens present and valid")

    # -------------------------------------------------------------------
    # Invariant tests (use math.isclose)
    # -------------------------------------------------------------------
    def test_invariant_radial_conduction(self):
        """Check that Q matches Fourier's law via radial conduction."""
        if any(v is None for v in [self.T_h, self.T_c, self.r1, self.r2, self.L, self.k, self.Q]):
            return (False, "SKIP: missing data for Q computation")
        Q_expected = self._compute_Q(self.T_h, self.T_c, self.r1, self.r2, self.L, self.k)
        if math.isclose(self.Q, Q_expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"Q matches formula: {self.Q} ~ {Q_expected}")
        else:
            return (False, f"Q does not match formula: computed {Q_expected}, got {self.Q}")

    def test_invariant_entropy_water(self):
        """Check S_dot_water = -Q/T_h."""
        if self.S_dot_water is None or self.Q is None or self.T_h is None:
            return (False, "SKIP: missing data for water entropy")
        expected = -self.Q / self.T_h
        if math.isclose(self.S_dot_water, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S_dot_water matches -Q/T_h: {self.S_dot_water} ~ {expected}")
        else:
            return (False, f"S_dot_water mismatch: expected {expected}, got {self.S_dot_water}")

    def test_invariant_entropy_insulation(self):
        """Check S_dot_ins == 0 (steady state)."""
        if self.S_dot_ins is None:
            return (False, "SKIP: missing S_dot_ins")
        if math.isclose(self.S_dot_ins, 0.0, abs_tol=1e-2):
            return (True, "S_dot_ins = 0 as expected")
        else:
            return (False, f"S_dot_ins should be 0, got {self.S_dot_ins}")

    def test_invariant_entropy_room(self):
        """Check S_dot_room = Q/T_c."""
        if self.S_dot_room is None or self.Q is None or self.T_c is None:
            return (False, "SKIP: missing data for room entropy")
        expected = self.Q / self.T_c
        if math.isclose(self.S_dot_room, expected, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S_dot_room matches Q/T_c: {self.S_dot_room} ~ {expected}")
        else:
            return (False, f"S_dot_room mismatch: expected {expected}, got {self.S_dot_room}")

    def test_invariant_entropy_universe(self):
        """Check S_dot_univ = S_dot_water + S_dot_ins + S_dot_room and also = Q*(1/T_c - 1/T_h)."""
        if any(v is None for v in [self.S_dot_univ, self.S_dot_water, self.S_dot_ins, self.S_dot_room, self.Q, self.T_c, self.T_h]):
            return (False, "SKIP: missing data for universe entropy")
        # Check via sum
        sum_entropies = self.S_dot_water + self.S_dot_ins + self.S_dot_room
        if not math.isclose(self.S_dot_univ, sum_entropies, rel_tol=1e-2, abs_tol=1e-2):
            return (False, f"S_dot_univ {self.S_dot_univ} != sum {sum_entropies}")
        # Check via direct formula
        direct = self.Q * (1.0/self.T_c - 1.0/self.T_h)
        if math.isclose(self.S_dot_univ, direct, rel_tol=1e-2, abs_tol=1e-2):
            return (True, f"S_dot_univ consistent: {self.S_dot_univ} ~ sum and direct formula")
        else:
            return (False, f"S_dot_univ {self.S_dot_univ} != direct formula {direct}")

    # -------------------------------------------------------------------
    # Modeling/trace tests (signs, physics constraints)
    # -------------------------------------------------------------------
    def test_modeling_heat_sign(self):
        """Check Q > 0 (heat flows from hot to cold)."""
        if self.Q is None:
            return (False, "SKIP: missing Q")
        if self.Q > 0:
            return (True, "Q > 0 as expected")
        else:
            return (False, f"Q must be positive, got {self.Q}")

    def test_modeling_entropy_water_sign(self):
        """Check S_dot_water < 0 (water loses heat)."""
        if self.S_dot_water is None:
            return (False, "SKIP: missing S_dot_water")
        if self.S_dot_water < 0:
            return (True, "S_dot_water < 0")
        else:
            return (False, f"S_dot_water must be negative, got {self.S_dot_water}")

    def test_modeling_entropy_room_sign(self):
        """Check S_dot_room > 0 (room gains heat)."""
        if self.S_dot_room is None:
            return (False, "SKIP: missing S_dot_room")
        if self.S_dot_room > 0:
            return (True, "S_dot_room > 0")
        else:
            return (False, f"S_dot_room must be positive, got {self.S_dot_room}")

    def test_modeling_entropy_univ_sign(self):
        """Check S_dot_univ > 0 (second law)."""
        if self.S_dot_univ is None:
            return (False, "SKIP: missing S_dot_univ")
        if self.S_dot_univ > 0:
            return (True, "S_dot_univ > 0 (second law satisfied)")
        else:
            return (False, f"S_dot_univ must be positive, got {self.S_dot_univ}")

    def test_modeling_insulation_entropy_zero(self):
        """Check S_dot_ins is exactly 0 (within tolerance)."""
        if self.S_dot_ins is None:
            return (False, "SKIP: missing S_dot_ins")
        if math.isclose(self.S_dot_ins, 0.0, abs_tol=1e-6):
            return (True, "Insulation entropy zero (steady state)")
        else:
            return (False, f"S_dot_ins should be 0, got {self.S_dot_ins}")

    # -------------------------------------------------------------------
    # Monotonicity tests (perturb one parameter, check trend)
    # -------------------------------------------------------------------
    def test_monotonicity_Q_increases_with_k(self):
        """If k increases, Q should increase."""
        if any(v is None for v in [self.T_h, self.T_c, self.r1, self.r2, self.L, self.k, self.Q]):
            return (False, "SKIP: missing data for monotonicity k")
        # Perturb k by +10%
        k_pert = self.k * 1.1
        Q_pert = self._compute_Q(self.T_h, self.T_c, self.r1, self.r2, self.L, k_pert)
        if Q_pert > self.Q:
            return (True, f"Increasing k increases Q: base {self.Q}, perturbed {Q_pert}")
        else:
            return (False, f"Monotonicity failed: perturbed Q {Q_pert} <= base {self.Q}")

    def test_monotonicity_Q_increases_with_L(self):
        """If L increases, Q should increase."""
        if any(v is None for v in [self.T_h, self.T_c, self.r1, self.r2, self.L, self.k, self.Q]):
            return (False, "SKIP: missing data for monotonicity L")
        L_pert = self.L * 1.1
        Q_pert = self._compute_Q(self.T_h, self.T_c, self.r1, self.r2, L_pert, self.k)
        if Q_pert > self.Q:
            return (True, f"Increasing L increases Q: base {self.Q}, perturbed {Q_pert}")
        else:
            return (False, f"Monotonicity failed: perturbed Q {Q_pert} <= base {self.Q}")

    def test_monotonicity_Q_decreases_with_thickness(self):
        """If insulation thickness increases (r2 larger), Q should decrease."""
        if any(v is None for v in [self.T_h, self.T_c, self.r1, self.r2, self.L, self.k, self.Q]):
            return (False, "SKIP: missing data for monotonicity thickness")
        # Increase r2 by 10% (but must stay > r1)
        r2_pert = self.r2 * 1.1
        if r2_pert <= self.r1:
            return (False, "SKIP: perturbed r2 <= r1")
        Q_pert = self._compute_Q(self.T_h, self.T_c, self.r1, r2_pert, self.L, self.k)
        if Q_pert < self.Q:
            return (True, f"Increasing r2 decreases Q: base {self.Q}, perturbed {Q_pert}")
        else:
            return (False, f"Monotonicity failed: perturbed Q {Q_pert} >= base {self.Q}")

    # -------------------------------------------------------------------
    # Weak answer sanity (broad magnitude, no isclose)
    # -------------------------------------------------------------------
    def test_answer_sanity_broad_magnitude(self):
        """Check that Q is in a plausible range (50-200 W) and other signs."""
        if self.Q is None:
            return (False, "SKIP: missing Q")
        if not (50 < self.Q < 200):
            return (False, f"Q outside plausible range [50,200]: {self.Q}")
        # Also check S_dot_univ is positive and less than 0.5
        if self.S_dot_univ is not None and not (0 < self.S_dot_univ < 0.5):
            return (False, f"S_dot_univ outside plausible range: {self.S_dot_univ}")
        return (True, "Magnitudes plausible")

    # -------------------------------------------------------------------
    # Main test runner
    # -------------------------------------------------------------------
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