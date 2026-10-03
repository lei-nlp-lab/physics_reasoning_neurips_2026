import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Extract givens from solution_instance (attributes or givens dict)
        self.n = getattr(solution_instance, 'n', None)
        self.R = getattr(solution_instance, 'R', None)
        self.d_o = getattr(solution_instance, 'd_o', None)

        # Extract result dict safely
        self.results = getattr(solution_instance, 'results', {}) or {}
        # All expected keys
        self.f_l = self.results.get('f_l', None)
        self.f_m = self.results.get('f_m', None)
        self.num_stages = self.results.get('num_stages', None)
        self.incident_dir_1 = self.results.get('incident_dir_1', None)
        self.incident_dir_2 = self.results.get('incident_dir_2', None)
        self.incident_dir_3 = self.results.get('incident_dir_3', None)
        self.q1 = self.results.get('q1', None)
        self.m1 = self.results.get('m1', None)
        self.p2 = self.results.get('p2', None)
        self.q2 = self.results.get('q2', None)
        self.m2 = self.results.get('m2', None)
        self.p3 = self.results.get('p3', None)
        self.q3 = self.results.get('q3', None)
        self.m_total = self.results.get('m_total', None)
        self.final_location = self.results.get('final_location', None)

    # ------------------------------------------------------------------
    # Helper: compute system values from given parameters
    # ------------------------------------------------------------------
    def _compute_stage1(self, d_o, f_l):
        """Return (q1, m1) for stage 1 (lens)."""
        if d_o is None or f_l is None:
            return None, None
        q1 = 1 / (1/f_l - 1/d_o)
        m1 = -q1 / d_o
        return q1, m1

    def _compute_stage2(self, p2, f_m):
        """Return (q2, m2) for stage 2 (mirror)."""
        if p2 is None or f_m is None:
            return None, None
        q2 = 1 / (1/f_m - 1/p2)
        m2 = -q2 / p2
        return q2, m2

    def _compute_stage3(self, p3, f_l):
        """Return (q3, m3) for stage 3 (lens again)."""
        if p3 is None or f_l is None:
            return None, None
        q3 = 1 / (1/f_l - 1/p3)
        m3 = -q3 / p3
        return q3, m3

    # ------------------------------------------------------------------
    # Interface checks
    # ------------------------------------------------------------------
    def test_interface_required_keys_exist(self):
        """Check that all expected result keys are present and have correct types."""
        expected_keys = {
            'f_l': (int, float),
            'f_m': (int, float),
            'num_stages': int,
            'incident_dir_1': str,
            'incident_dir_2': str,
            'incident_dir_3': str,
            'q1': (int, float),
            'm1': (int, float),
            'p2': (int, float),
            'q2': (int, float),
            'm2': (int, float),
            'p3': (int, float),
            'q3': (int, float),
            'm_total': (int, float),
            'final_location': (int, float)
        }
        missing = []
        wrong_type = []
        for key, exp_type in expected_keys.items():
            val = getattr(self, key, None)
            if val is None:
                missing.append(key)
            elif not isinstance(val, exp_type):
                wrong_type.append(f"{key} expected {exp_type}, got {type(val)}")
        msg_parts = []
        if missing:
            msg_parts.append(f"Missing keys: {missing}")
        if wrong_type:
            msg_parts.append(f"Wrong types: {wrong_type}")
        if msg_parts:
            return False, "; ".join(msg_parts)
        return True, "All required keys present and type-correct"

    # ------------------------------------------------------------------
    # Invariant checks (physics laws)
    # ------------------------------------------------------------------
    def test_invariant_lens_maker_focal(self):
        """Verify f_l using lens maker formula: 1/f_l = (n-1)*(1/R1 - 1/R2)."""
        if self.f_l is None or self.n is None or self.R is None:
            return False, "SKIP: missing n, R, or f_l"
        expected_inv = (self.n - 1) * (1/self.R - 1/(-self.R))
        actual_inv = 1 / self.f_l
        if not math.isclose(expected_inv, actual_inv, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"1/f_l = {actual_inv} != expected {expected_inv} (from lens maker)"
        return True, f"1/f_l = {actual_inv}, consistent with lens maker"

    def test_invariant_mirror_focal(self):
        """Verify f_m = R/2 (concave mirror)."""
        if self.f_m is None or self.R is None:
            return False, "SKIP: missing R or f_m"
        expected_fm = self.R / 2
        if not math.isclose(self.f_m, expected_fm, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"f_m = {self.f_m} != expected {expected_fm} (R/2)"
        return True, f"f_m = {self.f_m}, matches R/2"

    def test_invariant_imaging_stage1(self):
        """Check 1/p1 + 1/q1 = 1/f_l for stage 1."""
        if self.q1 is None or self.f_l is None or self.d_o is None:
            return False, "SKIP: missing d_o, q1, or f_l"
        p1 = self.d_o
        lhs = 1/p1 + 1/self.q1
        rhs = 1/self.f_l
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"1/{p1} + 1/{self.q1} = {lhs} != 1/{self.f_l} = {rhs}"
        return True, f"Stage1 imaging equation satisfied"

    def test_invariant_imaging_stage2(self):
        """Check 1/p2 + 1/q2 = 1/f_m for stage 2."""
        if self.p2 is None or self.q2 is None or self.f_m is None:
            return False, "SKIP: missing p2, q2, or f_m"
        lhs = 1/self.p2 + 1/self.q2
        rhs = 1/self.f_m
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"1/{self.p2} + 1/{self.q2} = {lhs} != 1/{self.f_m} = {rhs}"
        return True, f"Stage2 imaging equation satisfied"

    def test_invariant_imaging_stage3(self):
        """Check 1/p3 + 1/q3 = 1/f_l for stage 3."""
        if self.p3 is None or self.q3 is None or self.f_l is None:
            return False, "SKIP: missing p3, q3, or f_l"
        lhs = 1/self.p3 + 1/self.q3
        rhs = 1/self.f_l
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"1/{self.p3} + 1/{self.q3} = {lhs} != 1/{self.f_l} = {rhs}"
        return True, f"Stage3 imaging equation satisfied"

    def test_invariant_magnification_stage1(self):
        """Check m1 = -q1/p1"""
        if self.m1 is None or self.q1 is None or self.d_o is None:
            return False, "SKIP: missing m1, q1, or d_o"
        expected_m1 = -self.q1 / self.d_o
        if not math.isclose(self.m1, expected_m1, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"m1 = {self.m1} != expected {expected_m1}"
        return True, f"m1 = {self.m1}, matches -q1/p1"

    def test_invariant_magnification_stage2(self):
        """Check m2 = -q2/p2"""
        if self.m2 is None or self.q2 is None or self.p2 is None:
            return False, "SKIP: missing m2, q2, or p2"
        expected_m2 = -self.q2 / self.p2
        if not math.isclose(self.m2, expected_m2, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"m2 = {self.m2} != expected {expected_m2}"
        return True, f"m2 = {self.m2}, matches -q2/p2"

    def test_invariant_magnification_stage3(self):
        """Check m3 = -q3/p3"""
        if self.m3 is None or self.q3 is None or self.p3 is None:
            return False, "SKIP: missing m3, q3, or p3"
        expected_m3 = -self.q3 / self.p3
        if not math.isclose(self.m3, expected_m3, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"m3 = {self.m3} != expected {expected_m3}"
        return True, f"m3 = {self.m3}, matches -q3/p3"

    def test_invariant_total_magnification(self):
        """Check m_total = m1 * m2 * m3"""
        if self.m_total is None or self.m1 is None or self.m2 is None or self.m3 is None:
            return False, "SKIP: missing one of m_total, m1, m2, m3"
        product = self.m1 * self.m2 * self.m3
        if not math.isclose(self.m_total, product, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"m_total = {self.m_total} != product {product}"
        return True, f"m_total = {self.m_total}, equals m1*m2*m3"

    # ------------------------------------------------------------------
    # Modeling quality checks (based on explanation steps)
    # ------------------------------------------------------------------
    def test_modeling_number_of_stages(self):
        """Check that num_stages is 3."""
        if self.num_stages is None:
            return False, "SKIP: num_stages missing"
        if self.num_stages != 3:
            return False, f"num_stages = {self.num_stages}, expected 3"
        return True, "num_stages = 3"

    def test_modeling_incident_directions(self):
        """Check incident directions: left, left, right."""
        if any(d is None for d in [self.incident_dir_1, self.incident_dir_2, self.incident_dir_3]):
            return False, "SKIP: incident direction(s) missing"
        expected = ('left', 'left', 'right')
        actual = (self.incident_dir_1, self.incident_dir_2, self.incident_dir_3)
        if actual != expected:
            return False, f"Incident directions {actual} != expected {expected}"
        return True, f"Incident directions correct: {actual}"

    def test_modeling_object_distances(self):
        """Check that p2 = -q1 and p3 = -q2 (virtual objects)."""
        if self.p2 is None or self.q1 is None or self.p3 is None or self.q2 is None:
            return False, "SKIP: missing p2, q1, p3, or q2"
        if not math.isclose(self.p2, -self.q1, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"p2 = {self.p2} != -q1 = {-self.q1}"
        if not math.isclose(self.p3, -self.q2, rel_tol=1e-2, abs_tol=1e-2):
            return False, f"p3 = {self.p3} != -q2 = {-self.q2}"
        return True, f"p2 = -q1, p3 = -q2"

    # ------------------------------------------------------------------
    # Monotonicity checks (derivative/trend checks)
    # ------------------------------------------------------------------
    def test_monotonicity_image_distance_with_object_distance_stage1(self):
        """For stage1 (lens), as p1 increases (beyond f), q1 decreases.
           Baseline: p1 = d_o = 84, q1 = 84.  Perturb p1 upward by 10% → q1 should be smaller."""
        if self.d_o is None or self.f_l is None or self.q1 is None:
            return False, "SKIP: missing d_o, f_l, or q1"
        p1_base = self.d_o
        q1_base = self.q1
        # apply +10%
        p1_new = p1_base * 1.1
        # compute q1_new using formula (assuming f_l is fixed)
        # ensure p1_new > f_l (it will be 92.4 > 42)
        q1_new = 1 / (1/self.f_l - 1/p1_new)
        if math.isnan(q1_new) or math.isinf(q1_new):
            return False, "Computed q1_new is invalid"
        if q1_new >= q1_base:
            return False, f"Monotonicity violated: q1_new ({q1_new}) >= q1_base ({q1_base})"
        return True, f"Perturb p1 from {p1_base} to {p1_new}, q1 decreases from {q1_base} to {q1_new}"

    def test_monotonicity_image_distance_with_mirror_object_distance(self):
        """For stage2 (mirror), as p2 becomes more negative, q2 decreases.
           Baseline p2 = -84, q2 = 16.8.  Perturb p2 to -92.4 (more negative) → q2 should be smaller."""
        if self.p2 is None or self.f_m is None or self.q2 is None:
            return False, "SKIP: missing p2, f_m, or q2"
        p2_base = self.p2
        q2_base = self.q2
        # make p2 more negative by 10% (absolute value increases)
        p2_new = p2_base * 1.1   # becomes -92.4
        # compute q2_new
        q2_new = 1 / (1/self.f_m - 1/p2_new)
        if math.isnan(q2_new) or math.isinf(q2_new):
            return False, "Computed q2_new is invalid"
        if q2_new >= q2_base:
            return False, f"Monotonicity violated: q2_new ({q2_new}) >= q2_base ({q2_base})"
        return True, f"Perturb p2 from {p2_base} to {p2_new}, q2 decreases from {q2_base} to {q2_new}"

    # ------------------------------------------------------------------
    # Weak answer sanity check (only one allowed)
    # ------------------------------------------------------------------
    def test_answer_sanity_final_image_properties(self):
        """Weak sanity: final_location should be negative (left of lens) and m_total negative (inverted)."""
        if self.final_location is None or self.m_total is None:
            return False, "SKIP: missing final_location or m_total"
        # Do NOT compute expected values; just check sign and finite
        if not (isinstance(self.final_location, (int, float)) and math.isfinite(self.final_location)):
            return False, "final_location is not a finite number"
        if not (isinstance(self.m_total, (int, float)) and math.isfinite(self.m_total)):
            return False, "m_total is not a finite number"
        if self.final_location >= 0:
            return False, f"final_location = {self.final_location} should be negative (left of lens)"
        if self.m_total >= 0:
            return False, f"m_total = {self.m_total} should be negative (inverted)"
        # Check magnitude: |m_total| should be less than 1 (smaller image) - weak
        if abs(self.m_total) > 1:
            return False, f"|m_total| = {abs(self.m_total)} > 1, but image should be smaller"
        return True, "final_location negative, m_total negative and |m|<1 (weak sanity passed)"

    # ------------------------------------------------------------------
    # Runner: discover all test_ methods and execute
    # ------------------------------------------------------------------
    def run_all_tests(self):
        logs = []
        all_pass = True
        for name in sorted(dir(self)):
            if not name.startswith("test_"):
                continue
            method = getattr(self, name, None)
            if not callable(method):
                continue
            try:
                passed, msg = method()
            except Exception as e:
                passed = False
                msg = f"{name} CRASHED: {e}\n{traceback.format_exc()}"
            logs.append(f"{'PASS' if passed else 'FAIL'}: {name}: {msg}")
            if not passed:
                all_pass = False
        return all_pass, logs