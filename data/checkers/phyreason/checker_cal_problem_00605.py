import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from the solution instance.
        Use getattr and .get() to avoid AttributeError.
        """
        # Inputs (with safe defaults, though problem provides them)
        self.m = getattr(solution_instance, 'm', 60.0)  # kg
        self.t = getattr(solution_instance, 't', 1.0)   # s per phase
        self.v0 = getattr(solution_instance, 'v0', 4.0) # m/s
        self.S_total = getattr(solution_instance, 'S_total', 20.0)  # m

        # Results dictionary (expected from solution_instance.results)
        results = getattr(solution_instance, 'results', {})

        # Extract with safe defaults (0.0 if missing, but should be present)
        self.f = results.get('f', 0.0)          # resistance force (N)
        self.a1 = results.get('a_1', 0.0)       # acceleration during power (m/s²)
        self.F = results.get('F', 0.0)          # driving force (N)
        self.ratio_value = results.get('a_1:a_2', 0.0)  # could be string or numeric
        self.v1 = results.get('v_1', 0.0)       # max speed (m/s)
        self.F_minus_f = results.get('F-f', 0.0) # net force during power (N)
        self.a2 = results.get('a_2', 0.0)       # magnitude of recovery acceleration (m/s²)
        self.x1 = results.get('x_1', 0.0)       # displacement in one power phase (m)

        # Additional derived quantities for checks
        self.a2_actual = -self.a2  # actual acceleration during recovery (negative)

    def test_invariants(self) -> tuple[bool, list[str]]:
        """Verify Newton's laws, energy/work, and given conditions."""
        logs = []
        success = True

        # 1. Check given condition: F = 2f
        if not math.isclose(self.F, 2 * self.f, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F = {self.F} ≠ 2*f = {2 * self.f}")
            success = False
        else:
            logs.append("PASS: F = 2f")

        # 2. Newton's second law during power phase: F - f = m * a1
        lhs = self.F - self.f
        rhs = self.m * self.a1
        if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F - f = {lhs} ≠ m*a1 = {rhs}")
            success = False
        else:
            logs.append("PASS: F - f = m*a1")

        # 3. Newton's second law during recovery phase: f = m * |a2| (magnitude)
        #    Net force is -f, acceleration is -a2 (a2 positive).
        if not math.isclose(self.f, self.m * self.a2, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: f = {self.f} ≠ m*a2 = {self.m * self.a2}")
            success = False
        else:
            logs.append("PASS: f = m*a2 (magnitude)")

        # 4. Work-energy over two strokes: net work = 0 (speed returns to v0)
        #    Work by driving force: F * (2 * self.x1)  (only during two power phases)
        #    Work by resistance: -self.f * self.S_total
        work_drive = self.F * (2 * self.x1)
        work_resist = -self.f * self.S_total
        net_work = work_drive + work_resist
        if not math.isclose(net_work, 0.0, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Net work = {net_work} ≠ 0 (work_drive={work_drive}, work_resist={work_resist})")
            success = False
        else:
            logs.append("PASS: Net work over two strokes is zero (consistent kinetic energy cycle)")

        return success, logs

    def test_trace_consistency(self) -> tuple[bool, list[str]]:
        """Check kinematic consistency and intermediate results."""
        logs = []
        success = True

        # 1. Check displacement during one power phase: x1 = v0*t + 0.5*a1*t^2
        x1_calc = self.v0 * self.t + 0.5 * self.a1 * self.t**2
        if not math.isclose(self.x1, x1_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: x1 = {self.x1} ≠ v0*t + 0.5*a1*t² = {x1_calc}")
            success = False
        else:
            logs.append("PASS: x1 from kinematics")

        # 2. Check max speed: v1 = v0 + a1*t
        v1_calc = self.v0 + self.a1 * self.t
        if not math.isclose(self.v1, v1_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v1 = {self.v1} ≠ v0 + a1*t = {v1_calc}")
            success = False
        else:
            logs.append("PASS: v1 = v0 + a1*t")

        # 3. Check ratio a1:a2 (should be 1:1)
        if self.a2 != 0:
            ratio = self.a1 / self.a2
            if not math.isclose(ratio, 1.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: a1/a2 = {ratio} ≠ 1")
                success = False
            else:
                logs.append("PASS: a1:a2 = 1:1 (numerical)")
        else:
            logs.append("SKIP: a2 is zero, cannot compute ratio")
            success = False

        # Check ratio representation (handle both string and numeric)
        if isinstance(self.ratio_value, str):
            if self.ratio_value.strip() != "1:1":
                logs.append(f"FAIL: ratio string '{self.ratio_value}' ≠ '1:1'")
                success = False
            else:
                logs.append("PASS: ratio string is '1:1'")
        else:
            # Numeric ratio - check if it's approximately 1.0
            if not math.isclose(float(self.ratio_value), 1.0, rel_tol=1e-2, abs_tol=1e-2):
                logs.append(f"FAIL: ratio value {self.ratio_value} ≠ 1.0")
                success = False
            else:
                logs.append("PASS: ratio value is 1.0")

        # 4. Check total displacement over two strokes (4 phases)
        #    Compute displacements for each phase:
        #    Phase 1 (power): x1 (already checked)
        #    Phase 2 (recovery): start at v1, acceleration = -a2, t=1
        x2 = self.v1 * self.t + 0.5 * self.a2_actual * self.t**2
        v2 = self.v1 + self.a2_actual * self.t
        #    Phase 3 (power): start at v2, acceleration = a1
        x3 = v2 * self.t + 0.5 * self.a1 * self.t**2
        v3 = v2 + self.a1 * self.t
        #    Phase 4 (recovery): start at v3, acceleration = -a2
        x4 = v3 * self.t + 0.5 * self.a2_actual * self.t**2
        total_calc = self.x1 + x2 + x3 + x4
        if not math.isclose(self.S_total, total_calc, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Total displacement {self.S_total} ≠ sum of phases {total_calc}")
            logs.append(f"      x1={self.x1}, x2={x2}, x3={x3}, x4={x4}")
            success = False
        else:
            logs.append("PASS: Total displacement matches sum of four phases")

        return success, logs

    def test_monotonicity(self) -> tuple[bool, list[str]]:
        """
        Perturb mass, recalculate accelerations and forces, and check scaling.
        Restore original mass after.
        """
        logs = []
        success = True

        original_m = self.m
        # Perturb mass by +10%
        perturbed_m = original_m * 1.1  # 66 kg

        # The acceleration magnitude a is determined by kinematics independent of m.
        # From total displacement: S_total = 4*v0*t + 2*a*t^2  (since each of 4 phases gives same displacement formula)
        # So a = (S_total - 4*v0*t) / (2*t**2)
        a_calc = (self.S_total - 4 * self.v0 * self.t) / (2 * self.t**2)
        # This should be 2.0 m/s² with given numbers.

        # Compute forces using the perturbed mass:
        f_calc = perturbed_m * a_calc
        F_calc = 2 * f_calc

        # Check that with perturbed mass, the forces scale proportionally:
        # f_new / f_original = m_new / m_original
        if not math.isclose(f_calc / self.f, perturbed_m / original_m, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: f does not scale linearly with m (f_ratio={f_calc/self.f}, m_ratio={perturbed_m/original_m})")
            success = False
        else:
            logs.append("PASS: f scales linearly with m")

        if not math.isclose(F_calc / self.F, perturbed_m / original_m, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: F does not scale linearly with m (F_ratio={F_calc/self.F}, m_ratio={perturbed_m/original_m})")
            success = False
        else:
            logs.append("PASS: F scales linearly with m")

        # Acceleration should remain the same (independent of m)
        if not math.isclose(a_calc, self.a1, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: Acceleration a changes with m (a_calc={a_calc}, original a1={self.a1})")
            success = False
        else:
            logs.append("PASS: Acceleration magnitude is independent of m")

        # Restore original mass (just in case, though class instance is not reused)
        self.m = original_m
        return success, logs

    def run_all_tests(self) -> tuple[bool, list[str]]:
        """Run all checks and return overall success and log."""
        all_logs = ["=== Physics Checker Started ==="]
        overall_success = True

        # Run each test and collect results
        tests = [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]

        for name, test_func in tests:
            all_logs.append(f"\n--- {name} ---")
            try:
                success, logs = test_func()
                overall_success = overall_success and success
                all_logs.extend(logs)
            except Exception as e:
                all_logs.append(f"ERROR during {name}: {e}")
                all_logs.append(traceback.format_exc())
                overall_success = False

        all_logs.append(f"\n=== Overall: {'PASS' if overall_success else 'FAIL'} ===")
        return overall_success, all_logs