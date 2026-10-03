import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extracts results and constants from the solution instance.
        Uses getattr to avoid AttributeError in sandbox environments.
        """
        self.results = getattr(solution_instance, 'results', {})

        # Extract expected results with safe fallbacks
        self.v1 = getattr(solution_instance, 'v_1', self.results.get('v_1', 0.0))
        self.t1 = getattr(solution_instance, 't_1', self.results.get('t_1', 0.0))
        self.t = getattr(solution_instance, 't', self.results.get('t', 0.0))
        self.s_car = getattr(solution_instance, 's_car', self.results.get('s_car', 0.0))
        self.s_stone = getattr(solution_instance, 's_stone', self.results.get('s_stone', 0.0))

        # Problem constants (hardcoded from problem statement)
        self.S1 = 180.0           # distance from mudslide start to bottom (m)
        self.v0_mud = 2.0         # initial mudslide velocity (m/s)
        self.a_mud = 0.7          # mudslide acceleration (m/s²)
        self.reaction_time = 1.0  # driver reaction time (s)
        self.a_car = 0.5          # car acceleration (m/s²)

    def test_invariants(self):
        """
        Checks fundamental kinematic equations using fuzzy numerical matching.
        """
        logs = []
        success = True

        # 1. v1 from v1^2 = v0^2 + 2*a*S1
        v1_expected = math.sqrt(self.v0_mud**2 + 2 * self.a_mud * self.S1)
        if not math.isclose(self.v1, v1_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: v1 = {self.v1}, expected {v1_expected:.2f} from v1² = v0² + 2*a*S1")
            success = False
        else:
            logs.append(f"PASS: v1 = {self.v1} matches kinematic equation.")

        # 2. t1 from v1 = v0 + a_mud * t1
        t1_expected = (self.v1 - self.v0_mud) / self.a_mud
        if not math.isclose(self.t1, t1_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: t1 = {self.t1}, expected {t1_expected:.2f} from v1 = v0 + a_mud*t1")
            success = False
        else:
            logs.append(f"PASS: t1 = {self.t1} matches velocity-time relation.")

        # 3. t from v1 = a_car * t (car starts from rest after reaction)
        t_expected = self.v1 / self.a_car
        if not math.isclose(self.t, t_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: t = {self.t}, expected {t_expected:.2f} from v1 = a_car*t")
            success = False
        else:
            logs.append(f"PASS: t = {self.t} matches car acceleration to v1.")

        # 4. s_car from s = 0.5 * a_car * t^2 (t is time accelerating)
        s_car_expected = 0.5 * self.a_car * self.t**2
        if not math.isclose(self.s_car, s_car_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: s_car = {self.s_car}, expected {s_car_expected:.2f} from 0.5*a_car*t²")
            success = False
        else:
            logs.append(f"PASS: s_car = {self.s_car} matches displacement under constant acceleration.")

        # 5. s_stone from uniform motion after reaching bottom
        #    s_stone = v1 * (reaction_time + t - t1)
        s_stone_expected = self.v1 * (self.reaction_time + self.t - self.t1)
        if not math.isclose(self.s_stone, s_stone_expected, rel_tol=1e-2, abs_tol=1e-2):
            logs.append(f"FAIL: s_stone = {self.s_stone}, expected {s_stone_expected:.2f} from v1*(t_reaction+t - t1)")
            success = False
        else:
            logs.append(f"PASS: s_stone = {self.s_stone} matches mudslide uniform motion.")

        return success, logs

    def test_trace_consistency(self):
        """
        Verifies that intermediate results lead to a physically consistent conclusion.
        """
        logs = []
        success = True

        # Check that the car is ahead of the mudslide at the time it reaches v1
        if self.s_car <= self.s_stone:
            logs.append(f"FAIL: Car distance ({self.s_car} m) not greater than mudslide distance ({self.s_stone} m).")
            success = False
        else:
            logs.append(f"PASS: Car leads mudslide by {self.s_car - self.s_stone:.1f} m.")

        # Verify temporal ordering: car reaches v1 after mudslide reaches bottom
        t_car_v1 = self.reaction_time + self.t  # absolute time when car reaches v1
        if not t_car_v1 > self.t1:
            logs.append(f"FAIL: Car reaches v1 at t={t_car_v1} s, but mudslide reaches bottom at t1={self.t1} s.")
            success = False
        else:
            logs.append(f"PASS: Car reaches v1 after mudslide reaches bottom ({t_car_v1} > {self.t1}).")

        # Cross-check using symbolic equations (fallback to numeric if symbolic fails)
        try:
            t_sym = math.sqrt(2 * self.s_car / self.a_car)
            if not math.isclose(t_sym, self.t, rel_tol=1e-2):
                logs.append(f"WARNING: Numeric solve for t from s_car gives {t_sym}, but t is {self.t}")
        except Exception as e:
            logs.append(f"Symbolic cross-check skipped: {e}")

        return success, logs

    def test_monotonicity(self):
        """
        Perturbs an input parameter (car acceleration) and verifies expected trends.
        CORRECTED: s_car = v1²/(2*a_car) decreases as a_car increases.
        """
        logs = []
        success = True

        # Perturb car acceleration upward by 10%
        a_car_perturbed = self.a_car * 1.10
        t_perturbed = self.v1 / a_car_perturbed
        s_car_perturbed = 0.5 * a_car_perturbed * t_perturbed**2
        # s_stone changes because t changes: s_stone = v1*(reaction_time + t_perturbed - t1)
        s_stone_perturbed = self.v1 * (self.reaction_time + t_perturbed - self.t1)

        # Check that increasing acceleration DECREASES car distance (s_car = v1²/(2*a_car))
        if not s_car_perturbed < self.s_car:
            logs.append(f"FAIL: Increasing a_car should decrease s_car (original {self.s_car}, perturbed {s_car_perturbed})")
            success = False
        else:
            logs.append(f"PASS: Increasing a_car decreases s_car ({self.s_car} → {s_car_perturbed:.1f}).")

        # Check that safety margin (car - mudslide) changes appropriately
        margin_original = self.s_car - self.s_stone
        margin_perturbed = s_car_perturbed - s_stone_perturbed
        
        # The safety margin should increase because while s_car decreases, s_stone decreases more
        # (since t_perturbed < self.t, so (reaction_time + t_perturbed - t1) decreases)
        if not margin_perturbed > margin_original:
            logs.append(f"FAIL: Safety margin should increase (original {margin_original}, perturbed {margin_perturbed})")
            success = False
        else:
            logs.append(f"PASS: Safety margin increases ({margin_original} → {margin_perturbed:.1f} m).")

        return success, logs

    def run_all_tests(self):
        """
        Executes all tests and returns overall success and log messages.
        """
        all_success = True
        logs = []

        tests = [
            self.test_invariants,
            self.test_trace_consistency,
            self.test_monotonicity
        ]

        for test_func in tests:
            try:
                test_name = test_func.__name__
                logs.append(f"--- {test_name} ---")
                success, test_logs = test_func()
                logs.extend(test_logs)
                if not success:
                    all_success = False
            except Exception as e:
                logs.append(f"ERROR in {test_func.__name__}: {e}")
                all_success = False

        logs.append(f"\nOverall test result: {'PASS' if all_success else 'FAIL'}")
        return all_success, logs