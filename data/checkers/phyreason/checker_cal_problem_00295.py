import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from the solution instance.
        """
        # Physical constants and inputs (from problem statement)
        self.L = 1.5          # m, box length
        self.h = 0.45         # m, box height
        self.M = 10.0         # kg, box mass
        self.v0 = 3.6         # m/s, initial velocity to the right
        self.F = 50.0         # N, applied force to the left
        self.m = 1.0          # kg, ball mass
        self.mu = 0.2         # coefficient of kinetic friction
        self.g = 10.0         # m/s^2, gravity
        
        # Safely extract results from the solution instance
        # Using getattr with fallback to results dictionary
        self.results = getattr(solution_instance, 'results', {})
        
        # Extract specific result quantities with safe defaults
        self.a1 = getattr(solution_instance, 'a_1', self.results.get('a_1', None))
        self.v2 = getattr(solution_instance, 'v_2', self.results.get('v_2', None))
        self.t2 = getattr(solution_instance, 't_2', self.results.get('t_2', None))
        self.t_fall = getattr(solution_instance, 't', self.results.get('t', None))
        self.a2 = getattr(solution_instance, 'a_2', self.results.get('a_2', None))
        self.x2 = getattr(solution_instance, 'x_2', self.results.get('x_2', None))
        self.x_max = getattr(solution_instance, 'x_max', self.results.get('x_max', None))
        self.v_ball = getattr(solution_instance, 'v_ball', self.results.get('v_ball', None))
        self.x1 = getattr(solution_instance, 'x_1', self.results.get('x_1', None))
        
        # Initialize logs
        self.logs = []
    
    def log(self, message, is_error=False):
        """Helper to log messages."""
        self.logs.append(f"{'ERROR' if is_error else 'INFO'}: {message}")
    
    def is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy numerical comparison."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Verify conservation laws and Newton's laws."""
        self.log("=== Testing Invariants ===")
        
        # 1. Check Newton's 2nd Law for Phase 1 (box with ball)
        if self.a1 is not None:
            # Net force from problem: F_left + friction_left (both oppose motion to the right)
            # Friction = mu * (M+m) * g, direction opposite to motion (left when box moves right)
            net_force_expected = self.F + self.mu * (self.M + self.m) * self.g
            net_force_from_a1 = self.M * self.a1  # Only M accelerates (ball slides without friction)
            
            if not self.is_close(net_force_expected, net_force_from_a1):
                self.log(f"Newton's 2nd Law (Phase 1) failed: expected net force {net_force_expected}, got {net_force_from_a1}", is_error=True)
            else:
                self.log(f"Newton's 2nd Law (Phase 1) satisfied: net force ≈ {net_force_from_a1:.2f} N")
        
        # 2. Check Newton's 2nd Law for Phase 2 (box without ball, after ball falls)
        if self.a2 is not None:
            # After ball falls, mass is just M, friction is mu*M*g opposing leftward motion (so to the right)
            net_force_expected = self.F - self.mu * (self.M + self.m) * self.g  # Ball still on box until it leaves
            net_force_from_a2 = self.M * self.a2
            
            if not self.is_close(net_force_expected, net_force_from_a2):
                self.log(f"Newton's 2nd Law (Phase 2) failed: expected net force {net_force_expected}, got {net_force_from_a2}", is_error=True)
            else:
                self.log(f"Newton's 2nd Law (Phase 2) satisfied: net force ≈ {net_force_from_a2:.2f} N")
        
        # 3. Check energy conservation (work-energy theorem) for Phase 1
        if self.x1 is not None and self.a1 is not None:
            # Initial KE of box: 0.5 * M * v0^2
            initial_ke = 0.5 * self.M * self.v0**2
            # Work done by friction and applied force (both negative as force opposite displacement)
            work_friction = -self.mu * (self.M + self.m) * self.g * self.x1
            work_F = -self.F * self.x1
            total_work = work_friction + work_F
            # Change in KE should equal total work
            if not self.is_close(total_work, -initial_ke):
                self.log(f"Work-energy theorem (Phase 1) failed: ΔKE = {-initial_ke:.2f} J, total work = {total_work:.2f} J", is_error=True)
            else:
                self.log(f"Work-energy theorem (Phase 1) satisfied: work = {total_work:.2f} J, ΔKE = {-initial_ke:.2f} J")
        
        # 4. Check free fall time
        if self.t_fall is not None:
            t_expected = math.sqrt(2 * self.h / self.g)
            if not self.is_close(self.t_fall, t_expected):
                self.log(f"Free fall time mismatch: expected {t_expected:.3f} s, got {self.t_fall}", is_error=True)
            else:
                self.log(f"Free fall time correct: t = {self.t_fall:.3f} s")
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        self.log("=== Testing Trace Consistency ===")
        
        # 1. Check x1 (maximum displacement) from kinematics
        if self.a1 is not None and self.x1 is not None:
            x1_from_kinematics = self.v0**2 / (2 * self.a1)
            if not self.is_close(self.x1, x1_from_kinematics):
                self.log(f"x1 inconsistency: from kinematics {x1_from_kinematics:.3f} m, got {self.x1}", is_error=True)
            else:
                self.log(f"x1 consistent: {self.x1:.3f} m")
        
        # 2. Check x2 = x1 + L/3 (from step 7)
        if self.x1 is not None and self.x2 is not None:
            x2_expected = self.x1 + self.L / 3
            if not self.is_close(self.x2, x2_expected):
                self.log(f"x2 inconsistency: expected {x2_expected:.3f} m, got {self.x2}", is_error=True)
            else:
                self.log(f"x2 consistent: {self.x2:.3f} m")
        
        # 3. Check t2 = sqrt(2*x2/a2)
        if self.x2 is not None and self.a2 is not None and self.t2 is not None:
            t2_expected = math.sqrt(2 * self.x2 / self.a2)
            if not self.is_close(self.t2, t2_expected):
                self.log(f"t2 inconsistency: expected {t2_expected:.3f} s, got {self.t2}", is_error=True)
            else:
                self.log(f"t2 consistent: {self.t2:.3f} s")
        
        # 4. Check v2 = a2 * t2
        if self.a2 is not None and self.t2 is not None and self.v2 is not None:
            v2_expected = self.a2 * self.t2
            if not self.is_close(self.v2, v2_expected):
                self.log(f"v2 inconsistency: expected {v2_expected:.3f} m/s, got {self.v2}", is_error=True)
            else:
                self.log(f"v2 consistent: {self.v2:.3f} m/s")
        
        # 5. Compare with target answers
        target_t_fall = 0.3
        target_x_max = 0.9
        target_v2 = 2.8
        
        if self.t_fall is not None and not self.is_close(self.t_fall, target_t_fall):
            self.log(f"Target answer mismatch: t_fall = {self.t_fall} s, expected {target_t_fall} s", is_error=True)
        
        if self.x_max is not None and not self.is_close(self.x_max, target_x_max):
            self.log(f"Target answer mismatch: x_max = {self.x_max} m, expected {target_x_max} m", is_error=True)
        
        if self.v2 is not None and not self.is_close(self.v2, target_v2):
            self.log(f"Target answer mismatch: v2 = {self.v2} m/s, expected {target_v2} m/s", is_error=True)
    
    def test_monotonicity(self):
        """Perturb an input and check trend, then restore."""
        self.log("=== Testing Monotonicity ===")
        
        # Perturb ball mass m by +10%
        original_m = self.m
        perturbed_m = original_m * 1.1
        
        # Recalculate a1, x1, a2 with perturbed mass (using formulas from steps)
        # Phase 1 acceleration: a1 = [F + mu*(M+m)*g] / M
        a1_perturbed = (self.F + self.mu * (self.M + perturbed_m) * self.g) / self.M
        # Maximum displacement: x1 = v0^2 / (2*a1)
        x1_perturbed = self.v0**2 / (2 * a1_perturbed)
        # Phase 2 acceleration: a2 = [F - mu*(M+m)*g] / M
        a2_perturbed = (self.F - self.mu * (self.M + perturbed_m) * self.g) / self.M
        
        # Expected trend: increasing m increases friction, so a1 increases, x1 decreases
        if self.a1 is not None and self.x1 is not None:
            if a1_perturbed > self.a1:
                self.log("Monotonicity: a1 increases with increased ball mass ✓")
            else:
                self.log("Monotonicity: a1 did not increase with increased ball mass", is_error=True)
            
            if x1_perturbed < self.x1:
                self.log("Monotonicity: x1 decreases with increased ball mass ✓")
            else:
                self.log("Monotonicity: x1 did not decrease with increased ball mass", is_error=True)
            
            if a2_perturbed < self.a2:
                self.log("Monotonicity: a2 decreases with increased ball mass ✓")
            else:
                self.log("Monotonicity: a2 did not decrease with increased ball mass", is_error=True)
        else:
            self.log("Skipping monotonicity test due to missing a1 or x1", is_error=True)
    
    def run_all_tests(self):
        """Execute all tests and return success flag and logs."""
        self.logs = []
        success = True
        
        try:
            self.test_invariants()
            self.test_trace_consistency()
            self.test_monotonicity()
            
            # Check for any errors in logs
            for log_entry in self.logs:
                if log_entry.startswith("ERROR"):
                    success = False
                    break
        except Exception as e:
            self.log(f"Exception during testing: {str(e)}\n{traceback.format_exc()}", is_error=True)
            success = False
        
        return success, self.logs