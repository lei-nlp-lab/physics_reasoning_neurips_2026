import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        # Safely extract inputs
        self.m = getattr(solution_instance, 'm', None)
        self.q = getattr(solution_instance, 'q', None)
        self.L = getattr(solution_instance, 'L', None)
        self.g = getattr(solution_instance, 'g', None)
        self.theta = getattr(solution_instance, 'theta', None)
        
        # If theta is not provided, assume 60 degrees in radians
        if self.theta is None:
            self.theta = math.radians(60)
        elif isinstance(self.theta, (int, float)):
            # Assume it's in degrees if it's a number
            self.theta = math.radians(self.theta)
        
        # Safely extract results using .results dictionary
        results = getattr(solution_instance, 'results', {})
        self.E = results.get('E', 0.0)
        self.F = results.get('F', 0.0)
        
        # Store original values for monotonicity test
        self._original_m = self.m
        self._original_E = self.E
        self._original_F = self.F

    def test_invariants(self):
        logs = []
        success = True
        
        try:
            # Check 1: Energy conservation from A to B (both points at rest)
            # Work done by electric field: q*E*(horizontal displacement)
            # Horizontal displacement = L*sin(theta)
            # Work done by gravity: -m*g*(vertical displacement)
            # Vertical displacement = L*(1 - cos(theta))
            # Net work should be zero: q*E*L*sin(theta) - m*g*L*(1-cos(theta)) = 0
            
            if all(v is not None for v in [self.m, self.q, self.L, self.g, self.E]):
                electric_work = self.q * self.E * self.L * math.sin(self.theta)
                gravity_work = self.m * self.g * self.L * (1 - math.cos(self.theta))
                net_work = electric_work - gravity_work
                
                if not math.isclose(net_work, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Energy conservation failed: net_work = {net_work:.6f}")
                    success = False
                else:
                    logs.append(f"Energy conservation passed: net_work = {net_work:.6f}")
            else:
                logs.append("Energy conservation skipped due to missing parameters")
            
            # Check 2: Newton's second law at equilibrium point (θ_eq = 30°)
            # Tangential force should be zero: -m*g*sin(30°) + q*E*cos(30°) = 0
            theta_eq = math.radians(30)
            if all(v is not None for v in [self.m, self.q, self.g, self.E]):
                tangential_force = -self.m * self.g * math.sin(theta_eq) + self.q * self.E * math.cos(theta_eq)
                if not math.isclose(tangential_force, 0.0, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Tangential equilibrium failed: F_tang = {tangential_force:.6f}")
                    success = False
                else:
                    logs.append(f"Tangential equilibrium passed: F_tang = {tangential_force:.6f}")
            else:
                logs.append("Tangential equilibrium skipped due to missing parameters")
            
            # Check 3: Radial force balance at equilibrium point using computed tension
            # T - m*g*cos(30°) - q*E*sin(30°) = m*v_max²/L
            # We can compute v_max from energy conservation
            if all(v is not None for v in [self.m, self.q, self.L, self.g, self.E, self.F]):
                # Compute v_max from energy conservation from A to equilibrium point
                # Work from A (θ=0) to θ_eq=30°:
                # ΔKE = work_electric + work_gravity
                # Electric work: q*E*L*sin(θ_eq)
                # Gravity work: -m*g*L*(1-cos(θ_eq))
                # v_max² = 2*(q*E*L*sin(θ_eq) - m*g*L*(1-cos(θ_eq)))/m
                v_sq = 2 * (self.q * self.E * self.L * math.sin(theta_eq) 
                           - self.m * self.g * self.L * (1 - math.cos(theta_eq))) / self.m
                
                # Radial force balance
                lhs = self.F - self.m * self.g * math.cos(theta_eq) - self.q * self.E * math.sin(theta_eq)
                rhs = self.m * v_sq / self.L
                
                if not math.isclose(lhs, rhs, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"Radial force balance failed: LHS={lhs:.6f}, RHS={rhs:.6f}")
                    success = False
                else:
                    logs.append(f"Radial force balance passed: LHS={lhs:.6f}, RHS={rhs:.6f}")
            else:
                logs.append("Radial force balance skipped due to missing parameters")
                
        except Exception as e:
            logs.append(f"Error in test_invariants: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs

    def test_trace_consistency(self):
        logs = []
        success = True
        
        try:
            # Check E from work-energy theorem
            if all(v is not None for v in [self.m, self.g, self.q, self.theta]):
                expected_E = (self.m * self.g * (1 - math.cos(self.theta))) / (self.q * math.sin(self.theta))
                if not math.isclose(self.E, expected_E, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"E consistency failed: computed={self.E:.6f}, expected={expected_E:.6f}")
                    success = False
                else:
                    logs.append(f"E consistency passed: computed={self.E:.6f}, expected={expected_E:.6f}")
            else:
                logs.append("E consistency check skipped due to missing parameters")
            
            # Check F from radial force balance at equilibrium (θ_eq=30°)
            # First compute E if not already consistent
            if all(v is not None for v in [self.m, self.g, self.q, self.theta]):
                # Use the E from above expected_E or the one in results
                E_to_use = self.E
                theta_eq = math.radians(30)
                
                # Compute v_max² as before
                v_sq = 2 * (self.q * E_to_use * self.L * math.sin(theta_eq) 
                           - self.m * self.g * self.L * (1 - math.cos(theta_eq))) / self.m
                
                expected_F = (self.m * self.g * math.cos(theta_eq) + 
                             self.q * E_to_use * math.sin(theta_eq) + 
                             self.m * v_sq / self.L)
                
                if not math.isclose(self.F, expected_F, rel_tol=1e-2, abs_tol=1e-2):
                    logs.append(f"F consistency failed: computed={self.F:.6f}, expected={expected_F:.6f}")
                    success = False
                else:
                    logs.append(f"F consistency passed: computed={self.F:.6f}, expected={expected_F:.6f}")
            else:
                logs.append("F consistency check skipped due to missing parameters")
                
        except Exception as e:
            logs.append(f"Error in test_trace_consistency: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs

    def test_monotonicity(self):
        logs = []
        success = True
        
        try:
            # Only test if all required parameters are available
            if all(v is not None for v in [self.m, self.q, self.L, self.g]):
                # Perturb mass by +1%
                perturbed_m = self.m * 1.01
                
                # Recompute E using formula from work-energy theorem
                # E = (m*g*(1-cosθ))/(q*sinθ)
                perturbed_E = (perturbed_m * self.g * (1 - math.cos(self.theta))) / (self.q * math.sin(self.theta))
                
                # Recompute F using formula at equilibrium point
                theta_eq = math.radians(30)
                # First compute v_max² with perturbed values
                v_sq = 2 * (self.q * perturbed_E * self.L * math.sin(theta_eq) 
                           - perturbed_m * self.g * self.L * (1 - math.cos(theta_eq))) / perturbed_m
                perturbed_F = (perturbed_m * self.g * math.cos(theta_eq) + 
                              self.q * perturbed_E * math.sin(theta_eq) + 
                              perturbed_m * v_sq / self.L)
                
                # Check monotonicity: both E and F should increase with m
                # Since both are proportional to m in their formulas
                if perturbed_E <= self.E:
                    logs.append(f"E monotonicity failed: original={self.E:.6f}, perturbed={perturbed_E:.6f}")
                    success = False
                else:
                    logs.append(f"E monotonicity passed: original={self.E:.6f}, perturbed={perturbed_E:.6f}")
                
                if perturbed_F <= self.F:
                    logs.append(f"F monotonicity failed: original={self.F:.6f}, perturbed={perturbed_F:.6f}")
                    success = False
                else:
                    logs.append(f"F monotonicity passed: original={self.F:.6f}, perturbed={perturbed_F:.6f}")
            else:
                logs.append("Monotonicity test skipped due to missing parameters")
                
        except Exception as e:
            logs.append(f"Error in test_monotonicity: {str(e)}")
            logs.append(traceback.format_exc())
            success = False
            
        return success, logs

    def run_all_tests(self):
        all_logs = []
        overall_success = True
        
        # Run each test
        for test_name, test_method in [
            ("Invariants", self.test_invariants),
            ("Trace Consistency", self.test_trace_consistency),
            ("Monotonicity", self.test_monotonicity)
        ]:
            all_logs.append(f"=== Running {test_name} Test ===")
            success, logs = test_method()
            all_logs.extend(logs)
            if not success:
                overall_success = False
            all_logs.append("")
        
        return overall_success, all_logs