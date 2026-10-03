import sympy as sp
import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """
        Safely extract inputs and results from solution_instance.
        """
        self.g = 10.0  # Given gravitational acceleration
        self.v_final = 30.0  # Given final velocity
        
        # Extract results with fallbacks
        self.results = {}
        keys = ['t', 'v', 'h', 'h_1', 'h_2']
        for key in keys:
            self.results[key] = getattr(solution_instance, key, 
                                        getattr(solution_instance, 'results', {}).get(key, 0.0))
        
        # Convert to float if they're strings or other types
        for key in self.results:
            try:
                self.results[key] = float(self.results[key])
            except (ValueError, TypeError):
                self.results[key] = 0.0
        
        # Initialize symbolic variables for verification
        self.t_sym = sp.symbols('t')
        self.v_sym = sp.symbols('v')
        self.h_sym = sp.symbols('h')
        self.h1_sym = sp.symbols('h_1')
        self.h2_sym = sp.symbols('h_2')
        self.g_sym = sp.symbols('g')
        
        self.logs = []
        
    def log(self, message, success=True):
        """Record test results with status."""
        self.logs.append((success, message))
    
    def test_invariants(self):
        """Verify fundamental physics invariants."""
        try:
            # 1. Energy conservation: mgh = 1/2 mv^2 => gh = 1/2 v^2
            energy_lhs = self.g * self.results['h']
            energy_rhs = 0.5 * self.v_final**2
            if math.isclose(energy_lhs, energy_rhs, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Energy conservation ✓: gh={energy_lhs:.2f} ≈ 0.5v²={energy_rhs:.2f}")
            else:
                self.log(f"Energy conservation ✗: gh={energy_lhs:.2f} ≠ 0.5v²={energy_rhs:.2f}", False)
            
            # 2. Kinematic relation: v = g*t
            v_calculated = self.g * self.results['t']
            if math.isclose(self.v_final, v_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Kinematics v=gt ✓: {self.v_final:.2f} ≈ g*t={v_calculated:.2f}")
            else:
                self.log(f"Kinematics v=gt ✗: {self.v_final:.2f} ≠ g*t={v_calculated:.2f}", False)
            
            # 3. Displacement relation: h = 1/2 g t^2
            h_calculated = 0.5 * self.g * self.results['t']**2
            if math.isclose(self.results['h'], h_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Displacement h=½gt² ✓: {self.results['h']:.2f} ≈ ½gt²={h_calculated:.2f}")
            else:
                self.log(f"Displacement h=½gt² ✗: {self.results['h']:.2f} ≠ ½gt²={h_calculated:.2f}", False)
            
            # 4. Consistency check: h_1 + h_2 = h
            sum_h1_h2 = self.results['h_1'] + self.results['h_2']
            if math.isclose(self.results['h'], sum_h1_h2, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Partition consistency ✓: h={self.results['h']:.2f} = h₁+h₂={sum_h1_h2:.2f}")
            else:
                self.log(f"Partition consistency ✗: h={self.results['h']:.2f} ≠ h₁+h₂={sum_h1_h2:.2f}", False)
                
        except Exception as e:
            self.log(f"Invariants test failed with error: {str(e)}", False)
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        try:
            # Check time calculation: t = v/g
            t_calculated = self.v_final / self.g
            if math.isclose(self.results['t'], t_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Time consistency ✓: t={self.results['t']:.2f} ≈ v/g={t_calculated:.2f}")
            else:
                self.log(f"Time consistency ✗: t={self.results['t']:.2f} ≠ v/g={t_calculated:.2f}", False)
            
            # Check h_1 for first 2 seconds: h_1 = 1/2 g * (2)^2
            h1_calculated = 0.5 * self.g * 4  # t₁ = 2s
            if math.isclose(self.results['h_1'], h1_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"First 2s displacement ✓: h₁={self.results['h_1']:.2f} ≈ ½g*4={h1_calculated:.2f}")
            else:
                self.log(f"First 2s displacement ✗: h₁={self.results['h_1']:.2f} ≠ ½g*4={h1_calculated:.2f}", False)
            
            # Check h_2 calculation: h_2 = h - h_1
            h2_calculated = self.results['h'] - self.results['h_1']
            if math.isclose(self.results['h_2'], h2_calculated, rel_tol=1e-2, abs_tol=1e-2):
                self.log(f"Last 1s displacement ✓: h₂={self.results['h_2']:.2f} ≈ h-h₁={h2_calculated:.2f}")
            else:
                self.log(f"Last 1s displacement ✗: h₂={self.results['h_2']:.2f} ≠ h-h₁={h2_calculated:.2f}", False)
                
        except Exception as e:
            self.log(f"Trace consistency test failed with error: {str(e)}", False)
    
    def test_monotonicity(self):
        """Perturb g and verify monotonic trends, then restore."""
        try:
            original_g = self.g
            perturbed_g = original_g * 1.01  # 1% increase
            
            # Calculate with perturbed g using kinematic formulas
            t_perturbed = self.v_final / perturbed_g
            h_perturbed = 0.5 * perturbed_g * t_perturbed**2
            h1_perturbed = 0.5 * perturbed_g * 4  # first 2 seconds
            h2_perturbed = h_perturbed - h1_perturbed
            
            # Verify monotonic trends:
            # 1. As g increases, t should decrease
            if t_perturbed < self.results['t']:
                self.log(f"Monotonicity (t vs g) ✓: t decreases when g increases")
            else:
                self.log(f"Monotonicity (t vs g) ✗: t should decrease with g", False)
            
            # 2. As g increases, h should decrease (since v fixed: h = v²/2g)
            if h_perturbed < self.results['h']:
                self.log(f"Monotonicity (h vs g) ✓: h decreases when g increases")
            else:
                self.log(f"Monotonicity (h vs g) ✗: h should decrease with g", False)
            
            # 3. As g increases, h_1 should increase (h₁ ∝ g for fixed t₁=2s)
            if h1_perturbed > self.results['h_1']:
                self.log(f"Monotonicity (h₁ vs g) ✓: h₁ increases when g increases")
            else:
                self.log(f"Monotonicity (h₁ vs g) ✗: h₁ should increase with g", False)
            
            # 4. As g increases, h_2 should decrease (h₂ = v²/2g - 2g)
            if h2_perturbed < self.results['h_2']:
                self.log(f"Monotonicity (h₂ vs g) ✓: h₂ decreases when g increases")
            else:
                self.log(f"Monotonicity (h₂ vs g) ✗: h₂ should decrease with g", False)
                
            # Restore original g (not needed as we didn't modify instance)
            self.g = original_g
            
        except Exception as e:
            self.log(f"Monotonicity test failed with error: {str(e)}", False)
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []  # Reset logs
        
        # Run tests
        self.test_invariants()
        self.test_trace_consistency()
        self.test_monotonicity()
        
        # Determine overall success
        all_passed = all(success for success, _ in self.logs)
        
        # Compile detailed report
        report = [msg for _, msg in self.logs]
        
        return all_passed, report