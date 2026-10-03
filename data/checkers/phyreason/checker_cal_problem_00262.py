import math
import traceback

class PhysicsChecker:
    def __init__(self, solution_instance):
        """Initialize with safe attribute access."""
        # Extract input parameters
        self.m = 2.0  # Given mass in kg
        self.g = 10.0  # Given gravity in m/s^2
        
        # Safely extract results with fallbacks
        self.f = self._safe_getattr(solution_instance, 'f')
        self.x = self._safe_getattr(solution_instance, 'x')
        self.v2 = self._safe_getattr(solution_instance, 'v₂')
        self.a1 = self._safe_getattr(solution_instance, 'a₁')
        self.h = self._safe_getattr(solution_instance, 'h')
        self.a2 = self._safe_getattr(solution_instance, 'a₂')
        self.overlinev = self._safe_getattr(solution_instance, 'overlinev')
        
        # Store for perturbation tests
        self._original_f = self.f
        self._original_results = {
            'f': self.f, 'x': self.x, 'v2': self.v2, 
            'a1': self.a1, 'h': self.h, 'a2': self.a2,
            'overlinev': self.overlinev
        }
        
        self.logs = []
    
    def _safe_getattr(self, obj, attr_name):
        """Safe extraction with multiple fallbacks."""
        try:
            # Try direct attribute
            value = getattr(obj, attr_name, None)
            if value is None:
                # Try results dictionary
                value = getattr(obj, 'results', {}).get(attr_name, None)
            if value is None:
                # Try alternative names
                alt_names = {
                    'v₂': ['v2', 'v_2', 'vf', 'v_final'],
                    'a₁': ['a1', 'a_up', 'a_upward'],
                    'a₂': ['a2', 'a_down', 'a_downward'],
                    'overlinev': ['v_avg', 'v_average', 'average_velocity']
                }
                for alt in alt_names.get(attr_name, []):
                    value = getattr(obj, alt, None)
                    if value is not None:
                        break
                    value = getattr(obj, 'results', {}).get(alt, None)
                    if value is not None:
                        break
            
            # Convert to float if possible
            if isinstance(value, str):
                # Handle special strings like "4√6"
                if '√' in value:
                    parts = value.split('√')
                    if len(parts) == 2:
                        value = float(parts[0]) * math.sqrt(float(parts[1]))
                    else:
                        value = float(value.replace('√', ''))
                else:
                    value = float(value)
            return float(value) if value is not None else 0.0
        except:
            return 0.0
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison."""
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics laws."""
        logs = []
        try:
            # 1. Newton's 2nd Law consistency
            # Upward: f + mg = m*a1 (air resistance adds to gravity)
            net_upward = self.f + self.m * self.g
            expected_a1_up = net_upward / self.m
            if self._is_close(self.a1, expected_a1_up):
                logs.append("✓ Newton's 2nd Law (upward): f + mg = m*a1 consistent")
            else:
                logs.append(f"✗ Newton's 2nd Law (upward): f + mg = {net_upward}, m*a1 = {self.m * self.a1}")
            
            # Downward: mg - f = m*a2 (air resistance opposes gravity)
            net_downward = self.m * self.g - self.f
            expected_a2_down = net_downward / self.m
            if self._is_close(self.a2, expected_a2_down):
                logs.append("✓ Newton's 2nd Law (downward): mg - f = m*a2 consistent")
            else:
                logs.append(f"✗ Newton's 2nd Law (downward): mg - f = {net_downward}, m*a2 = {self.m * self.a2}")
            
            # 2. Energy-work theorem (with air resistance)
            # Initial kinetic energy = Work against gravity + Work against air resistance + Final kinetic energy
            # From kinematics: v0^2 = 2*a1*h, v2^2 = 2*a2*h
            v0_sq = 2 * self.a1 * self.h
            ke_initial = 0.5 * self.m * v0_sq
            ke_final = 0.5 * self.m * self.v2**2
            work_gravity = 0  # Round trip, gravity work is zero
            work_air = -self.f * 2 * self.h  # Air resistance acts over total distance 2h
            
            energy_balance = ke_final - ke_initial - work_gravity - work_air
            if self._is_close(energy_balance, 0, abs_tol=0.1):
                logs.append("✓ Energy-work theorem consistent (accounts for air resistance)")
            else:
                logs.append(f"✗ Energy-work theorem: ΔKE = {ke_final - ke_initial}, Work_air = {work_air}")
            
            # 3. Symbolic verification using numeric comparisons only
            # From upward motion: f = m*a1 - m*g
            f_expected = self.m * self.a1 - self.m * self.g
            # From downward motion: a2 = (m*g - f)/m
            a2_expected = (self.m * self.g - self.f) / self.m
            
            if self._is_close(self.f, f_expected) and self._is_close(self.a2, a2_expected):
                logs.append("✓ Symbolic Newton's laws verified")
            else:
                logs.append("✗ Symbolic verification failed")
                
        except Exception as e:
            logs.append(f"✗ Error in test_invariants: {str(e)}")
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("✗")]) == 0
    
    def test_trace_consistency(self):
        """Check intermediate results against final answers."""
        logs = []
        try:
            # 1. Check a1 from v-t graph assumption (12 m/s² from steps)
            expected_a1 = 12.0
            if self._is_close(self.a1, expected_a1):
                logs.append("✓ Upward acceleration matches v-t graph analysis")
            else:
                logs.append(f"✗ Upward acceleration: got {self.a1}, expected ~{expected_a1}")
            
            # 2. Check f from a1 and Newton's 2nd law
            expected_f = self.m * (self.a1 - self.g)
            if self._is_close(self.f, expected_f):
                logs.append("✓ Air resistance consistent with upward motion")
            else:
                logs.append(f"✗ Air resistance: got {self.f}, expected ~{expected_f}")
            
            # 3. Check h from kinematics
            # From v-t graph: h = (v0/2)*t1, and v0 = a1*t1 => h = a1*t1^2/2
            # But we can check v0^2 = 2*a1*h
            v0_sq = 2 * self.a1 * self.h
            if v0_sq > 0:
                logs.append(f"✓ Maximum height kinematics: v0² = {v0_sq:.2f}, 2*a1*h = {2*self.a1*self.h:.2f}")
            else:
                logs.append(f"✗ Maximum height calculation inconsistent")
            
            # 4. Check v2 from downward motion
            expected_v2_sq = 2 * self.a2 * self.h
            actual_v2_sq = self.v2**2
            if self._is_close(actual_v2_sq, expected_v2_sq, rel_tol=1e-2):
                logs.append(f"✓ Final velocity: v2² = {actual_v2_sq:.2f}, 2*a2*h = {expected_v2_sq:.2f}")
            else:
                logs.append(f"✗ Final velocity: v2² = {actual_v2_sq:.2f}, expected ~{expected_v2_sq:.2f}")
            
            # 5. Check displacement and average velocity
            if self._is_close(self.x, 0):
                logs.append("✓ Displacement is zero (returns to start)")
            else:
                logs.append(f"✗ Displacement should be 0, got {self.x}")
            
            if self._is_close(self.overlinev, 0):
                logs.append("✓ Average velocity is zero (zero displacement)")
            else:
                logs.append(f"✗ Average velocity should be 0, got {self.overlinev}")
            
            # Cross-check with target answers
            target_f = 4.0
            target_v2 = 4 * math.sqrt(6)
            
            if self._is_close(self.f, target_f):
                logs.append(f"✓ Air resistance matches target: {target_f} N")
            else:
                logs.append(f"✗ Air resistance: got {self.f}, target {target_f} N")
            
            if self._is_close(self.v2, target_v2, rel_tol=1e-2):
                logs.append(f"✓ Final velocity matches target: {target_v2:.3f} m/s")
            else:
                logs.append(f"✗ Final velocity: got {self.v2:.3f}, target {target_v2:.3f} m/s")
            
            if self._is_close(self.overlinev, 0):
                logs.append("✓ Average velocity matches target: 0 m/s")
            else:
                logs.append(f"✗ Average velocity: got {self.overlinev}, target 0 m/s")
                
        except Exception as e:
            logs.append(f"✗ Error in test_trace_consistency: {str(e)}")
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("✗")]) == 0
    
    def test_monotonicity(self):
        """Perturb mass and check expected trends."""
        logs = []
        try:
            # Perturb mass by +10%
            m_perturbed = self.m * 1.1
            
            # Recalculate using formulas from steps analysis
            # 1. Upward acceleration should remain same (from v-t graph)
            a1_expected = self.a1  # Same graph, same a1
            
            # 2. Air resistance formula: f = m*a1 - m*g
            f_perturbed = m_perturbed * a1_expected - m_perturbed * self.g
            
            # 3. Maximum height from same v-t graph (h unchanged for same initial velocity?)
            # Actually, h = v0²/(2*a1), and v0 determined by graph, so h should remain same
            h_expected = self.h
            
            # 4. Downward acceleration: a2 = (m*g - f)/m
            a2_perturbed = (m_perturbed * self.g - f_perturbed) / m_perturbed
            
            # 5. Final velocity: v2² = 2*a2*h
            v2_perturbed = math.sqrt(2 * a2_perturbed * h_expected)
            
            # Check expected trends:
            # 1. f increases with mass (f ∝ m for fixed a1 and g)
            if f_perturbed > self.f:
                logs.append("✓ Monotonicity: f increases with mass (as expected)")
            else:
                logs.append(f"✗ Monotonicity: f decreased from {self.f} to {f_perturbed} with mass increase")
            
            # 2. a2 decreases with mass? Let's analyze:
            # a2 = g - f/m = g - (a1 - g) = 2g - a1 (independent of m!)
            # Actually: f = m*(a1 - g) => a2 = g - (a1 - g) = 2g - a1
            # So a2 should remain constant with mass change
            if self._is_close(a2_perturbed, self.a2):
                logs.append("✓ Monotonicity: a2 independent of mass (as expected from formula)")
            else:
                logs.append(f"✗ Monotonicity: a2 changed from {self.a2} to {a2_perturbed}")
            
            # 3. v2 should remain same (since a2 and h unchanged)
            if self._is_close(v2_perturbed, self.v2, rel_tol=1e-3):
                logs.append("✓ Monotonicity: v2 independent of mass (constant air resistance)")
            else:
                logs.append(f"✗ Monotonicity: v2 changed from {self.v2} to {v2_perturbed}")
            
            # Restore original values
            self.f = self._original_f
            
        except Exception as e:
            logs.append(f"✗ Error in test_monotonicity: {str(e)}")
            traceback.print_exc()
        
        self.logs.extend(logs)
        return len([l for l in logs if l.startswith("✗")]) == 0
    
    def run_all_tests(self):
        """Execute all tests and return success status with logs."""
        self.logs = []
        
        # Run tests
        success1 = self.test_invariants()
        success2 = self.test_trace_consistency()
        success3 = self.test_monotonicity()
        
        # Summary
        total_tests = 3
        passed_tests = sum([success1, success2, success3])
        
        self.logs.append(f"\n=== SUMMARY ===")
        self.logs.append(f"Passed: {passed_tests}/{total_tests}")
        
        if passed_tests == total_tests:
            self.logs.append("✓ All physics checks passed!")
            overall_success = True
        else:
            self.logs.append("✗ Some physics checks failed")
            overall_success = False
        
        return overall_success, self.logs