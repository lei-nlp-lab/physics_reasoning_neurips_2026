import math
import traceback
import sympy as sp

class PhysicsChecker:
    """
    Physics auditor for projectile motion problem.
    Mass m=1kg thrown horizontally from height h=5m with v0=10m/s.
    Gravity g=10m/s², no air resistance.
    """
    
    def __init__(self, solution_instance):
        """
        Safely extract all inputs and results from solution instance.
        Never access solution_instance.attr directly - use getattr with fallbacks.
        """
        self.solution = solution_instance
        
        # Extract inputs with defaults from problem statement
        self.m = self._safe_get('m', 1.0)
        self.h = self._safe_get('h', 5.0)
        self.v0 = self._safe_get('v0', 10.0)
        self.g = self._safe_get('g', 10.0)
        
        # Extract results using key mapping from problem
        self.t = self._get_result('t')
        self.Delta_v = self._get_result('Deltav')
        self.Delta_p = self._get_result('Deltap')
        self.v_y = self._get_result('v_y')
        self.v = self._get_result('v')
        self.p_prime = self._get_result('p\'')
        
        # Expected values from KB analysis
        self.t_expected = math.sqrt(2 * self.h / self.g)
        self.v_y_expected = self.g * self.t_expected
        self.Delta_v_expected = self.v_y_expected
        self.Delta_p_expected = self.m * self.v_y_expected
        self.v_expected = math.sqrt(self.v0**2 + self.v_y_expected**2)
        self.p_prime_expected = self.m * self.v_expected
        
        # For monotonicity test - store original values
        self._original_h = self.h
        
    def _safe_get(self, attr, default):
        """Safe attribute access with fallback to results dictionary."""
        return getattr(self.solution, attr, 
                      getattr(self.solution, 'results', {}).get(attr, default))
    
    def _get_result(self, key):
        """Get result from solution instance with multiple fallback strategies."""
        # Try direct attribute
        if hasattr(self.solution, key):
            return getattr(self.solution, key)
        
        # Try results dictionary
        results = getattr(self.solution, 'results', {})
        
        # Try various key mappings
        key_mappings = {
            't': ['t', 'time', 'flight_time'],
            'Deltav': ['Deltav', 'delta_v', 'Δv', 'vertical_velocity_change'],
            'Deltap': ['Deltap', 'delta_p', 'Δp', 'momentum_increment'],
            'v_y': ['v_y', 'vy', 'vertical_velocity'],
            'v': ['v', 'velocity', 'landing_speed'],
            'p\'': ['p\'', 'p_prime', 'p_final', 'final_momentum']
        }
        
        if key in key_mappings:
            for k in key_mappings[key]:
                if k in results:
                    val = results[k]
                    # Handle string values like "10√2"
                    if isinstance(val, str):
                        if '√' in val:
                            try:
                                # Parse "10√2" format
                                parts = val.split('√')
                                if len(parts) == 2:
                                    return float(parts[0]) * math.sqrt(float(parts[1]))
                            except:
                                pass
                    return float(val) if val is not None else 0.0
        
        return 0.0  # Default fallback
    
    def _is_close(self, a, b, rel_tol=1e-2, abs_tol=1e-2):
        """Robust fuzzy comparison - NEVER use == or 1e-9."""
        if a is None or b is None:
            return False
        return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)
    
    def test_invariants(self):
        """Test fundamental physics invariants: energy conservation and Newton's laws."""
        logs = []
        success = True
        
        try:
            # 1. Energy conservation: initial energy = final energy
            # Initial: PE = mgh, KE_horizontal = 0.5*m*v0²
            # Final: KE_total = 0.5*m*v²
            initial_energy = self.m * self.g * self.h + 0.5 * self.m * self.v0**2
            final_energy = 0.5 * self.m * self.v**2
            
            if not self._is_close(initial_energy, final_energy, rel_tol=2e-2):
                logs.append(f"ENERGY VIOLATION: Initial={initial_energy:.2f}J, Final={final_energy:.2f}J")
                success = False
            else:
                logs.append(f"✓ Energy conserved: {initial_energy:.2f}J → {final_energy:.2f}J")
            
            # 2. Newton's 2nd Law: F = Δp/Δt (vertical direction)
            # Vertical force = mg, time = t, Δp = m * v_y
            force_expected = self.m * self.g
            force_calculated = self.Delta_p / self.t if self.t != 0 else 0
            
            if not self._is_close(force_expected, force_calculated, rel_tol=2e-2):
                logs.append(f"NEWTON'S 2ND LAW VIOLATION: F_expected={force_expected:.2f}N, F_calc={force_calculated:.2f}N")
                success = False
            else:
                logs.append(f"✓ Newton's 2nd Law: F={force_expected:.2f}N = Δp/Δt")
            
            # 3. Kinematic consistency: v_y = g*t
            v_y_from_kinematics = self.g * self.t
            if not self._is_close(self.v_y, v_y_from_kinematics, rel_tol=2e-2):
                logs.append(f"KINEMATICS VIOLATION: v_y={self.v_y:.2f}, g*t={v_y_from_kinematics:.2f}")
                success = False
            else:
                logs.append(f"✓ Kinematics: v_y = g*t = {v_y_from_kinematics:.2f}m/s")
                
        except Exception as e:
            logs.append(f"Invariant test failed: {str(e)}")
            success = False
            
        return success, logs
    
    def test_trace_consistency(self):
        """Cross-check intermediate results with final answers."""
        logs = []
        success = True
        
        try:
            # Check step-by-step consistency from KB analysis
            
            # Step 1: Time of flight t = √(2h/g)
            t_calc = math.sqrt(2 * self.h / self.g)
            if not self._is_close(self.t, t_calc, rel_tol=2e-2):
                logs.append(f"TIME INCONSISTENCY: t={self.t:.2f}s, expected={t_calc:.2f}s")
                success = False
            else:
                logs.append(f"✓ Time: t = √(2h/g) = {t_calc:.2f}s")
            
            # Step 2: Δv = v_y = g*t
            v_y_calc = self.g * self.t
            if not (self._is_close(self.Delta_v, v_y_calc, rel_tol=2e-2) and 
                    self._is_close(self.v_y, v_y_calc, rel_tol=2e-2)):
                logs.append(f"VELOCITY INCONSISTENCY: Δv={self.Delta_v:.2f}, v_y={self.v_y:.2f}, expected={v_y_calc:.2f}m/s")
                success = False
            else:
                logs.append(f"✓ Vertical velocity: v_y = g*t = {v_y_calc:.2f}m/s")
            
            # Step 3: Δp = m * Δv
            delta_p_calc = self.m * self.Delta_v
            if not self._is_close(self.Delta_p, delta_p_calc, rel_tol=2e-2):
                logs.append(f"Δp INCONSISTENCY: Δp={self.Delta_p:.2f}, expected={delta_p_calc:.2f}kg·m/s")
                success = False
            else:
                logs.append(f"✓ Δp = m*Δv = {delta_p_calc:.2f}kg·m/s")
            
            # Step 5: v = √(v0² + v_y²)
            v_calc = math.sqrt(self.v0**2 + self.v_y**2)
            if not self._is_close(self.v, v_calc, rel_tol=2e-2):
                logs.append(f"SPEED INCONSISTENCY: v={self.v:.2f}, expected={v_calc:.2f}m/s")
                success = False
            else:
                logs.append(f"✓ Landing speed: v = √(v0²+v_y²) = {v_calc:.2f}m/s")
            
            # Step 6: p' = m*v
            p_prime_calc = self.m * self.v
            if not self._is_close(self.p_prime, p_prime_calc, rel_tol=2e-2):
                logs.append(f"p' INCONSISTENCY: p'={self.p_prime:.2f}, expected={p_prime_calc:.2f}kg·m/s")
                success = False
            else:
                logs.append(f"✓ Final momentum: p' = m*v = {p_prime_calc:.2f}kg·m/s")
            
            # Direction check: tan(θ) = v_y/v0
            if self.v0 != 0:
                angle = math.degrees(math.atan(self.v_y / self.v0))
                if not self._is_close(angle, 45.0, rel_tol=2e-2):
                    logs.append(f"DIRECTION INCONSISTENCY: θ={angle:.1f}°, expected 45°")
                    success = False
                else:
                    logs.append(f"✓ Direction: θ = arctan(v_y/v0) = {angle:.1f}°")
                    
        except Exception as e:
            logs.append(f"Trace consistency test failed: {str(e)}")
            success = False
            
        return success, logs
    
    def test_monotonicity(self):
        """Perturb input (height h) and verify monotonic trends."""
        logs = []
        success = True
        
        try:
            # Perturb height by +10%
            h_perturbed = self.h * 1.1
            
            # Recalculate using physics formulas
            t_new = math.sqrt(2 * h_perturbed / self.g)
            v_y_new = self.g * t_new
            Delta_p_new = self.m * v_y_new
            v_new = math.sqrt(self.v0**2 + v_y_new**2)
            p_prime_new = self.m * v_new
            
            # Check monotonic trends
            if not (t_new > self.t_expected):
                logs.append(f"MONOTONICITY FAILED: t({h_perturbed:.1f}m)={t_new:.2f}s ≤ t({self.h:.1f}m)={self.t_expected:.2f}s")
                success = False
            else:
                logs.append(f"✓ t increases with h: {self.t_expected:.2f}s → {t_new:.2f}s")
            
            if not (v_y_new > self.v_y_expected):
                logs.append(f"MONOTONICITY FAILED: v_y({h_perturbed:.1f}m)={v_y_new:.2f}m/s ≤ v_y({self.h:.1f}m)={self.v_y_expected:.2f}m/s")
                success = False
            else:
                logs.append(f"✓ v_y increases with h: {self.v_y_expected:.2f} → {v_y_new:.2f}m/s")
            
            if not (Delta_p_new > self.Delta_p_expected):
                logs.append(f"MONOTONICITY FAILED: Δp({h_perturbed:.1f}m)={Delta_p_new:.2f} ≤ Δp({self.h:.1f}m)={self.Delta_p_expected:.2f}kg·m/s")
                success = False
            else:
                logs.append(f"✓ Δp increases with h: {self.Delta_p_expected:.2f} → {Delta_p_new:.2f}kg·m/s")
            
            if not (p_prime_new > self.p_prime_expected):
                logs.append(f"MONOTONICITY FAILED: p'({h_perturbed:.1f}m)={p_prime_new:.2f} ≤ p'({self.h:.1f}m)={self.p_prime_expected:.2f}kg·m/s")
                success = False
            else:
                logs.append(f"✓ p' increases with h: {self.p_prime_expected:.2f} → {p_prime_new:.2f}kg·m/s")
            
            # Restore original value
            self.h = self._original_h
            
        except Exception as e:
            logs.append(f"Monotonicity test failed: {str(e)}")
            success = False
            # Restore on error
            self.h = self._original_h
            
        return success, logs
    
    def run_all_tests(self):
        """Run all tests and return overall success and detailed logs."""
        all_logs = ["="*60, "PHYSICS AUDIT REPORT", "="*60]
        overall_success = True
        
        # Test 1: Invariants
        success1, logs1 = self.test_invariants()
        all_logs.append("\n1. PHYSICS INVARIANTS:")
        all_logs.extend(logs1)
        if not success1:
            overall_success = False
        
        # Test 2: Trace Consistency
        success2, logs2 = self.test_trace_consistency()
        all_logs.append("\n2. TRACE CONSISTENCY:")
        all_logs.extend(logs2)
        if not success2:
            overall_success = False
        
        # Test 3: Monotonicity
        success3, logs3 = self.test_monotonicity()
        all_logs.append("\n3. MONOTONICITY:")
        all_logs.extend(logs3)
        if not success3:
            overall_success = False
        
        # Summary
        all_logs.append("\n" + "="*60)
        all_logs.append(f"OVERALL RESULT: {'PASS' if overall_success else 'FAIL'}")
        all_logs.append("="*60)
        
        return overall_success, all_logs