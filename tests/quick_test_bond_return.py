"""Quick test script for bond_return function"""

def bond_return(rate_prev: float, rate_curr: float, maturity: float) -> float:
    """Per-period return on a zero-coupon bond with given maturity."""
    if maturity == 0.0:
        return 0.0
    if rate_curr == 0.0:
        return rate_prev * maturity
    return (
        rate_prev * (1 - (1 + rate_curr) ** -maturity) / rate_curr
        + (1 + rate_curr) ** -maturity
    ) - 1


def test_bond_return():
    """Run basic tests on bond_return function
    
    IMPORTANT: This function returns CAPITAL GAINS/LOSSES only, NOT total return.
    Total return would be: risk_free_rate + bond_return(...)
    """
    print("Testing bond_return function...")
    print("Note: This function returns CAPITAL GAINS only (price changes due to rate changes)\n")
    
    # Test 1: Zero maturity
    result = bond_return(0.05, 0.06, 0.0)
    assert result == 0.0, f"Test 1 failed: expected 0.0, got {result}"
    print(f"✓ Test 1 passed: Zero maturity (money market) returns 0.0")
    
    # Test 2: Unchanged rates (no capital gain)
    result = bond_return(0.05, 0.05, 10.0)
    expected = 0.0
    assert abs(result - expected) < 1e-9, f"Test 2 failed: expected {expected}, got {result}"
    print(f"✓ Test 2 passed: Unchanged rates (5% → 5%) returns {result:.4f} (no capital gain)")
    
    # Test 3: Rising rates (negative capital gain/loss)
    result = bond_return(0.05, 0.06, 10.0)
    assert result < 0, f"Test 3 failed: rising rates should produce capital loss, got {result}"
    print(f"✓ Test 3 passed: Rising rates (5% → 6%) returns {result:.4f} (capital loss)")
    
    # Test 4: Falling rates (positive capital gain)
    result = bond_return(0.05, 0.04, 10.0)
    assert result > 0, f"Test 4 failed: falling rates should produce capital gain, got {result}"
    print(f"✓ Test 4 passed: Falling rates (5% → 4%) returns {result:.4f} (capital gain)")
    
    # Test 5: Rate curr = 0 edge case
    result = bond_return(0.04, 0.0, 5.0)
    expected = 0.04 * 5.0
    assert abs(result - expected) < 1e-9, f"Test 5 failed: expected {expected}, got {result}"
    print(f"✓ Test 5 passed: rate_curr=0 edge case returns {result:.4f}")
    
    # Test 6: Duration effect (longer maturity = larger price change)
    return_short = bond_return(0.05, 0.06, 5.0)
    return_long = bond_return(0.05, 0.06, 20.0)
    assert return_long < return_short, f"Test 6 failed: longer maturity should have larger loss"
    print(f"✓ Test 6 passed: Duration effect (5yr: {return_short:.4f}, 20yr: {return_long:.4f})")
    
    # Test 7: Verify bond pricing formula
    rate_prev = 0.04
    rate_curr = 0.05
    maturity = 5.0
    
    # Manual calculation
    coupon_pv = rate_prev * (1 - (1 + rate_curr)**(-maturity)) / rate_curr
    principal_pv = (1 + rate_curr)**(-maturity)
    expected = coupon_pv + principal_pv - 1.0
    
    result = bond_return(rate_prev, rate_curr, maturity)
    assert abs(result - expected) < 1e-9, f"Test 7 failed: expected {expected}, got {result}"
    print(f"✓ Test 7 passed: Formula verification returns {result:.4f}")
    
    # Test 8: Large rate increase
    result = bond_return(0.02, 0.10, 10.0)
    assert result < -0.3, f"Test 8 failed: large rate increase should produce large capital loss, got {result}"
    print(f"✓ Test 8 passed: Large rate increase (2% → 10%) returns {result:.4f}")
    
    # Test 9: Large rate decrease
    result = bond_return(0.10, 0.02, 10.0)
    assert result > 0.5, f"Test 9 failed: large rate decrease should produce large capital gain, got {result}"
    print(f"✓ Test 9 passed: Large rate decrease (10% → 2%) returns {result:.4f}")
    
    # Test 10: Bond convexity
    base_rate = 0.05
    rate_change = 0.01
    maturity = 10.0
    
    return_up = bond_return(base_rate, base_rate + rate_change, maturity)
    return_down = bond_return(base_rate, base_rate - rate_change, maturity)
    
    # With convexity, gains from rate decrease should exceed losses from rate increase
    gain = return_down  # Capital gain when rates fall
    loss = -return_up   # Capital loss when rates rise (make positive for comparison)
    assert gain > loss, f"Test 10 failed: convexity - gain should exceed loss"
    print(f"✓ Test 10 passed: Convexity (gain: {gain:.4f} > loss: {loss:.4f})")
    
    print(f"\n{'='*50}")
    print(f"All 10 tests passed! ✓")
    print(f"{'='*50}\n")
    
    # Addit\nInterpretation example:")
    rate_prev = 0.06
    rate_curr = 0.05
    maturity = 10.0
    
    capital_gain = bond_return(rate_prev, rate_curr, maturity)
    bond_price = rate_prev * (1 - (1+rate_curr)**(-maturity)) / rate_curr + (1+rate_curr)**(-maturity)
    
    print(f"  Scenario: Bond with 6% coupon rate, rates fall from 6% to 5%")
    print(f"  Initial bond price: 1.0000 (par)")
    print(f"  New bond price: {bond_price:.4f}")
    print(f"  Capital gain: {capital_gain:.4f} ({capital_gain:.2%})")
    print(f"\n  For TOTAL return, add the coupon income:")
    print(f"  Total return = coupon + capital gain")
    print(f"             = {rate_prev:.4f} + {capital_gain:.4f}")
    print(f"             = {rate_prev + capital_gain:.4f} ({(rate_prev + capital_gain):.2%})")


if __name__ == '__main__':
    test_bond_return()
