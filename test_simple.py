def bond_return(rate_prev: float, rate_curr: float, maturity: float) -> float:
    """Per-period return on a zero-coupon bond with given maturity.
    
    NOTE: This returns CAPITAL GAINS/LOSSES only, not total return.
    In the simulation, total return = risk_free_rate + bond_return(...)
    """
    if maturity == 0.0:
        return 0.0
    if rate_curr == 0.0:
        return rate_prev * maturity
    return (
        rate_prev * (1 - (1 + rate_curr) ** -maturity) / rate_curr
        + (1 + rate_curr) ** -maturity
    ) - 1


print('Testing bond_return function...')
print('NOTE: This function returns CAPITAL GAINS only (not total return)\n')
print(f'Zero maturity: {bond_return(0.05, 0.06, 0.0)} (money market, no capital gain)')
print(f'Unchanged rates: {bond_return(0.05, 0.05, 10.0)} (no capital gain)')
print(f'Rising rates: {bond_return(0.05, 0.06, 10.0)} (capital loss)')
print(f'Falling rates: {bond_return(0.05, 0.04, 10.0)} (capital gain)')
print(f'Rate curr = 0: {bond_return(0.04, 0.0, 5.0)} (special case)')
print('\nAll basic tests completed!')
