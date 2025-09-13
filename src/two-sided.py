def ab_test_proportions(xA: int, nA: int, xB: int, nB: int,
                        alpha: float = 0.05,
                        alternative: Alt = "two-sided") -> ABProportionResult:
    assert 0 <= xA <= nA and 0 <= xB <= nB
    pA, pB = xA/nA, xB/nB

    # z-test (pooled SE under H0)
    p_pool = (xA + xB) / (nA + nB)
    se0 = math.sqrt(p_pool * (1 - p_pool) * (1/nA + 1/nB))
    z = (pB - pA) / se0 if se0 > 0 else 0.0
    pval = _p_from_z(z, alternative)

    # 95% CI for difference using *unpooled* SE (Wald)
    se = math.sqrt(pA*(1-pA)/nA + pB*(1-pB)/nB)
    zc = norm.ppf(1 - alpha/2)
    ci = ((pB - pA) - zc*se, (pB - pA) + zc*se)

    # Cohen's h (standardized effect for proportions)
    phi = lambda p: 2*math.asin(math.sqrt(p))
    h = phi(pB) - phi(pA)

    lift_abs = pB - pA
    lift_rel = (pB/pA - 1) if pA > 0 else float("inf")
    return ABProportionResult(pA, pB, lift_abs, lift_rel, z, pval, ci, h)
