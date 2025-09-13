def ab_test_means(
    A: np.ndarray,
    B: np.ndarray,
    alpha: float = 0.05,
    bootstrap: bool = False,
    boot_iters: int = 5000,
    seed: int = 0
) -> ABMeanResult:
    """
    Welch two-sample t-test for continuous outcomes.
    If bootstrap=True, also returns a (nonparametric) bootstrap CI for the mean difference.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    nA, nB = len(A), len(B)
    mA, mB = A.mean(), B.mean()
    sA2, sB2 = A.var(ddof=1), B.var(ddof=1)

    # Welch t-test (robust to unequal variances)
    t_stat, p_val = ttest_ind(A, B, equal_var=False)

    # Welch CI for mean difference
    se = np.sqrt(sA2/nA + sB2/nB)
    # Welch–Satterthwaite degrees of freedom
    df = (sA2/nA + sB2/nB)**2 / ((sA2**2)/(nA**2*(nA-1)) + (sB2**2)/(nB**2*(nB-1)))
    tcrit = t.ppf(1 - alpha/2, df)
    ci = ((mB - mA) - tcrit*se, (mB - mA) + tcrit*se)

    # Effect sizes
    sp = np.sqrt(((nA-1)*sA2 + (nB-1)*sB2) / (nA + nB - 2)) if (nA+nB-2) > 0 else 0.0
    d = (mB - mA) / sp if sp > 0 else 0.0
    J = 1 - 3/(4*(nA+nB) - 9) if (nA+nB) > 3 else 1.0
    g = J * d

    boot_ci = None
    if bootstrap:
        rng = np.random.default_rng(seed)
        diffs = []
        for _ in range(boot_iters):
            diff = rng.choice(A, nA, replace=True).mean() - rng.choice(B, nB, replace=True).mean()
            diffs.append(diff)
        lo, hi = np.quantile(diffs, [alpha/2, 1 - alpha/2])
        boot_ci = (lo, hi)

    return ABMeanResult(
        mean_A=mA, mean_B=mB, diff=(mB - mA),
        t_stat=t_stat, p_value=p_val, ci_diff=ci,
        cohen_d=d, hedges_g=g, boot_ci_diff=boot_ci
    )
