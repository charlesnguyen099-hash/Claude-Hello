from bot import risk


def test_low_confidence_rejected():
    assert risk.plan_position(10_000, 100, 99, confidence=10, atr_pct=0.01) is None


def test_zero_stop_distance_rejected():
    assert risk.plan_position(10_000, 100, 100, confidence=90, atr_pct=0.01) is None


def test_negative_equity_rejected():
    assert risk.plan_position(0, 100, 99, confidence=90, atr_pct=0.01) is None


def test_plan_never_exceeds_absolute_caps():
    for confidence in (35, 50, 65, 80, 95):
        for atr_pct in (0.001, 0.005, 0.01, 0.02, 0.05):
            plan = risk.plan_position(10_000, 100, 99, confidence=confidence, atr_pct=atr_pct)
            if plan is None:
                continue
            assert plan.leverage <= risk.ABSOLUTE_MAX_LEVERAGE
            assert plan.equity_risk_pct <= risk.ABSOLUTE_MAX_EQUITY_RISK_PCT
            assert plan.margin_used <= 10_000 * risk.ABSOLUTE_MAX_EQUITY_RISK_PCT + 1e-6
            assert plan.qty > 0


def test_higher_confidence_gets_at_least_as_much_allocation():
    low = risk.plan_position(10_000, 100, 99, confidence=50, atr_pct=0.005)
    high = risk.plan_position(10_000, 100, 99, confidence=85, atr_pct=0.005)
    assert high.equity_risk_pct >= low.equity_risk_pct


def test_margin_cap_holds_even_when_stop_is_very_tight():
    # A very tight stop (0.3% away) triggers the liquidation-safety
    # leverage derate; margin_used must still never exceed the equity cap
    # after that derating is applied.
    plan = risk.plan_position(10_000, 100, 99.7, confidence=90, atr_pct=0.001)
    assert plan is not None
    assert plan.margin_used <= 10_000 * risk.ABSOLUTE_MAX_EQUITY_RISK_PCT + 1e-6


def test_high_volatility_derates_leverage():
    calm = risk.dynamic_leverage_cap(atr_pct=0.001, tier_leverage=20)
    wild = risk.dynamic_leverage_cap(atr_pct=0.02, tier_leverage=20)
    assert wild < calm
    assert wild <= 5
