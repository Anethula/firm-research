"""Explicit boundaries for research methods that are not implemented yet."""

FIRM_BLOCKERS = (
    "BiasCircuitTracer assigns importance from layer position, not causal patching; "
    "CausalPinpointTuner saves an initialized adapter without optimizer steps; "
    "the joint variant does not reload a trained model and apply aligned steering. "
    "See RESEARCH_AUDIT.md before running paper experiments."
)


def require_firm_implementation():
    """Never produce a nominal FIRM result from the legacy placeholder path."""
    raise NotImplementedError("FIRM research execution is incomplete: " + FIRM_BLOCKERS)
