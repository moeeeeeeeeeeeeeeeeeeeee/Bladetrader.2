# Case 4 North Star (Working Build Context)

We are currently iterating and testing, but every task should move toward this final objective:

## Objective

Build a sentiment-driven prediction system that tests whether AI-related text signals improve post-earnings stock prediction, including:

- direct company sentiment effects
- spillover sentiment effects from related companies/events/regulation

Target prediction: whether a stock's five-trading-day return after earnings is positive or negative.

## Required Deliverables

- Baseline prediction model vs sentiment-enhanced model.
- Feature framework with company-level sentiment and at least one spillover feature.
- Dashboard/visuals for:
  - sentiment trends
  - sentiment vs return behavior
  - spillover relationships
  - model comparison results
- Short presentation of:
  - data pipeline
  - sentiment methodology
  - feature design
  - validation approach
  - main findings
- Clear data leakage controls:
  - only information in [T-7d, T] for features
  - time-aware validation
- Real-world improvement roadmap.

## Build Principles

- End-to-end pipeline first, then optimize model quality.
- Keep evidence traceable (document links, feature provenance, model run metadata).
- Prefer explainable outputs over black-box-only scores.
- Keep direct and spillover channels separate in scoring and reporting.

