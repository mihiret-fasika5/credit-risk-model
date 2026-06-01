## Credit Scoring Business Understanding
## 1. Basel II and the Need for Interpretable Models

The Basel II Accord introduced internal ratings-based (IRB) approaches, allowing banks to use their own models to estimate credit risk and determine regulatory capital requirements.

Because model outputs directly affect capital adequacy and financial stability, regulators require that models are:

Interpretable (clear relationship between inputs and risk output)
Well-documented (transparent methodology, assumptions, and validation)
Stable and auditable (consistent performance over time)
Why interpretability matters

A credit risk model must explain:

why a borrower is assigned a specific probability of default (PD)
how each feature contributes to the final score
whether the model is robust across time and economic conditions

Black-box models are difficult to validate, increasing regulatory and operational risk. As a result, simpler models (e.g., logistic regression with WoE) are often preferred in regulated environments.

## 2. Why Proxy Variables Are Needed?

In practice, true default labels are often unavailable or delayed. Instead, banks use proxy variables, such as:

30+ days past due (DPD)
charge-off status
delinquency or restructuring events

These proxies are used because:

default events occur infrequently
observation windows are long
“true default” definitions vary across institutions
Risks of proxy-based modeling

Using proxies introduces several business risks:

1. Label misalignment
Proxies may not fully represent economic default, leading to incorrect risk estimation.

2. Selection bias
Only approved borrowers generate repayment data, causing biased training samples.

3. Policy feedback loops
Credit policy changes affect observed defaults, which in turn affect future models.

4. Temporal instability
Proxy behavior may change across economic cycles, reducing model robustness.

## 3. Trade-offs: Interpretable Models vs High-Performance Models

In credit risk modeling, there is a fundamental trade-off between interpretability and predictive performance.

Logistic Regression combined with Weight of Evidence (WoE) transformations is widely used because it produces transparent, monotonic relationships between features and default risk. Each variable’s contribution can be clearly interpreted, and the model is stable, easy to validate, and aligned with regulatory expectations. However, this simplicity limits its ability to capture complex nonlinear relationships and feature interactions, which can reduce predictive power.

In contrast, Gradient Boosting models (such as XGBoost or LightGBM) provide significantly higher predictive accuracy by capturing nonlinear patterns and interactions automatically. They often improve risk ranking and discrimination ability. However, they are more difficult to interpret, require post-hoc explanation methods such as SHAP, and may face challenges in regulatory approval due to reduced transparency and stability concerns.

In regulated financial environments, this creates a clear tension: improving predictive performance often comes at the cost of explainability, governance simplicity, and regulatory acceptance.

Summary

The Basel II framework strongly influences credit risk modeling by prioritizing transparency, interpretability, and model governance. Because true default labels are often unavailable, proxy variables are necessary but introduce bias and instability. As a result, financial institutions must balance the need for predictive accuracy with strict regulatory expectations, often favoring interpretable models despite the availability of more powerful machine learning techniques.