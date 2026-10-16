pyhealth.models.TPC
===================================

The separate callable TPCBlock and the complete TPC model.

Overview
--------
Temporal Pointwise Convolutional (TPC) networks are designed for remaining length of stay
prediction and in-hospital mortality prediction in intensive care units, adapted from:

    Emma Rocheteau, Pietro Liò, and Stephanie Hyland.
    "Temporal Pointwise Convolutional Networks for Length of Stay Prediction in the Intensive Care Unit."
    ACM Conference on Health, Inference, and Learning (CHIL), 2021.
    DOI: https://doi.org/10.1145/3450439.3451860

The model consumes multi-modal clinical inputs:
- ``vitals``: Continuous dynamic time-series features (shape: ``(B, T, F)``).
- ``vitals_decay``: Exponential decay indicators ``0.75^j`` (shape: ``(B, T, F)``).
- ``static``: Baseline flat demographics and admission features (shape: ``(B, S)``).
- ``mask``: Observation presence indicators (shape: ``(B, T)`` or ``(B, T, F)``).

Outputs:
- ``y_prob``: Predicted remaining length of stay in days with softplus floor (shape: ``(B, T, 1)``).
- ``logit``: Raw linear projection for length of stay (shape: ``(B, T, 1)``).
- ``loss``: Mean squared logarithmic error (MSLE) or MSE loss across valid stay hours.
- ``mortality_prob``: In-hospital mortality probabilities when ``multitask=True`` (shape: ``(B, T, 1)``).
- ``mortality_logit``: Raw mortality logits when ``multitask=True`` (shape: ``(B, T, 1)``).

.. autoclass:: pyhealth.models.TPCBlock
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TPC
    :members:
    :undoc-members:
    :show-inheritance:
