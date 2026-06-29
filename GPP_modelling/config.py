"""Shared configuration for the GPP modelling workflow."""

CUBE_IDS = ["003", "004", "005"]

# False -> use mean features only:
#          7 features without radiation, 8 with radiation
# True  -> use mean + std features:
#          14 features without radiation, 16 with radiation
INCLUDE_STD_FEATURES = True

FEATURE_SET_TAG = "meanstd" if INCLUDE_STD_FEATURES else "mean"
