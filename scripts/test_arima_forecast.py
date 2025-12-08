#!/usr/bin/env python3
"""
Test script to diagnose ARIMA forecast issue.
"""

from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from quantfolio_engine.forecasting.models import ForecastingEngine

print("=" * 80)
print("ARIMA FORECAST DIAGNOSTIC TEST")
print("=" * 80)

# Load data
reports_dir = project_root / "reports" / "forecasting_assignment"
y_train = pd.read_csv(reports_dir / "y_train.csv", index_col=0, parse_dates=True).iloc[
    :, 0
]

print(f"\nTraining data:")
print(f"  Length: {len(y_train)}")
print(f"  Mean: {y_train.mean():.6f}")
print(f"  Std: {y_train.std():.6f}")
print(f"  Last 5 values:")
print(y_train.tail())

# Test 1: Direct ARIMA fitting and forecasting
print("\n" + "=" * 80)
print("TEST 1: Direct ARIMA (0,0,0) - Manual")
print("=" * 80)

try:
    # Fit ARIMA(0,0,0) directly
    model = ARIMA(y_train, order=(0, 0, 0))
    results = model.fit()

    print(f"\nModel fitted:")
    print(f"  Order: (0, 0, 0)")
    print(f"  AIC: {results.aic:.2f}")
    print(f"  Constant term: {results.params.get('const', 'N/A')}")
    print(f"  All params: {results.params}")

    # Generate forecast
    forecast_result = results.get_forecast(steps=5)
    forecast_mean = forecast_result.predicted_mean
    conf_int = forecast_result.conf_int(alpha=0.05)

    print(f"\nForecast (5 steps):")
    print(f"  Forecast mean:")
    print(forecast_mean)
    print(f"  Confidence intervals:")
    print(conf_int)

    # Check if constant
    if forecast_mean.nunique() == 1:
        print(
            f"\n⚠️  WARNING: Forecast is constant! All values = {forecast_mean.iloc[0]:.6f}"
        )
        print(f"   This is expected for ARIMA(0,0,0) - it's just the mean")
    else:
        print(f"\n✅ Forecast varies across steps")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 2: Using ForecastingEngine
print("\n" + "=" * 80)
print("TEST 2: Using ForecastingEngine")
print("=" * 80)

try:
    engine = ForecastingEngine(debug=False)

    # Fit ARIMA
    result = engine.fit_arima(y_train, auto_select=False, order=(0, 0, 0))
    print(f"\nModel fitted:")
    print(f"  Order: {result.get('order', 'N/A')}")
    print(
        f"  AIC: {result.get('model', {}).aic if hasattr(result.get('model'), 'aic') else 'N/A'}"
    )

    # Generate forecast
    forecast = engine.forecast("ARIMA", steps=5)

    if forecast is not None and not forecast.empty:
        print(f"\nForecast (5 steps):")
        print(forecast[["forecast", "lower", "upper"]])

        # Check if constant
        if forecast["forecast"].nunique() == 1:
            print(
                f"\n⚠️  WARNING: Forecast is constant! All values = {forecast['forecast'].iloc[0]:.6f}"
            )
        else:
            print(f"\n✅ Forecast varies across steps")
    else:
        print("❌ Forecast is None or empty")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 3: Check what order was actually selected
print("\n" + "=" * 80)
print("TEST 3: Auto-selected ARIMA order")
print("=" * 80)

try:
    engine = ForecastingEngine(debug=False)
    result = engine.fit_arima(y_train, auto_select=True)
    print(f"\nAuto-selected order: {result.get('order', 'N/A')}")

    # Now forecast
    forecast = engine.forecast("ARIMA", steps=5)
    if forecast is not None and not forecast.empty:
        print(f"\nForecast values:")
        print(forecast["forecast"].values)
        print(f"  Unique values: {forecast['forecast'].nunique()}")
        if forecast["forecast"].nunique() == 1:
            print(f"  ⚠️  Constant forecast: {forecast['forecast'].iloc[0]:.6f}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

# Test 4: Check the actual forecast method implementation
print("\n" + "=" * 80)
print("TEST 4: Inspecting forecast method")
print("=" * 80)

try:
    engine = ForecastingEngine(debug=False)
    result = engine.fit_arima(y_train, auto_select=True)

    # Get the fitted model
    fitted_model = engine.fitted_models.get("arima", {}).get("model")
    if fitted_model is not None:
        print(f"\nFitted model type: {type(fitted_model)}")
        print(
            f"Model order: {engine.fitted_models.get('arima', {}).get('order', 'N/A')}"
        )

        # Try direct forecast
        print(f"\nDirect forecast test:")
        forecast_result = fitted_model.get_forecast(steps=5)
        print(f"  Forecast type: {type(forecast_result)}")
        print(f"  Predicted mean type: {type(forecast_result.predicted_mean)}")
        print(f"  Predicted mean values:")
        print(forecast_result.predicted_mean.values)
        print(f"  Unique values: {np.unique(forecast_result.predicted_mean.values)}")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()

print("\n" + "=" * 80)
print("DIAGNOSIS COMPLETE")
print("=" * 80)
