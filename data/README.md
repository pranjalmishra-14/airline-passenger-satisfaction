# Dataset

This project uses the **Airline Passenger Satisfaction** dataset.

## Expected file

```
data/airline_passenger_satisfaction.csv
```

## Properties (verified by `outputs/reports/data_quality.md`)

| Property | Value |
|---|---|
| Records | 129,880 |
| Columns | 24 |
| Target | `Satisfaction` (`Satisfied` / `Neutral or Dissatisfied`) |
| Missing values | 393, all in `Arrival Delay` (0.30%) |
| Duplicate rows | 0 |
| Class balance | 56.55% Neutral or Dissatisfied / 43.45% Satisfied |

## Columns

**Identifier (dropped before modelling):** `ID`

**Demographic:** `Gender`, `Age`, `Customer Type`

**Travel:** `Type of Travel`, `Class`, `Flight Distance`

**Delay:** `Departure Delay`, `Arrival Delay` (minutes)

**Service ratings (0–5):** `Departure and Arrival Time Convenience`, `Ease of Online Booking`,
`Check-in Service`, `Online Boarding`, `Gate Location`, `On-board Service`, `Seat Comfort`,
`Leg Room Service`, `Cleanliness`, `Food and Drink`, `In-flight Service`,
`In-flight Wifi Service`, `In-flight Entertainment`, `Baggage Handling`

**Target:** `Satisfaction`

## Important: the 0 rating

Service ratings are documented as 1–5 but contain **0**. Analysis of the data shows 0 means
*"not applicable"* rather than *"worst"*: passengers who rated In-flight Wifi Service 0 are
**99.7% satisfied**, compared with roughly 25% for ratings of 1–3.

This project keeps ratings as plain 0–5 numeric values, consistent with the reference literature,
and documents the behaviour in the report instead of re-encoding it.

## Alternative format

Some distributions ship this dataset as separate `train.csv` / `test.csv` files (103,904 + 25,976
rows) with an unnamed index column and different column spellings. The loader in
`src/data_loader.py` infers column roles rather than hard-coding them, but the combined
single-file layout above is what this project expects. If you have the two-file version, combine
them and rename the columns to match the list above.

## Note

The dataset is not committed to this repository. Obtain it from its public source (Kaggle:
"Airline Passenger Satisfaction") and place it at the path above.
