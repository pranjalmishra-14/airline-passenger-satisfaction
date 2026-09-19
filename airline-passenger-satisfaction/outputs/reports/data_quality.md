# Data Quality Report

_Generated automatically from the dataset; every figure below is computed, not assumed._

## 1. Dimensions

- Rows: **129,880**
- Columns: **24**
- Target column: **`Satisfaction`**

## 2. Missing values

| Column | Missing | % |
|---|---|---|
| Arrival Delay | 393 | 0.30% |

## 3. Duplicates

- Fully duplicated rows (excluding dropped identifier columns): **0**

## 4. Columns removed before modelling

| Column | Reason for removal |
|---|---|
| `ID` | unique identifier (129880 distinct values, one per row) |

## 5. Target distribution

| Class | Count | Share |
|---|---|---|
| Neutral or Dissatisfied | 73,452 | 56.55% |
| Satisfied | 56,428 | 43.45% |

Imbalance ratio (majority:minority) = **1.30:1**. Mild imbalance -- resampling (e.g. SMOTE) is not required; stratified splitting is sufficient.

## 6. Feature types

- Numeric columns (19): `ID`, `Age`, `Flight Distance`, `Departure Delay`, `Arrival Delay`, `Departure and Arrival Time Convenience`, `Ease of Online Booking`, `Check-in Service`, `Online Boarding`, `Gate Location`, `On-board Service`, `Seat Comfort`, `Leg Room Service`, `Cleanliness`, `Food and Drink`, `In-flight Service`, `In-flight Wifi Service`, `In-flight Entertainment`, `Baggage Handling`

- Categorical columns (5): `Gender`, `Customer Type`, `Type of Travel`, `Class`, `Satisfaction`

## 7. Numeric summary

|                                        |   count |     mean |      std |   min |     25% |     50% |     75% |    max |
|:---------------------------------------|--------:|---------:|---------:|------:|--------:|--------:|--------:|-------:|
| ID                                     |  129880 | 64940.5  | 37493.3  |     1 | 32470.8 | 64940.5 | 97410.2 | 129880 |
| Age                                    |  129880 |    39.43 |    15.12 |     7 |    27   |    40   |    51   |     85 |
| Flight Distance                        |  129880 |  1190.32 |   997.45 |    31 |   414   |   844   |  1744   |   4983 |
| Departure Delay                        |  129880 |    14.71 |    38.07 |     0 |     0   |     0   |    12   |   1592 |
| Arrival Delay                          |  129487 |    15.09 |    38.47 |     0 |     0   |     0   |    13   |   1584 |
| Departure and Arrival Time Convenience |  129880 |     3.06 |     1.53 |     0 |     2   |     3   |     4   |      5 |
| Ease of Online Booking                 |  129880 |     2.76 |     1.4  |     0 |     2   |     3   |     4   |      5 |
| Check-in Service                       |  129880 |     3.31 |     1.27 |     0 |     3   |     3   |     4   |      5 |
| Online Boarding                        |  129880 |     3.25 |     1.35 |     0 |     2   |     3   |     4   |      5 |
| Gate Location                          |  129880 |     2.98 |     1.28 |     0 |     2   |     3   |     4   |      5 |
| On-board Service                       |  129880 |     3.38 |     1.29 |     0 |     2   |     4   |     4   |      5 |
| Seat Comfort                           |  129880 |     3.44 |     1.32 |     0 |     2   |     4   |     5   |      5 |
| Leg Room Service                       |  129880 |     3.35 |     1.32 |     0 |     2   |     4   |     4   |      5 |
| Cleanliness                            |  129880 |     3.29 |     1.31 |     0 |     2   |     3   |     4   |      5 |
| Food and Drink                         |  129880 |     3.2  |     1.33 |     0 |     2   |     3   |     4   |      5 |
| In-flight Service                      |  129880 |     3.64 |     1.18 |     0 |     3   |     4   |     5   |      5 |
| In-flight Wifi Service                 |  129880 |     2.73 |     1.33 |     0 |     2   |     3   |     4   |      5 |
| In-flight Entertainment                |  129880 |     3.36 |     1.33 |     0 |     2   |     4   |     4   |      5 |
| Baggage Handling                       |  129880 |     3.63 |     1.18 |     1 |     3   |     4   |     5   |      5 |

## 8. Categorical levels

- `Gender`: Female (65,899), Male (63,981)
- `Customer Type`: Returning (106,100), First-time (23,780)
- `Type of Travel`: Business (89,693), Personal (40,187)
- `Class`: Business (62,160), Economy (58,309), Economy Plus (9,411)
- `Satisfaction`: Neutral or Dissatisfied (73,452), Satisfied (56,428)

## 9. Service-rating scale note

Service ratings are documented as 1-5 but contain **0** values. Measured satisfaction rate by rating shows 0 does *not* behave like 'worst' -- it behaves like *not applicable*:

| Feature | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Departure and Arrival Time Convenience | 48.1% | 49.0% | 44.5% | 44.0% | 38.9% | 42.5% |
| Ease of Online Booking | 66.6% | 37.9% | 30.3% | 31.0% | 53.2% | 73.5% |
| Check-in Service | 0.0% | 24.0% | 25.1% | 45.1% | 46.0% | 61.2% |
| Online Boarding | 56.5% | 13.8% | 11.5% | 13.8% | 62.3% | 87.1% |
| Gate Location | 100.0% | 49.9% | 46.4% | 34.7% | 39.1% | 56.8% |

**Decision:** ratings are kept as plain 0-5 numeric values (consistent with the reference literature). This observation is documented and discussed in the report rather than encoded as a separate indicator.
