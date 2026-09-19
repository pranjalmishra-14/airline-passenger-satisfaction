# Data Quality Report

_Generated automatically from the dataset; every figure below is computed, not assumed._

## 1. Dimensions

- Rows: **129,880**
- Columns: **24**
- Target column: **`satisfaction_v2`**

## 2. Missing values

| Column | Missing | % |
|---|---|---|
| Arrival Delay in Minutes | 393 | 0.30% |

## 3. Duplicates

- Fully duplicated rows (excluding dropped identifier columns): **0**

## 4. Columns removed before modelling

| Column | Reason for removal |
|---|---|
| `id` | unique identifier (129880 distinct values, one per row) |

## 5. Target distribution

| Class | Count | Share |
|---|---|---|
| satisfied | 71,087 | 54.73% |
| neutral or dissatisfied | 58,793 | 45.27% |

Imbalance ratio (majority:minority) = **1.21:1**. Mild imbalance -- resampling (e.g. SMOTE) is not required; stratified splitting is sufficient.

## 6. Feature types

- Numeric columns (19): `id`, `Age`, `Flight Distance`, `Seat comfort`, `Departure/Arrival time convenient`, `Food and drink`, `Gate location`, `Inflight wifi service`, `Inflight entertainment`, `Online support`, `Ease of Online booking`, `On-board service`, `Leg room service`, `Baggage handling`, `Checkin service`, `Cleanliness`, `Online boarding`, `Departure Delay in Minutes`, `Arrival Delay in Minutes`

- Categorical columns (5): `satisfaction_v2`, `Gender`, `Customer Type`, `Type of Travel`, `Class`

## 7. Numeric summary

|                                   |   count |     mean |      std |   min |     25% |     50% |     75% |    max |
|:----------------------------------|--------:|---------:|---------:|------:|--------:|--------:|--------:|-------:|
| id                                |  129880 | 64940.5  | 37493.3  |     1 | 32470.8 | 64940.5 | 97410.2 | 129880 |
| Age                               |  129880 |    39.43 |    15.12 |     7 |    27   |    40   |    51   |     85 |
| Flight Distance                   |  129880 |  1981.41 |  1027.12 |    50 |  1359   |  1925   |  2544   |   6951 |
| Seat comfort                      |  129880 |     2.84 |     1.39 |     0 |     2   |     3   |     4   |      5 |
| Departure/Arrival time convenient |  129880 |     2.99 |     1.53 |     0 |     2   |     3   |     4   |      5 |
| Food and drink                    |  129880 |     2.85 |     1.44 |     0 |     2   |     3   |     4   |      5 |
| Gate location                     |  129880 |     2.99 |     1.31 |     0 |     2   |     3   |     4   |      5 |
| Inflight wifi service             |  129880 |     3.25 |     1.32 |     0 |     2   |     3   |     4   |      5 |
| Inflight entertainment            |  129880 |     3.38 |     1.35 |     0 |     2   |     4   |     4   |      5 |
| Online support                    |  129880 |     3.52 |     1.31 |     0 |     3   |     4   |     5   |      5 |
| Ease of Online booking            |  129880 |     3.47 |     1.31 |     0 |     2   |     4   |     5   |      5 |
| On-board service                  |  129880 |     3.47 |     1.27 |     0 |     3   |     4   |     4   |      5 |
| Leg room service                  |  129880 |     3.49 |     1.29 |     0 |     2   |     4   |     5   |      5 |
| Baggage handling                  |  129880 |     3.7  |     1.16 |     1 |     3   |     4   |     5   |      5 |
| Checkin service                   |  129880 |     3.34 |     1.26 |     0 |     3   |     3   |     4   |      5 |
| Cleanliness                       |  129880 |     3.71 |     1.15 |     0 |     3   |     4   |     5   |      5 |
| Online boarding                   |  129880 |     3.35 |     1.3  |     0 |     2   |     4   |     4   |      5 |
| Departure Delay in Minutes        |  129880 |    14.71 |    38.07 |     0 |     0   |     0   |    12   |   1592 |
| Arrival Delay in Minutes          |  129487 |    15.09 |    38.47 |     0 |     0   |     0   |    13   |   1584 |

## 8. Categorical levels

- `satisfaction_v2`: satisfied (71,087), neutral or dissatisfied (58,793)
- `Gender`: Female (65,899), Male (63,981)
- `Customer Type`: Loyal Customer (106,100), disloyal Customer (23,780)
- `Type of Travel`: Business travel (89,693), Personal Travel (40,187)
- `Class`: Business (62,160), Eco (58,309), Eco Plus (9,411)

## 9. Service-rating scale note

Service ratings are documented as 1-5 but contain **0** values. Measured satisfaction rate by rating shows 0 does *not* behave like 'worst' -- it behaves like *not applicable*:

| Feature | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Seat comfort | 99.8% | 45.1% | 35.8% | 35.6% | 65.2% | 99.2% |
| Departure/Arrival time convenient | 54.2% | 58.6% | 54.0% | 53.9% | 52.5% | 55.7% |
| Food and drink | 77.9% | 50.8% | 43.2% | 42.8% | 59.0% | 78.0% |
| Gate location | 100.0% | 61.1% | 58.0% | 46.3% | 49.8% | 65.6% |
| Inflight wifi service | 44.7% | 26.8% | 50.2% | 51.0% | 63.8% | 66.9% |

**Decision:** ratings are kept as plain 0-5 numeric values (consistent with the reference literature). This observation is documented and discussed in the report rather than encoded as a separate indicator.
