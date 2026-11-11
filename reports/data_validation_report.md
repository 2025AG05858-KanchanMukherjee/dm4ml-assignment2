# Data Validation Report

Generated automatically by `src/validation/validate.py`. ERROR rows are quarantined in `data/errors/`; WARN fields are blanked and imputed during preparation. One row can violate several rules.

## Row reconciliation

| Table | Rows in | Valid | Quarantined | Rows with warnings |
|---|---:|---:|---:|---:|
| users | 1210 | 1187 | 23 | 85 |
| items | 605 | 597 | 8 | 38 |
| interactions | 49707 | 48029 | 1678 | 0 |

`Rows in = Valid + Quarantined` is asserted for every table.

## Violations by rule

| Table | Rule | Severity | Violations |
|---|---|---|---:|
| users | `missing_country` | WARN | 40 |
| users | `missing_age` | WARN | 25 |
| users | `age_out_of_range` | WARN | 15 |
| users | `duplicate_row` | ERROR | 10 |
| users | `invalid_signup_date` | ERROR | 8 |
| users | `invalid_dtype_age` | WARN | 5 |
| users | `missing_user_id` | ERROR | 5 |
| items | `missing_title` | WARN | 20 |
| items | `missing_genres` | WARN | 8 |
| items | `release_year_out_of_range` | WARN | 8 |
| items | `duplicate_key` | ERROR | 5 |
| items | `missing_item_id` | ERROR | 3 |
| items | `invalid_dtype_release_year` | WARN | 2 |
| interactions | `fk_user_not_found` | ERROR | 663 |
| interactions | `fk_item_not_found` | ERROR | 378 |
| interactions | `duplicate_row` | ERROR | 146 |
| interactions | `rating_out_of_range` | ERROR | 120 |
| interactions | `missing_rating` | ERROR | 100 |
| interactions | `timestamp_in_future` | ERROR | 80 |
| interactions | `timestamp_before_signup` | ERROR | 59 |
| interactions | `duplicate_user_item` | ERROR | 58 |
| interactions | `invalid_dtype_rating` | ERROR | 30 |
| interactions | `invalid_timestamp` | ERROR | 30 |
| interactions | `missing_user_id` | ERROR | 30 |

## Detection check against injected defects (synthetic data only)

The generator logs exactly what it corrupted (`data/raw/_injected_defects.json`). Detected can exceed injected where one defect cascades (e.g. a rejected user makes that user's interactions orphans) and can be lower where a row is caught first by another rule (each defect is reported once, by the first rule that fires).

| Defect | Injected | Detected |
|---|---:|---:|
| Missing rating | 100 | 100 |
| Missing user_id (interactions) | 30 | 30 |
| Rating out of range | 120 | 120 |
| Non-numeric rating | 30 | 30 |
| Future timestamp | 80 | 80 |
| Unparsable timestamp | 30 | 30 |
| Timestamp before signup | 60 | 59 |
| Orphan item_id | 60 | 378 |
| Orphan user_id (+ cascade from rejected users) | 80 | 663 |
| Exact duplicate interactions | 150 | 146 |
| Conflicting (user,item) duplicates | 60 | 58 |
| Users: duplicate rows | 10 | 10 |
| Users: invalid signup_date | 8 | 8 |
| Users: invalid age (dtype + range) | 20 | 20 |
| Users: missing age | 25 | 25 |
| Users: missing country | 40 | 40 |
| Items: duplicate item_id | 5 | 5 |
| Items: missing title | 20 | 20 |
| Items: missing genres | 8 | 8 |
