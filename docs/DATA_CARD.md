# Synthetic data card

**Dataset:** FraudScope scenario simulation v1. **Origin:** generated locally from code and random seeds, without real customer data or fitted IEEE distributions. **Purpose:** demonstrate fraud engineering and evaluation behavior. **Excluded use:** making claims about a bank, country, demographic group, or real fraud prevalence.

The shipped experiment includes 60,000 transactions for each of seeds 42, 73 and 101 across 120 simulated days. A hidden attack intent drives overlapping transaction patterns. The fraud outcome is sampled stochastically, so suspicious events are not always fraud and some legitimate-looking events are fraud. Attack bursts increase the prevalence above the initial independent intent probability. The final quarter increases app usage and shifts attack types toward social engineering. This creates an intentional transfer challenge.

| Field group | Fields | Availability |
|---|---|---|
| Event identity | `transaction_id`, `event_time` | At event time |
| Entity linkage | `account_id`, `device_id` | Assumed available at event time |
| Transaction | `amount`, `channel`, `product`, `device_type`, `cross_border` | At event time |
| Context | `account_age_days`, `email_age_days`, `geo_distance_km`, `failed_logins_24h` | Assumed event-time snapshot |
| Outcome | `is_fraud` | After label maturity |
| Label timing | `label_available_time` | Simulated event time plus two days |
| Simulator audit | `scenario_audit_only` | Never a model predictor |

## Feature dictionary

| Feature | Definition |
|---|---|
| `log_amount` | `log(1 + amount)` |
| `hour_sin`, `hour_cos` | Cyclic encoding of relative time within a day |
| `account_txn_1h`, `account_txn_24h`, `account_txn_7d` | Prior account event counts in open-left, open-right rolling windows |
| `amount_ratio_7d` | Current amount divided by the prior seven-day mean, capped at 1000. Cold-start value is 1 |
| `device_accounts_24h` | Distinct accounts previously seen on this device within 24 hours |
| `new_device_7d` | Device not observed on this account during the prior seven days |
| Remaining contract fields | Numeric context and three categorical transaction descriptors |

No ID, outcome, availability time, hidden intent or audit scenario is passed to the estimator. Entity IDs are used only to construct prior activity.

## Known limitations

- The simulator is designed, so its signal structure is simpler than evolving real fraud. High or low synthetic scores do not validate a real model.
- Fraud prevalence, amounts, recovery and costs are invented assumptions. The generator is IEEE-inspired only in task type and mixed feature structure. It is not a statistical replica.
- All synthetic labels become known after two days. Real chargebacks can be delayed, censored, disputed or selected by interventions.
- There are no protected demographic attributes. Channel and device slices are operational diagnostics, not a fairness audit.
- Amounts use one assumed currency. There are no exchange rates, settlement reversals, fees, merchant balances or actual financial loss records.
- Transactions are globally ordered. The generator does not reproduce network delivery delays or timestamp corruption.
- Age fields trend with time and can drift even without fraud change. PSI alerts on them require interpretation.
- Anonymized real competition fields may not reveal when information was available. That gap cannot be fixed by a chronological split alone.

