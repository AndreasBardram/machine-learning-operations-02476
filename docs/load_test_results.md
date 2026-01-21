# Load Test Results

Target: `https://ml-ops-api-1070209272290.europe-north1.run.app`  
Tooling: Locust (`locustfile.py`)  
Run date: 2026-01-21 (local system time)

## Test configuration

Baseline (batch size 1)
- Users: 25
- Spawn rate: 5 users/second
- Duration: 60s
- Payload: single `text`

Batch test (batch size 8)
- Users: 25
- Spawn rate: 5 users/second
- Duration: 60s
- Payload: `texts` with 8 entries

## Results summary

Baseline (batch size 1)
- Aggregated: 1545 requests, 0 failures, 25.97 req/s
- Aggregated latency (ms): avg 330, median 160, min 34, max 2255
- `/predict` latency (ms): avg 377, median 170, min 60, max 2255
- `/health` latency (ms): avg 107, median 61, min 34, max 670
- Aggregated percentiles (ms): p50 240, p90 1500, p95 1700, p99 1900, p99.9 2300

Batch test (batch size 8)
- Aggregated: 843 requests, 0 failures, 14.16 req/s
- Aggregated latency (ms): avg 1097, median 360, min 35, max 6352
- `/predict` latency (ms): avg 1269, median 440, min 93, max 6352
- `/health` latency (ms): avg 167, median 88, min 35, max 1265
- Aggregated percentiles (ms): p50 310, p90 4200, p95 4800, p99 5700, p99.9 5900

## Notes

- Raw Locust outputs are stored as CSVs in the repo root:
  - `locust_baseline_stats.csv`
  - `locust_baseline_stats_history.csv`
  - `locust_batch8_stats.csv`
  - `locust_batch8_stats_history.csv`
