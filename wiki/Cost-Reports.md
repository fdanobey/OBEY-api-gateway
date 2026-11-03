# Cost Reports

OBEY API Gateway persists a per-request USD cost in the request log (`logs.db`) and accrues provider and virtual-key spend in separate budget stores. Cost Reports aggregate that durable ledger into stable, explainable, restart-proof reports for operational review, finance handoff, and provider-invoice reconciliation.

Reports are read-only, require no configuration, and never read in-memory gauges. Restarting the gateway does not change historical report output.

---

## Cost Report Endpoint

```text
GET /admin/cost-reports?start=<RFC3339>&end=<RFC3339>&group_by=<day|provider|model|virtual_key>&provider=<name>&key_id=<id>&format=<csv|json>
```

Query parameters:

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `start` | yes | — | Window start as RFC 3339, e.g. `2026-01-01T00:00:00Z` |
| `end` | yes | — | Window end as RFC 3339; must be strictly after `start` |
| `group_by` | no | `day` | `day`, `provider`, `model`, or `virtual_key` |
| `provider` | no | — | Filter to a single provider name (conjunctive with `key_id`) |
| `key_id` | no | — | Filter to a single virtual key id (conjunctive with `provider`) |
| `format` | no | `json` | `json` or `csv` |

Validation rules:

- Missing or unparseable `start` returns HTTP 400 naming `start`.
- Missing or unparseable `end`, or `end <= start`, returns HTTP 400 naming `end`.
- A range longer than 366 days returns HTTP 400 naming `range`.
- An invalid `group_by` or `format` returns HTTP 400 naming that parameter and lists the allowed values.

### Row Fields

Every report returns ordered rows plus a `totals` row over the requested window. Each row contains:

| Field | Meaning |
|-------|---------|
| `dimension` | `YYYY-MM-DD` for `group_by=day`; provider name, model name, or `key_id` for other groupings; `"unattributed"` for virtual-key grouping when a request carried no key |
| `requests` | Request count in the bucket |
| `input_tokens` | Total input/prompt tokens |
| `output_tokens` | Total completion/output tokens |
| `cache_read_tokens` | Total cached-read tokens |
| `cache_creation_tokens` | Total cache-creation tokens |
| `reasoning_tokens` | Total reasoning/thinking tokens |
| `cost_usd` | Aggregated cost in USD, exact to 6 decimal places |
| `cost_unpriced_requests` | Requests whose cost could not be computed (missing model pricing), counted in `requests` but excluded from `cost_usd` |
| `window_start` | Epoch seconds of the bucket start, inclusive |
| `window_end` | Epoch seconds of the bucket end, exclusive |

### Grouping Behavior

- `group_by=day` returns one row per UTC day in the range, including empty days with zero values, in calendar order.
- `group_by=provider`, `model`, or `virtual_key` returns one row per distinct dimension value present in the range, ordered by `cost_usd` descending; ties break by dimension ascending.
- `group_by=virtual_key` attributes requests with no virtual key to the dimension value `"unattributed"`. The sum of `cost_usd` across all virtual-key rows equals the total `cost_usd` of the same range grouped by day (within 2-decimal rounding tolerance).

### JSON Example

```bash
curl -u admin:admin \
  "http://localhost:8080/admin/cost-reports?start=2026-01-01T00:00:00Z&end=2026-01-08T00:00:00Z&group_by=day"
```

```json
{
"rows": [
{
"dimension": "2026-01-01",
"window_start": 1767225600,
"window_end": 1767312000,
"requests": 12,
"input_tokens": 3400,
"output_tokens": 920,
"cache_read_tokens": 0,
"cache_creation_tokens": 0,
"reasoning_tokens": 110,
"cost_usd": 0.00456,
"cost_unpriced_requests": 0
}
],
"totals": {
"dimension": "total",
"window_start": 1767225600,
"window_end": 1767312000,
"requests": 12,
"input_tokens": 3400,
"output_tokens": 920,
"cache_read_tokens": 0,
"cache_creation_tokens": 0,
"reasoning_tokens": 110,
"cost_usd": 0.00456,
"cost_unpriced_requests": 0
}
}
```

### CSV Format

Pass `format=csv` to receive `text/csv` with a single header row and one row per result row. Empty reports return the header only.

- Encoding: UTF-8
- Line endings: CRLF
- `cost_usd` rendered with exactly 6 decimal places

```bash
curl -u admin:admin \
  "http://localhost:8080/admin/cost-reports?start=2026-01-01T00:00:00Z&end=2026-01-08T00:00:00Z&group_by=provider&format=csv"
```

```csv
dimension,requests,input_tokens,output_tokens,cache_read_tokens,cache_creation_tokens,reasoning_tokens,cost_usd,cost_unpriced_requests
openai,12,3400,920,0,0,110,0.004560,0
```

---

## Reconciliation Endpoint

```text
GET /admin/cost-reports/reconcile?start=<RFC3339>&end=<RFC3339>
```

The reconciliation endpoint returns three independent totals for the same window, plus the explicit difference between the ledger and provider spend:

| Field | Meaning |
|-------|---------|
| `ledger.cost_usd` / `ledger.requests` | Totals from the durable request log ledger |
| `provider_spend.cost_usd` / `provider_spend.window_start` / `provider_spend.window_end` / `provider_spend.caveat` | Total accrued provider spend from the provider budget store, with the budget window bounds and a caveat flag |
| `key_usage.cost_usd` | Total virtual-key accrued cost from the key usage store |
| `unpriced_requests` | Requests in the window whose cost could not be computed |
| `difference_usd` | `ledger.cost_usd - provider_spend.cost_usd` |

### Provider Spend Caveat

The provider budget store retains **only the current budget window's accrued spend**. When the requested `start`/`end` window exactly matches the provider's current budget window, the provider spend total is exact. For any other window the store can report only the current window, so the returned `provider_spend` reflects that current window and `provider_spend.caveat` is `true`.

Use this endpoint to detect drift, not as a strict invoice verifier for arbitrary historical windows.

Empty stores return zero totals, not an error.

### Example

```bash
curl -u admin:admin \
  "http://localhost:8080/admin/cost-reports/reconcile?start=2026-01-01T00:00:00Z&end=2026-01-08T00:00:00Z"
```

```json
{
  "ledger": {
    "cost_usd": 12.345678,
    "requests": 1500
  },
"provider_spend": {
"cost_usd": 12.34,
"window_start": 1767225600,
"window_end": 1767312000,
"caveat": true
},
  "key_usage": {
    "cost_usd": 12.345678
  },
  "unpriced_requests": 3,
  "difference_usd": 0.005678
}
```

---

## Unpriced Requests

A request is "unpriced" when the gateway cannot resolve a model price for it (for example, an unknown model or missing pricing entry). Unpriced requests:

- Are counted in `requests` and token fields
- Are excluded from `cost_usd`
- Are reported separately in `cost_unpriced_requests` per row and in the totals

This keeps the total request count auditable while preventing an underestimated cost total from silently presenting as complete.

---

## Restart-Proofness

All report numbers are computed from durable SQLite state:

- Request log ledger (`logs.db`)
- Provider budget store (`budgets.db`)
- Virtual-key usage store (`keys.db`)

In-memory cumulative cost gauges reset on restart; reports do not use them. Re-running the same report before and after a restart or an in-place upgrade returns identical rows for closed past ranges.

---

## Admin Panel

The embedded admin panel includes a **Cost Reports** view under the operations section. It offers:

- Range presets: **Last 7 days**, **Last 30 days**, **Last 90 days**, or a custom start/end range
- Group-by selector: **Day**, **Provider**, **Model**, or **Virtual Key**
- Optional filters: provider name and virtual key id
- A results table with all report columns and a totals row
- **Export CSV** and **Export JSON** buttons that download the currently filtered report

If the chosen filters produce a 400 error, the error message from the response body is surfaced without discarding your selections. An empty report shows an explanatory empty state instead of a blank table.
