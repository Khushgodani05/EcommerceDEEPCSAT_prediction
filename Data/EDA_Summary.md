# DeepCSAT — Shopzilla Customer Satisfaction Analysis
### Exploratory Data Analysis | Executive Report

---

## At a Glance

| Dataset Size | Period | Target | Clean Records |
|:---:|:---:|:---:|:---:|
| **85,907 records · 20 features** | Jan – Dec 2023 | CSAT Score (1–5) | **79,738** |

> **Goal:** Predict real-time CSAT scores to replace slow, incomplete survey-based methods — enabling faster, data-driven service improvements.

---

## Data Quality & Preprocessing

| Action | Detail |
|---|---|
| **Drop threshold** | Features with > 40% missing values removed |
| **Dropped features** | `customerremarks` (66.5%), `orderdatetime` (80%), `customercity` (80.2%), `productcategory` (80%), `itemprice` (80%), `connectedhandlingtime` (99.7%) |
| **Also removed** | `uniqueid`, `orderid` — non-predictive identifiers |
| **Invalid rows removed** | 6,169 rows with negative response time |
| **Engineered features** | `responsehours`, `responsehourscategory` |
| **Final shape** | **79,738 rows · 14 features** |

---

## Key Findings

### 1 · Channel Performance

| Channel | CSAT Score 5 |
|---|:---:|
| Outcall | **70.31%** |
| Inbound | 69.59% |
| Email | 60.64% |

> Outcall leads satisfaction. **Invest in outbound engagement** to drive CSAT gains.

---

### 2 · Issue Categories

> **Returns (51.3%) + Order Related (27.0%) = 78.1% of all issues** — yet 54.7% of customers in these categories still rate Shopzilla a 5. Fix here, and overall CSAT climbs fast.

| Top Sub-Categories | Share |
|---|:---:|
| Reverse Pickup Enquiry | 26.1% |
| Return Request | 9.9% |
| Order Delayed | 8.6% |
| Order Status Enquiry | 8.1% |

---

### 3 · Response Time & SLA

| Response Window | % of Cases | CSAT 5 Rate |
|---|:---:|:---:|
| **Within 6 hours** | **89.1%** | **61.7%** |
| 6–12 hours | 3.0% | 2.1% |
| 12–24 hours | 2.4% | 1.7% |
| 24+ hours | 5.5% | 3.8% |

> **89% of issues resolved within 6 hours.** Across all time windows, CSAT 5 consistently dominates — strong SLA performance.

---

### 4 · Seasonal Pattern

| Metric | Peak Month | Share |
|---|:---:|:---:|
| Issues Reported | **August** | **66.0%** |
| Issues Responded | **August** | **66.4%** |
| Survey Responses | August (entire month) | — |

> August is a pressure point. **Proactive capacity planning** before August is critical.

---

### 5 · Agent & Team Performance

| Metric | Finding |
|---|---|
| **Top Agent** | Wendy Taylor — 280 CSAT 5 scores |
| **Top Supervisor** | Carter Park — 2,707 CSAT 5 scores |
| **Top Manager** | John Smith — 23,632 interactions managed |
| **Best Tenure Group** | >90 days (28,617 cases) & On-Job-Training (23,221 cases) both drive majority of CSAT 5 |
| **Best Shift** | Morning (38,446 cases) — CSAT 5 majority across all shifts |

> All 6 managers lead teams with a **CSAT 5 majority** — leadership quality is uniformly strong.

---

### 6 · Geographic & Product Snapshot

**Top Cities by Volume:** Hyderabad (657) › New Delhi (636) › Pune (409) › Mumbai (368) › Bangalore (328)

**Top Products by Interaction Volume:** Electronics (4,377) › LifeStyle (3,816) › Books & General Merchandise (3,067)

> Focus **regional support depth** on Hyderabad & Delhi. Improve **product quality** for Electronics & Lifestyle to cut issue volume at the source.

---

## Strategic Recommendations

| Priority | Action |
|:---:|---|
| 🔴 **High** | Streamline **Returns & Reverse Pickup** workflows — impacts 78% of all issues |
| 🔴 **High** | Scale support capacity in **July–August** before the seasonal surge |
| 🟡 **Medium** | Replicate **Wendy Taylor & Carter Park's** practices team-wide |
| 🟡 **Medium** | Deepen support presence in **Hyderabad & New Delhi** |
| 🟢 **Ongoing** | Sustain the **<6-hour response SLA** — it directly correlates with CSAT 5 |
| 🟢 **Ongoing** | Invest in **Outcall channel** — highest CSAT 5 rate at 70.31% |

---

*Source: `CSAT_EDA.ipynb` · Shopzilla DeepCSAT Project · Data Period: Jan–Dec 2023*