Abbreviated Report of my findings using Locust as a local and cloud host, split into sections for organization and viewing pleasure.

-----
Response Times

Localhost:
MedianMedian (50%) response times: ~65–80 ms
95th percentile: ~90–100 ms
Zero network overhead has created very consistent low-latency performance.

Cloud Run:
Median response times: ~500–800 ms for most endpoints
95th percentile: ~1,000–1,200 ms
Higher latency with network transit, cold starts, and model load time likely contribute.

My assumption is that the cloud ran 6 to 10x slower than locally due to poor infastructure and model initilization.

-----
Failure Rates

Localhost had a 0% failure rate across every single scenario.
Cloud was not as effective, as it occasionally spiked to 1-2% on high-concurrency scenarios. I believe it is due to either cold starts, request timeouts, or both of them at the same time.

-----
Throughput

Localost: Sustains 150-170 req/s under load.
Cloud: Peaked at 15-25 req/s before latency and occasional failures.

This means that the Cloud was roughly 8x lower for throughput when compared to the local. Based on the log report, I suspect this is because of container CPU/memory limits and sequential model execution.

-----
Bottlenecks

Model Loading Time: Cloud showed first request latency problems on a cold start (first 2-3 seconds), especially with no recent traffic.
Concurrency Limit: Cloud concurrency was low, often hitting a ceiling of 80 per instance.
Network Overhead: Internet routing and the HTTPS likely added 100-200 ms per request.
Serialization: Python single-thread handling may be a cause for the model's slow input/output for the Cloud.

-----
Recommendations

Scaling: Increase cloud run concurrency, use minimum instances more often, optimize model load, enable autoscaling.
Optimization: Support batch requests and reduce payloads.