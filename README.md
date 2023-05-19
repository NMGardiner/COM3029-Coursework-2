# COM3029-Coursework-2
Group 9

## Stress Testing
Run a stress test from command line and export to a log file with

```
locust -f locustfile.py --host=http://localhost:5000 --no-web  --hatch-rate=<spawns_per_second> --clients=<users_per_spawn> --only-summary --statsfile=result.log
```