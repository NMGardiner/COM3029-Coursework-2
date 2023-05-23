# COM3029-Coursework-2
Group 9

## Stress Testing
Start our webservice on windows with 
```
waitress-serve --listen=127.0.0.1:5000 webserver:app
```
or run the notebook webserver-notebook.ipynb

Run a stress test from command line and export to a log file with

```
locust -f locustfile.py --host=http://localhost:5000 --headless  --logfile=results.log -u <user_count> -r <spawn_rate> --run-time <duration_of_test_seconds>s --csv=metrics
```
or run as a python script
```
python stress_test.py -u <user_count> -r <spawn_rate> -l <duration_of_test>
```