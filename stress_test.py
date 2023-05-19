import gevent
from locust import HttpUser, task, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history, StatsCSVFileWriter
from locust.log import setup_logging

from locustfile import StressTest

setup_logging("INFO", None)


# setup Environment and Runner
env = Environment(user_classes=[StressTest], events=events)
runner = env.create_local_runner()

# start a WebUI instance
web_ui = env.create_web_ui("127.0.0.1", 8089)

logging = StatsCSVFileWriter(environment=env, base_filepath='./', full_history=True, percentiles_to_report=[90.0])

# execute init event handlers (only really needed if you have registered any)
env.events.init.fire(environment=env, runner=runner, web_ui=web_ui)

# start a greenlet that periodically outputs the current stats
gevent.spawn(stats_printer(env.stats))

gevent.spawn(logging)

# start a greenlet that save current stats to history
gevent.spawn(stats_history, env.runner)

# start the test
runner.start(user_count=100, spawn_rate=1000)

# in 60 seconds stop the runner
gevent.spawn_later(15, lambda: runner.quit())

# wait for the greenlets
runner.greenlet.join()

# stop the web server for good measures
web_ui.stop()