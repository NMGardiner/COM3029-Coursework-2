import gevent
from locust import HttpUser, task, events
from locust.env import Environment
from locust.stats import stats_printer, stats_history, StatsCSVFileWriter
from locust.log import setup_logging

from locustfile import StressTest

import matplotlib.pyplot as plt
import pandas as pd

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-u", "--user_count", dest="user_count", help="How many users to be created during a spawn", type=int, default=1)
    parser.add_argument("-r", "--spawn_rate", dest="spawn_rate", help="How many groups of users should be created every second", type=int, default=1)
    parser.add_argument("-l", "--length", dest="duration", help="How long the stress test should last for", type=int, default=30)

    stress_args = parser.parse_args()


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
    runner.start(user_count=stress_args.user_count, spawn_rate=stress_args.spawn_rate)

    # in duration seconds stop the runner
    gevent.spawn_later(stress_args.duration, lambda: runner.quit())

    # wait for the greenlets
    runner.greenlet.join()

    # stop the web server for good measures
    web_ui.stop()

    graph_stats = pd.read_csv('_stats_history.csv')
    graph_stats = graph_stats[graph_stats['Name'] == 'Aggregated']
    x_axis = graph_stats['Timestamp']
    x_axis = x_axis - x_axis.min()

    target_metrics = ['Requests/s','Failures/s','Total Request Count','Total Failure Count','Total Average Response Time']

    for metric in target_metrics:
        y_axis = graph_stats[metric].astype(float)

        plt.plot(list(x_axis.values), list(y_axis.values))
        plt.title(metric)
        save_name = metric.replace('/', '_per_')
        plt.savefig(f'{save_name}.png')
        plt.clf()