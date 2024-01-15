import luigi_tasks
import luigi


def main():
    tasks_to_run = [
        luigi_tasks.PlotCompareScenarios(
            benchmark_name="mnist_nn",
            bytes_within_packages=20,
        )
    ]

    luigi.build(tasks_to_run, local_scheduler=True,workers=2)


if __name__ == "__main__":
    main()
