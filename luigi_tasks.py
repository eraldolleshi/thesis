import luigi
from luigi.util import requires
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import subprocess
import itertools
import pathlib
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import accuracy_score
import har_class
from benchmark_class import Benchmark
import cifar_class
import mnist_class
import emg_nn_class
import emg_rf_class
import emg_knn_class
import emg_svm_class
import mnist_nn_class
import mnist_rf_class
import mnist_knn_class
import mnist_svm_class
import cifar_cnn_class
import cifar_mlp_class
import cifar_rf_class
BASELINE = 1
RESEND = 2
DROP = 3
BIT_FLIP = 1
SET_TO_0 = 2
SET_TO_1 = 3


class config(luigi.Config):
    base_output: pathlib.Path = luigi.PathParameter(default="outputs")
    simulator_bin_folder_path: pathlib.Path = luigi.PathParameter(default="bsim/bin")

    @property
    def train_dataset_path(self):
        return (
            self.base_output / "train_datasets"
        )  # TODO once sending the training datasets through the simulator, change the path to where the simulator output path

    @property
    def plot_path(self):
        return self.base_output / "plots"

    @property
    def model_path(self):
        return self.base_output / "models"

    @property
    def results_path(self):
        return self.base_output / "results"


class RunSimulation(luigi.Task):
    input_path = luigi.Parameter()
    simulation_length = luigi.IntParameter()
    benchmark_name = luigi.Parameter()
    bytes_within_package = luigi.NumericalParameter(
        var_type=int, min_value=1, max_value=20
    )
    scenario = luigi.ChoiceParameter(choices=[BASELINE, RESEND, DROP], var_type=int)
    bit_error_rate = luigi.NumericalParameter(
        var_type=float, min_value=0.0, max_value=1.0
    )
    sending_train_dataset = luigi.BoolParameter()
    error_type = luigi.ChoiceParameter(choices=[BIT_FLIP,SET_TO_0,SET_TO_1], var_type=int)
    filler_value = luigi.ChoiceParameter(choices=[-128,0,127,255], var_type=int)

    @property
    def benchmark(self):
        benchmark = Benchmark.benchmark_classes[self.benchmark_name]()
        return benchmark

    @property
    def file_name(self):
        return (
            f"{self.benchmark.application_name}_{self.scenario}_{self.bit_error_rate}_{self.filler_value}_{self.error_type}"
        )
    # @property
    # def resources(self):
       
    #     return {'file_name': 1}

    @property
    def command_central_device(self):
        return f"./bs_nrf52_bsim_samples_bluetooth_central_ht -s={self.file_name}.{self.sending_train_dataset} -d=1"

    @property
    def command_peripheral_device(self):
        return f"./bs_nrf52_bsim_samples_bluetooth_peripheral_ht -s={self.file_name}.{self.sending_train_dataset} -d=0"

    @property
    def command_phy_layer_simulator(self):
        return f"./bs_2G4_phy_v1 -s={self.file_name}.{self.sending_train_dataset} -D=2 -sim_length={self.simulation_length}e6 -defmodem=Magic -argsdefmodem -BER={self.bit_error_rate}"

    def run(self):
        # Execute the commands to run the Bluetooth simulation

        
        env = {
            "INPUT_PATH": f"{self.input_path}",
            "OUTPUT_PATH": f"outputs/{self.file_name}_{self.sending_train_dataset}.bin",  # TODO create a path at the config class
            "SIZE": str(self.benchmark.sample_size),
            "NUMBER": str(self.bytes_within_package),
            "SCENARIO": str(self.scenario),
            "ERROR_TYPE" : str(self.error_type),
            "FILLER_VALUE" : str(self.filler_value)
        }

        processes = [
            subprocess.Popen(
                cmd, env=env, shell=True, cwd=config().simulator_bin_folder_path
            )
            for cmd in [
                self.command_central_device,
                self.command_peripheral_device,
                self.command_phy_layer_simulator,
            ]
        ]

        for process in processes:
            process.wait()

    def output(self):
        return luigi.LocalTarget(
            config().simulator_bin_folder_path
            / f"outputs/{self.file_name}_{self.sending_train_dataset}.bin"  # TODO create a path at the config class
        )


@requires(RunSimulation)
class RunTraining(luigi.Task):
    def output(self):
        return luigi.LocalTarget(
            config().model_path / f"{self.requires().file_name}_{self.requires().benchmark.model_name}.pkl"
        )

    def run(self):
        # Run the ML model
        x_train, y_train = self.requires().benchmark.reconstruct_training_dataset(self.input().path)
        y_train = self.requires().benchmark.encode_labels(y_train)
        model = self.requires().benchmark.run_training(x_train, y_train)
        self.requires().benchmark.save_model(model, self.output().path)


class RunInference(luigi.Task):
    benchmark_name = luigi.Parameter()
    bytes_within_package = luigi.NumericalParameter(
        var_type=int, min_value=1, max_value=20
    )
    scenario = luigi.ChoiceParameter(choices=[BASELINE, RESEND, DROP], var_type=int)
    bit_error_rate = luigi.NumericalParameter(
        var_type=float, min_value=0.0, max_value=1.0
    )
    train_ber = luigi.NumericalParameter(
        var_type=float, min_value=0.0, max_value=1.0
    )  # shows the values of bit error rate used when sending the training dataset through the simulator; added to be able to combine different values of bit error rate between test and train dataset
    sending_train_dataset = luigi.BoolParameter()
    error_type = luigi.ChoiceParameter(choices=[BIT_FLIP,SET_TO_0,SET_TO_1], var_type=int)
    filler_value = luigi.ChoiceParameter(choices=[-128,0,127,255], var_type=int)
    @property
    def benchmark(self):
        benchmark = Benchmark.benchmark_classes[self.benchmark_name]()
        return benchmark

    def requires(self):
        return RunSimulation(
            simulation_length = self.benchmark.test_simulation_length,
            input_path = self.benchmark.path_to_test_dataset,
            benchmark_name=self.benchmark_name,
            bytes_within_package=self.bytes_within_package,
            scenario=self.scenario,
            bit_error_rate=self.bit_error_rate,
            sending_train_dataset=False,
            error_type = self.error_type,
            filler_value = self.filler_value
        ), RunTraining(
            simulation_length = self.benchmark.train_simulation_length,
            input_path = self.benchmark.path_to_train_dataset,
            benchmark_name=self.benchmark_name,
            bytes_within_package=self.bytes_within_package,
            scenario=3,
            bit_error_rate=self.train_ber,
            sending_train_dataset=True,
            error_type = self.error_type,
            filler_value = self.filler_value
        )

    def run(self):
        output_path = self.input()[0].path
        x_test, y_test = self.requires()[0].benchmark.reconstruct_test_dataset(
            output_path
        )
        y_test = self.requires()[0].benchmark.encode_labels(y_test)
        model = self.requires()[0].benchmark.load_model(self.input()[1].path)
        y_pred = self.requires()[0].benchmark.run_inference(model, x_test)

        data_dict = {
            "y_test": y_test.tolist(),
            "y_pred": y_pred.tolist(),
        }

        with self.output().open("w") as f:
            json.dump(data_dict, f)

    def output(self):
        return luigi.LocalTarget(
            config().results_path
            / f"{self.requires()[0].file_name}_{self.train_ber}_{self.requires()[0].benchmark.model_name}.json"
        )

# This class plots the difference in loss inference for the different scenarios. 
# One plot for each model(models trained with difference ber values). 
# All plot of this class are outputed in the same pdf file.
class PlotCompareScenarios(luigi.Task):
    benchmark_name = luigi.Parameter()
    bytes_within_packages = luigi.ListParameter(list(range(1, 21)))
    scenarios = luigi.ListParameter([1,3])
    bit_error_rates = luigi.ListParameter([0.0,0.0025,0.005,0.0075,0.0125,0.0175])
    sending_train_dataset = luigi.ListParameter([False])
    error_types = luigi.ListParameter([BIT_FLIP])
    ber_to_plot = luigi.ListParameter([0]) #added this parameter to be able to choose which values of traininig ber to plot
    filler_values = luigi.ListParameter([])

    @property
    def benchmark(self):
        benchmark = Benchmark.benchmark_classes[self.benchmark_name]()
        self.filler_values = [benchmark.filler_value_min, benchmark.filler_value_average, benchmark.filler_value_max]
        #self.filler_values = [benchmark.filler_value_min]
        return benchmark

    def requires(self):
        return [
            RunInference(
                train_ber=ber_value,
                benchmark_name=self.benchmark_name,
                bytes_within_package=self.bytes_within_packages,
                scenario=scen,
                bit_error_rate=value,
                filler_value = filler,
                sending_train_dataset=self.sending_train_dataset,
                error_type = error
            )
            for value, scen, ber_value, filler, error in list(
                itertools.product(
                    self.bit_error_rates, self.scenarios, self.bit_error_rates, self.filler_values, self.error_types
                )
            )
        ]

    def output(self):
        return luigi.LocalTarget(
            config().plot_path
            / f"{self.benchmark.application_name}_{self.benchmark.model_name}.pdf"  # TODO set the file path at the config class
        )

    def run(self):
        scenario_dict = {}
        # Iterate over each value in self.train_bers
        for error in self.error_types:
         for filler in self.filler_values:
          for train_ber_value in self.bit_error_rates:
            for s in self.scenarios:
                key = (
                    train_ber_value,
                    s,
                    filler,
                    error
                )  
                scenario_dict[key] = {"bit error rate": [], "accuracy drop": []}
        for task in self.requires():
            with task.output().open("r") as file:
                data_dict = json.load(file)
            key = (task.train_ber, task.scenario, task.filler_value, task.error_type)
            scenario_dict[key]["bit error rate"].append(task.bit_error_rate)
            scenario_dict[key]["accuracy drop"].append(
                accuracy_score(data_dict["y_test"], data_dict["y_pred"])
            )
        
        fig, ax = plt.subplots(
            1,
            len(self.ber_to_plot)*len(self.filler_values)*len(self.error_types),
            figsize=(6*len(self.ber_to_plot)*len(self.filler_values)*len(self.error_types),4.5),
            sharey=True,
        )
        plt.tight_layout(pad=3.0)
        plt.subplots_adjust(hspace=0.6)
        # if not isinstance(ax, list):
        #            ax = [ax] 
        ax_index = 0
        for error in self.error_types:
         for filler in self.filler_values:
          for ber_value in self.ber_to_plot:
              for scen in self.scenarios:
                  baseline_accuracy = max(
                    scenario_dict[ber_value, scen, filler, error ]["accuracy drop"]
                )  # TODO make sure it is correct value; check also the accuracy drop naming

                
                  ber_values_percentage = [value * 100 for value in scenario_dict[ber_value, scen,filler, error]["bit error rate"]]
                  accuracy_percentage = [value * 100 for value in scenario_dict[ber_value, scen,filler, error]["accuracy drop"]]
                  color, linestyle = {
                    1: ("green", "solid"),
                    2: ("blue", "solid"),
                    3: ("red", "solid"),
                }[scen]
                  if scen == 1:
                     label_text = "BASELINE"
                  elif scen == 2:
                     label_text = "RESEND"
                  elif scen == 3:
                     label_text = "DROP"
                  ax[ax_index].plot(
                      ber_values_percentage,
                      accuracy_percentage,
                      color=color,
                    linestyle=linestyle,
                    label=f"Scenario {label_text}",
                )

           
              ax[ax_index].set_xlabel("Bit Error Rate(%)")
              ax[0].set_ylabel("Inference accuracy (%)")
              
              ax[ax_index].set_ylim(bottom=30, top=100)
              # Remove y-label for all but the first subplot
              if ax_index % len(self.ber_to_plot) != 0:
                ax[ax_index].set_yticklabels([])

            
            #  ax[ax_index].set_title(f"Baseline accuracy: {baseline_accuracy:.2%}\nBit error rate used for training:{ber_value:.2%}\nFiller value: {filler}")
              ax[ax_index].legend()
              ax_index = ax_index + 1
          plt.savefig(self.output().path)


# TODO add another plotting class which outputs the difference in loss inference when using models trained with different bit error rate values.One plot for each scenario


class PlotTrainedWithDifferentBers(luigi.Task):
    benchmark_name = luigi.Parameter()
    bytes_within_packages = luigi.ListParameter(list(range(1, 21)))
    scenarios = luigi.ListParameter([3])
    bit_error_rates = luigi.ListParameter([0.0,0.002,0.004,0.006,0.008,0.009])
    sending_train_dataset = luigi.ListParameter([True, False])
    error_types = luigi.ListParameter([BIT_FLIP])
    ber_to_plot = luigi.ListParameter([0.0,0.002,0.004,0.006]) #added this parameter to be able to choose which values of traininig ber to plot
    filler_values = luigi.ListParameter([])

    @property
    def benchmark(self):
        benchmark = Benchmark.benchmark_classes[self.benchmark_name]()
        self.filler_values = [benchmark.filler_value_min]
        return benchmark

    def requires(self):
        return [
            RunInference(
                train_ber=ber_value,
                benchmark_name=self.benchmark_name,
                bytes_within_package=self.bytes_within_packages,
                scenario=scen,
                bit_error_rate=value,
                filler_value = filler,
                sending_train_dataset=self.sending_train_dataset,
                error_type = error
            )
            for value, scen, ber_value, filler, error in list(
                itertools.product(
                    self.bit_error_rates, self.scenarios, self.bit_error_rates, self.filler_values, self.error_types
                )
            )
        ]


    def output(self):
        return luigi.LocalTarget(
            config().plot_path / f"{self.benchmark.application_name}_{self.benchmark.model_name}_secondplot.pdf"#TODO change the file path at the config class
        )

    def run(self):

        scenario_dict={}
        # Iterate over each value in self.train_bers
        for error in self.error_types:  
         for filler in self.filler_values:
          for train_ber_value in self.bit_error_rates:
            for s in self.scenarios:
               key = (train_ber_value, s,filler,error)  # Create a tuple key combining s and train_ber_value
               scenario_dict[key] = {"bit error rate": [], "accuracy drop": []}
        for task in self.requires():
           
            with task.output().open("r") as file:
                data_dict = json.load(file)
            key = (task.train_ber,task.scenario,task.filler_value,task.error_type)    
            scenario_dict[key]["bit error rate"].append(task.bit_error_rate)
            scenario_dict[key]["accuracy drop"].append(
                accuracy_score(data_dict["y_test"], data_dict["y_pred"])
            )

          
               
        fig, ax = plt.subplots(len(self.filler_values)*len(self.scenarios), 1, figsize=(6, 4 *len(self.filler_values)*len(self.scenarios)), sharex=False)
        plt.subplots_adjust(hspace=0.5)
        if not isinstance(ax, list):
                 ax = [ax] 
        offset = 1e-6  # Adding an offset to the values of the y axis, to avoid having problems with 0 values, since log scale is used
        ax_index = 0
        for error in self.error_types:
         for filler in self.filler_values:
          for ber_value in self.ber_to_plot:

             baseline_accuracy = max(
                scenario_dict[ber_value, 3,filler,error]["accuracy drop"]
            )  # TODO make sure it is correct value; check also the accuracy drop naming

          
             min_ber, max_ber = self.ber_to_plot[0], self.ber_to_plot[-1]

             cmap = plt.get_cmap('RdYlGn')
             norm = Normalize(vmin=min_ber, vmax=max_ber)
             normalized_value = norm(ber_value)

             color = cmap(normalized_value)

             color = "#{:02X}{:02X}{:02X}{:02X}".format(int(color[0] * 255), int(color[1] * 255), int(color[2] * 255), int(color[3] * 255))
             linestyle = "solid"
             ber_values_percentage = [value * 100 for value in scenario_dict[ber_value, 3, filler,error]["bit error rate"]]
             accuracy_percentage = [value * 100 for value in scenario_dict[ber_value, 3,filler, error]["accuracy drop"]]
             if ber_value == 0.009:
                ax[ax_index].plot(
        ber_values_percentage,
        accuracy_percentage,
        color=color,
        linestyle=linestyle,
        label="Combined training dataset",
    )
             else:
              ax[ax_index].plot(
        ber_values_percentage,
        accuracy_percentage,
        color=color,
        linestyle=linestyle,
        label=f"Trained with BER: {ber_value:.2%}",
    )


        #  ax[ax_index].set_yscale("log")
          ax[ax_index].set_xlabel("Bit Error Rate (%)")
          ax[ax_index].set_ylabel("Inference accuracy (%)")
          ax[ax_index].set_ylim(bottom=40, top=100)
          ax[ax_index].legend()
          ax_index = ax_index + 1

        # Save the plot to the specified file path
        plt.savefig(self.output().path)   

class PlotFillerValues(luigi.Task):
    benchmark_name = luigi.Parameter()
    bytes_within_packages = luigi.ListParameter(list(range(1, 21)))
    scenarios = luigi.ListParameter([1,2,3])
    bit_error_rates = luigi.ListParameter([0.0,0.001,0.002,0.003])
    sending_train_dataset = luigi.ListParameter([True, False])
    error_type = luigi.ListParameter([BIT_FLIP,SET_TO_0,SET_TO_1])
    ber_to_plot = luigi.ListParameter([0.0,0.001,0.002,0.003]) #added this parameter to be able to choose which values of traininig ber to plot
    filler_values = luigi.ListParameter([])

    @property
    def benchmark(self):
        benchmark = Benchmark.benchmark_classes[self.benchmark_name]()
        self.filler_values = [benchmark.filler_value_min, benchmark.filler_value_average, benchmark.filler_value_max]
        return benchmark

    def requires(self):
        return [
            RunInference(
                train_ber=ber_value,
                benchmark_name=self.benchmark_name,
                bytes_within_package=self.bytes_within_packages,
                scenario=scen,
                bit_error_rate=value,
                filler_value = filler,
                sending_train_dataset=self.sending_train_dataset,
                error_type = self.error_type
            )
            for value, scen, ber_value, filler in list(
                itertools.product(
                    self.bit_error_rates, self.scenarios, self.bit_error_rates, self.filler_values
                )
            )
        ]


    def output(self):
        return luigi.LocalTarget(
            config().plot_path / f"{self.benchmark.application_name}_{self.benchmark.model_name}_thirdplot.pdf"#TODO change the file path at the config class
        )

    def run(self):

        scenario_dict={}
        # Iterate over each value in self.train_bers
        for filler in self.filler_values:
          for train_ber_value in self.bit_error_rates:
            for s in self.scenarios:
               key = (train_ber_value, s,filler)  # Create a tuple key combining s and train_ber_value
               scenario_dict[key] = {"bit error rate": [], "accuracy drop": []}
        for task in self.requires():
           
            with task.output().open("r") as file:
                data_dict = json.load(file)
            key = (task.train_ber,task.scenario,task.filler_value)    
            scenario_dict[key]["bit error rate"].append(task.bit_error_rate)
            scenario_dict[key]["accuracy drop"].append(
                accuracy_score(data_dict["y_test"], data_dict["y_pred"])
            )

          
               
        fig, ax = plt.subplots(len(self.ber_to_plot), 1, figsize=(6, 4 *len(self.ber_to_plot)), sharex=False)
        plt.subplots_adjust(hspace=0.5)
        offset = 1e-6  # Adding an offset to the values of the y axis, to avoid having problems with 0 values, since log scale is used
        ax_index = 0
        for ber_value in self.bit_error_rates:
          for filler in self.filler_values:
             color, linestyle = {
                 -128: ("red", "solid"),
                    0:("yellow","solid"),
                127: ("green", "solid"),
                255: ("blue", "solid"),
                
                #TODO make sure to have all the values of bit error rate set as the listParameter of the task
                
             }[filler]
             ber_values_percentage = [value * 100 for value in scenario_dict[ber_value, 3, filler]["bit error rate"]]
             ax[ax_index].plot(
                ber_values_percentage,
                scenario_dict[ber_value, 3, filler]["accuracy drop"],
                color=color,
                linestyle=linestyle,
                label=f"Filler value: {filler}",
            )

        #  ax[ax_index].set_yscale("log")
          ax[ax_index].set_xlabel("BER (%)")
          ax[ax_index].set_ylabel("Inference accuracy (%)")
          ax[ax_index].set_title(f"Error injection scenario: Drop\n Trained with:{ber_value}")
          ax[ax_index].legend()
          ax_index = ax_index + 1

        # Save the plot to the specified file path
        plt.savefig(self.output().path)   
        