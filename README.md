# Setup

Use the instructions in the following link to install Zephyr: [Zephyr Installation](https://docs.zephyrproject.org/latest/develop/getting_started/index.html).
After installing Zephyr, replace the src directory of the peripheral_ht and central_ht samples with the ones provided in this repository.

To create the simulated setup, build the peripheral_ht and central_ht Zephyr applications targeting the simulated nrf52_bsim board. The instructions to install Babblesim and build the peripheral and central are shown in the following link: [Instructions](https://docs.zephyrproject.org/2.7.5/boards/posix/nrf52_bsim/doc/index.html). After installing Babblesim, replace the p2G4_main.c and p2G4_dump.c files at ext_2G4_phy_v1 directory of bsim with the files provided and add the input datasets at the directory shown in this repository.
The structure of the directory is shown below.
- root
  - bsim
  - outputs
      - models
      - plots
      - results
  
Also the python codes are located in the root directory. Any changes in the directory structure should be associated with the proper changes in the path properties at the luigi_tasks.py file. 

# Code Structure

The benchmark class is an abstract class which shows the attributes and method that each benchmark should define. For each of the ML applications conducted, a benchmark subclass is created. To run each application using different ML algorithms, subclasses of each application are created where the ML models are defined. Adding another ML model is as simple as adding another  subclass. Furthermore, to add another ML application, a new benchmark subclass can be created.

Run the main.py file to start the execution of the pipeline. Set the benchmark name using this format: {dataset name}_{ML model name}. For example, to run Random Forest for MNIST dataset, set the benchmark name as "mnist_rf". The different settings, such as the values of Bit Error   Rate injected in the test and training samples, error injection scenario, bit error type, replacement value for the lost bytes, are set at the plotting task. The figure below shows the structure of the pipeline created. 

![figure](https://github.com/eraldolleshi/thesis/blob/main/luigitasks.jpg?raw=true)
