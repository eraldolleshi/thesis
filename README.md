Use the instruction in the following link to install Zephyr: https://docs.zephyrproject.org/latest/develop/getting_started/index.html

To create the simulated setup, build the peripheral and central Zephyr applications targeting the simulated nrf52_bsim board. The instructions to install Babblesim and build the peripheral and central are shown in the following link: 
https://docs.zephyrproject.org/2.7.5/boards/posix/nrf52_bsim/doc/index.html

The benchmark class is an abstract class which shows the attributes and method that each benchmark should define. For each of the ML applications conducted, a benchmark subclass is created. To run each application using different ML algorithms, subclasses of each application are created where the ML models are defined. Adding another ML model is as simple as adding another  subclass. Furthermore, to add another ML application, a new benchmark subclass can be created.

Run the main.py file to start the execution of the pipeline. 
