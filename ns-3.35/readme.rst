Important information about running ns3!

1. Ns3 runs through the waf tool and this is waht we use to execute simulations!
2. Our projects are constructed inside the scratch folder. 
3. The sole requirement for an executable project is to have one file with a main function that is located by the waf tool. Additional main files may cause problems.
4. A small project that contains only the main file does not necessarily need to be included in a project folder. The user may construct it the way he prefares. Ns3 
   handles both cases,  yet for better organization it is recommended to create project folders.
5. The following steps are necessary to run a simulation in NS3:
  a. Configure the project: ./waf configure
    Note: ./waf configure has 3 profiles to use: debug (default), optimized, release. The first is providing more information about execution but is slow.
    The second and third are preferred for perforamnce showcasing significant boost in speed. In the provided version of Ns3 optimized profile is slightly modified to 
    contain specific prints that the user has added to the source code to be able to execute the code effectively while undestanding execution.
    To choose between profiles one has to do the following: ./waf configure --build-profile=profile_type       ex:  ./waf configure --build-profile=optimized

  b. Build the project: ./waf build
  
  c. Run simulation: ./waf --run your-simulation-script 
    your-simulation-script is searched inside the scratch directory. In case that the script is a sole file containing the main function inside the scratch directory
    then only its name needs to specified. ex: ./waf --run run_example.  This way the run_example.cc is executed that is in /scratch/run_example.cc
    Alternatively, the file is contained in a project directory with one file containing the main function then only the directory is specified:
    Ex. Consider the project: /scratch/wifi_exp/ where the main file is located in /scratch/wifi_exp/main.cc. Then to execute: ./waf --run wifi_exp 
    The above call is sufficient.
    