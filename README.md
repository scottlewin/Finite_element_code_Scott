# Finite_element_code_Scott

Can we use a simple finite element solver to model pollutant spread?

This code contains a finite element solver which solves the 2D timedependent advection diffusion equation on a grid. The intent is to model pollutant spread from Southampton and observe how much passes over the nearby Reading, shown in pink, and 'Reading', shown in orange. When run, it will plot snapshots of the simulation every 1000 timesteps, a graph of the pollutant concentration in 'Reading' over time, and print out the pollutant concentration at both Southampton and 'Reading' at the time at which they both at a maximum. 

How to use: 
The desired grid resolution can be chosen in the section of the code marked as #Setup grid. It can be manually altered from e.g. 5k to 20k resolution. Length of simulation and timestep can be changed in the line of code beneath #Solve. Once the desired simulation parameters are chosen, simply press run and the results from the report will be printed out. 
