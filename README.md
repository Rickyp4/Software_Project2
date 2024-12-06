# Software_Project2

The project is in the file "kernal.cu".
The project was created in visual studio, and all relevant files are in the github. Other than the CUDA toolkit, there should be no other dependencies.
The pen size and goat density can be changed within function main.

At a map size of 20x20 (22x22 including borders) and goat density of 50%, the cpu implementation finished 0.15 seconds after 12,709 cycles,
while the gpu implementation took 38 seconds and 23,524 cycles. These are from a single test. The current project uses the C++ rand() function,
which uses the same seed every time. The cpu and gpu algorithms are also slightly different, partially due to the need to handle collisions.

The current output.txt file is for a map size of 11x11 (13x13 including borders) and goat density of 50%. The number of iterations was 3,939.
