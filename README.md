# OpenMP-MPI-tf-idf
Implementation of Tf-IDF in openMp MPI
The code runs on a distributed cluster.

Sample input data is in BOOKS folder and output will be generated in Output Folder

to Run:

	mpiCC "FILE_NAME" -o -fopenmp "EXECUTABLE NAME"

	mpirun -np "NUMBER OF NODES" ./"EXECUTABLE NODE"
