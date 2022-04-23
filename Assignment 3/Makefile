compiledss:
	mpic++ -std=c++11 -fopenmp -o dss dss.cpp -lstdc++fs
rundss:
	mpirun -np 16 ./dss /scratch/cse/phd/anz198717/TA/COL380/A3/to_students ./conv_main_5

compile:
	mpic++ -std=c++11 -fopenmp -o main main.cpp

run:
	mpirun -np 24 ./main ./conv_main_5 5 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred.dat

run55:
	mpirun --bind-to none ./main ./conv_main_5 5 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred55.dat

run510:
	mpirun --bind-to none ./main ./conv_main_5 10 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred510.dat

run515:
	mpirun --bind-to none ./main ./conv_main_5 15 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred515.dat


run25:
	mpirun --bind-to none ./main ./conv_main_5 5 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred25.dat
run210:
	mpirun --bind-to none ./main ./conv_main_5 10 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred210.dat


run215:
	mpirun --bind-to none ./main ./conv_main_5 15 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred215.dat

run105:
	mpirun --bind-to none ./main ./conv_main_5 5 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred105.dat
run1010:
	mpirun --bind-to none ./main ./conv_main_5 10 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred1010.dat


run1015:
	mpirun --bind-to none ./main ./conv_main_5 15 /scratch/cse/phd/anz198717/TA/COL380/A3/to_students/user.txt pred1015.dat
