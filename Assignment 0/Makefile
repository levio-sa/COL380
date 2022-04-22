CC=g++
#g++ -std=c++11 -O2 -mavx2 -c classify.cpp 
CFLAGS=-std=c++11 -O2 -fopenmp

sources=main.cpp classify.cpp
objects=$(sources:.cpp=.o)

classify:$(objects)
	$(CC) $(CFLAGS) $^ -o $@

run: classify
#	 ./classify rfile dfile 15 4 3
	 ./classify rfile dfile 1009072 4 3

val: classify
#	valgrind --tool=cachegrind ./classify rfile dfile 12 4 1 
#	valgrind --leak-check=full ./classify rfile dfile 12 4 1 
#	valgrind --tool=cachegrind ./classify rfile dfile 1009072 4 3
	valgrind --tool=cachegrind ./classify rfile dfile 1009072 4 3		

gprof: classify
	gprof classify

gprofA: classify
	gprof -A classify

gprofZ: classify
	gprof -Z classify


#classify.o: classify.h
#	touch classify.o
#	$(CC) $(CFLAGS) -c $<
#main.o: classify.h
#	$(CC) $(CFLAGS) -c $<

%.o: %.cpp classify.h
	$(CC) $(CFLAGS) -c $<

clean:
	rm main.o classify.o classify