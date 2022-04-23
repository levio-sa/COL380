all:
	g++ -std=c++11 -fPIC -g -c -Wall psort.cpp -fopenmp
	g++ -std=c++11 -shared -Wl,-soname,libsort.so -o libsort.so psort.o -lc -fopenmp
	g++ -std=c++11 -fopenmp -Wall psort.cpp driver.cpp -o out

run:
	./out input_10000000.txt 4

val:
	valgrind --tool=cachegrind ./out input_10000000.txt 4
clean:
	rm *.so *.o