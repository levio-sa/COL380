#include "classify.h"
#include <omp.h>

Data classify(Data &D, const Ranges &R, unsigned int numt)
{ // Classify each item in D into intervals (given by R). Finally, produce in D2 data sorted by interval
   assert(numt < MAXTHREADS);
   Counter counts[R.num()]; // I need on counter per interval. Each counter can keep pre-thread subcount.
   
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num(); // I am thread number tid
   //    for(int i=tid; i<D.ndata; i+=numt) { // Threads together share-loop through all of Data
   //       int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key,
	// 						  // and store the interval id in value. D is changed.
   //       counts[v].increase(tid); // Found one key in interval v
   //    }
   // }

   // CB 3
   // int bblock = 1024;
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    for(int x=0; x<D.ndata; x+=bblock){
   //       int y = ((x+bblock)<D.ndata)?(x+bblock):(D.ndata);
   //       int len = (y-x)/numt;
   //       int tmp = tid*len;
   //       if(tid+1<numt){
   //          for(int i=x+tmp; i<x+tmp+len; ++i){
   //             int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key,
   //                         // and store the interval id in value. D is changed.
   //             counts[v].increase(tid); // Found one key in interval v
   //          }
   //       }
   //       else{
   //          for(int i=x+tmp; i<y; ++i){
   //             int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key,
   //                         // and store the interval id in value. D is changed.
   //             counts[v].increase(tid); // Found one key in interval v
   //          }
   //       }
   //    }
   // }

   int sblock = 16;
   int bblock = 16*numt;
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num();
      for(int x=0; x<D.ndata; x+=bblock){
         int y = ((x+bblock)<D.ndata)?(x+bblock):(D.ndata);
         int tmp = x+tid*sblock;
         for(int i=tmp; i<std::min(tmp+sblock,y);++i){
            int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key,
                        // and store the interval id in value. D is changed.
            counts[v].increase(tid); // Found one key in interval v
         }
      }
   }



   // Contiguous blocks
   // # pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    int a = ((tid)*D.ndata)/numt;
   //    int b = ((tid+1)*D.ndata)/numt;
   //    for(int i=a; i<b; ++i){
   //       int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key,
	// 						  // and store the interval id in value. D is changed.
   //       counts[v].increase(tid); // Found one key in interval v
   //    }
   // }

   // Contiguous blocks 2
   // int len = D.ndata/numt;
   // # pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    if(tid+1<numt){
   //       int tmp = tid*len;
   //       for(int i=tmp; i<tmp+len; ++i){
   //          int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key,
   //                      // and store the interval id in value. D is changed.
   //          counts[v].increase(tid); // Found one key in interval v
   //       }
   //    }
   //    else{
   //       for(int i=tid*len; i<D.ndata; ++i){
   //          int v = D.data[i].value = R.range(D.data[i].key);// For each data, find the interval of data's key,
   //                      // and store the interval id in value. D is changed.
   //          counts[v].increase(tid); // Found one key in interval v
   //       }
   //    }
   // }




   // Accumulate all sub-counts (in each interval;'s counter) into rangecount
   unsigned int *rangecount = new unsigned int[R.num()];
   for(int r=0; r<R.num(); r++) { // For all intervals
      rangecount[r] = 0;
      for(int t=0; t<numt; t++) // For all threads
         rangecount[r] += counts[r].get(t);
      // std::cout << rangecount[r] << " elements in Range " << r << "\n"; // Debugging statement
   }

   // Parallel
   // int pad = 16;
   // unsigned int *rangecount = new unsigned int[R.num()*pad];
   // int rlen = R.num()/numt;
   // #pragma omp parallel num_threads(numt)
   // {  
   //    int tid = omp_get_thread_num();
   //    int tmp = tid*rlen;
   //    if(tid+1<numt)

   //       for(int r=tmp; r<tmp+rlen; r++) { // For all intervals
   //          rangecount[r*pad] = 0;
   //          for(int t=0; t<numt; t++) // For all threads
   //             rangecount[r*pad] += counts[r].get(t);
   //       }
   //    else 
   //       for(int r=tmp; r<R.num(); r++) { // For all intervals
   //          rangecount[r*pad] = 0;
   //          for(int t=0; t<numt; t++) // For all threads
   //             rangecount[r*pad] += counts[r].get(t);
   //       }
   //    // std::cout << rangecount[r] << " elements in Range " << r << "\n"; // Debugging statement
   // }




   // Compute prefx sum on rangecount.
   for(int i=1; i<R.num(); i++) {
      rangecount[i] += rangecount[i-1];
   }

   // Now rangecount[i] has the number of elements in intervals before the ith interval.

   Data D2 = Data(D.ndata); // Make a copy
   
   // #pragma omp parallel num_threads(numt)
   // {
   //    int tid = omp_get_thread_num();
   //    for(int r=tid; r<R.num(); r+=numt) { // Thread together share-loop through the intervals 
   //       int rcount = 0;
   //       for(int d=0; d<D.ndata; d++) // For each interval, thread loops through all of data and  
   //           if(D.data[d].value == r) // If the data item is in this interval 
   //               D2.data[((r==0)?0:rangecount[r-1])+rcount++] = D.data[d]; // Copy it to the appropriate place in D2.
   //    }
   // }

   // Without parallel 
   // unsigned int *rct = new unsigned int[R.num()];
   // for(int d=0; d<D.ndata; ++d){
   //    int r = D.data[d].value;
   //    D2.data[((r==0)?0:rangecount[r-1])+rct[r]++] = D.data[d];
   // }

   // With parallel
   // for(int i=0;i<D.ndata;++i){
   //    if(D2.data[i].value==-1){
   //       std::cout<<"FAULT\n";
   //       // exit(0);
   //       break;
   //    }
   // }
   unsigned int *rct = new unsigned int[R.num()];
   #pragma omp parallel num_threads(numt)
   {
      int tid = omp_get_thread_num();
      int a = ((tid)*R.num())/numt;
      int b = ((tid+1)*R.num())/numt;
      for(int d=0; d<D.ndata; ++d){
         int r = D.data[d].value;
         if(a<=r && r<b){
            D2.data[((r==0)?0:rangecount[r-1])+rct[r]++] = D.data[d];
         }
      }
   }

   // D2.inspect();
   // D.inspect();
   return D2;
}
