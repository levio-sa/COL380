#include "psort.h"
#include <omp.h>
#include<iostream>

// Sorting
void SequentialSort(uint32_t *data, uint32_t n);
void MergeSort(uint32_t *data, uint32_t n);
void InsertionSort(uint32_t *data, uint32_t n);

// Access control
inline int foo(int x){
    return x*64;
}

void ParallelSort(uint32_t *data, uint32_t n, int p)
{
    long long skip = n/p;

    // Threshold for SequentialSort
    if(skip < 2*p || p == 1) // Try changing p to 2p
        SequentialSort(data, n);
    
    else{

        int ind = 0;
        // Pseudo-splitters
        uint32_t *R = new uint32_t[p*p];
        long long bucket=0;
        for(int j=0; j<p; j++,bucket+=skip){
            for(int i=0; i<p; ++i){
                R[ind++] = data[bucket+i];
            }
        }

        // Sort pseudo-splitters
        SequentialSort(R, p*p);

        // Store Actual Splitters
        uint32_t *S = new uint32_t[p-1];
        for(int i=0; i<p; ++i){
            if(i!=p-1) S[i]=R[(i+1)*p];
        }

        uint64_t *C = new uint64_t[p*64];
        uint32_t **B = new uint32_t*[p];

        // Find size of each partition
        for(int part=0; part<p; ++part){
            #pragma omp task firstprivate(part)
            {   
                uint64_t ct = 0;
                if(part == 0){
                    for(uint64_t i=0; i<n; ++i){
                        if(data[i]<=S[part]) {
                            ct+=1;
                        }
                    }
                }
                else if(part == p-1){
                    for(uint64_t i=0; i<n; ++i){
                        if(data[i]>S[part-1]) {
                            ct+=1;
                        }
                    }
                }
                else{
                    for(uint64_t i=0; i<n; ++i){
                        if(data[i]>S[part-1] && data[i]<=S[part]) {
                            ct+=1;
                        }
                    }
                }                
                B[part] = new uint32_t[ct];
                C[foo(part)]=ct;
            }
        }
        #pragma omp taskwait

        // Prefix Sum
        for(int i=0; i<p; ++i){
            if(i!=0) C[foo(i)]+=C[foo(i-1)];
        }

        // Threshold
        uint32_t thresh = 2*(n/p);

        // Filling the partitions and sorting them
        for(int part=0; part<p; ++part){
            #pragma omp task firstprivate(part)
            {   
                uint64_t ct = 0;
                if(part == 0){
                    for(uint64_t i=0; i<n; ++i){
                        if(data[i]<=S[part]) {
                            B[part][ct]=data[i];
                            ct+=1;
                        }
                    }
                }
                else if(part == p-1){
                    for(uint64_t i=0; i<n; ++i){
                        if(data[i]>S[part-1]) {
                            B[part][ct]=data[i];
                            ct+=1;
                        }
                    }
                }
                else{
                    for(uint64_t i=0; i<n; ++i){
                        if(data[i]>S[part-1] && data[i]<=S[part]) {
                            B[part][ct]=data[i];
                            ct+=1;
                        }
                    }
                }
                
                if(ct<thresh)
                    SequentialSort(B[part], ct);
                else
                    ParallelSort(B[part], ct, p);
            }
        }
        #pragma omp taskwait
        
        // Filling data array
        #pragma omp task
        {
            for(uint64_t i=0;i<C[foo(0)];++i){
                data[i]=B[0][i];
            }
        }
        
        for(int part=1; part<p; ++part){
            #pragma omp task firstprivate(part)
            {
                for(uint64_t i=C[foo(part-1)];i<C[foo(part)];++i){
                    data[i]=B[part][i-C[foo(part-1)]];
                }
            }
        }
        #pragma omp taskwait

        delete[] R;
        delete[] S;
        for(int i=0;i<p;++i) delete[] B[i];
        delete[] B;
        delete[] C;

    }    
}

// Mix of Insertion and Merge Sort
inline void SequentialSort(uint32_t *data, uint32_t n){
    if(n>50)
    MergeSort(data, n);
    else InsertionSort(data, n);
}

void MergeSort(uint32_t *data, uint32_t n){
    if(n == 1) return;
    if(n == 2){
        if(data[0]>data[1]){
            uint32_t tmp = data[0];
            data[0] = data[1];
            data[1] = tmp;
        }
        return;
    }
    long long a = n/2;
    long long b = n-n/2;
    uint32_t *d1 = new uint32_t[a];
    uint32_t *d2 = new uint32_t[b];
    for(uint32_t i=0; i<n/2; ++i){
        d1[i]=data[i];
    }
    for(uint32_t i=n/2; i<n; ++i){
        d2[i-n/2]=data[i];
    }
    
    SequentialSort(d1, n/2);
    SequentialSort(d2, n-n/2);
    
    
    long long i=0;
    long long j=0;
    while(i<a && j<b){
        if(d1[i]<d2[j]){
            data[i+j]=d1[i];
            i++;
        }
        else{
            data[i+j]=d2[j];
            j++;
        }
    }
    while(i<a){
        data[i+j]=d1[i];
        i++;
    }
    while(j<b){
        data[i+j]=d2[j];
        j++;
    }

    delete[] d1;
    delete[] d2;
}

void InsertionSort(uint32_t *data, uint32_t n){
    uint64_t i, j, tmp;
    int x=0;
    for(i = 1; i < n; ++i){
        j = i;
        tmp = data[i];

        while(j>=1 && tmp < data[j-1]){
            data[j]=data[j-1];
            j--;
            x+=1;
        }
        data[j]=tmp;
    }
}

