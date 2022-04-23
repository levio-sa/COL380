#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;


pair<int,int> read_img(string img_file, int *&img){
    ifstream ifs(img_file);

    while(ifs.is_open()){
        int x,y,tmp;
        ifs >> x >> y;

        cudaMallocManaged(&img, x*y*3*sizeof(int));

        for(int i = 0; i < x; ++i){
            for(int j = 0; j < y; ++j){
                for(int k = 0; k < 3; ++k){
                    ifs>>tmp;
                    // cout<<tmp<<"\n";
                    img[(x-1-i)*y*3 +j*3 +k] = tmp;
                    // cout<<img[(x-1-i)*y*3 +j*3 +k];
                }
            }
        }
        // cout<<"HELLO"<<endl;
        ifs.close();
        return {x,y};
    }

}

__device__ float square_root(float x){
    return sqrt(x);
}

__device__ void bin_inter(int* &data_img, int n, int m, float x, float y, float* res){
    int a = x/1;
    int b = y/1;
    x-=a;
    y-=b;
    res = new float[3];

    res[0] = data_img[a*m*3+b*3+0]*(1-x)*(1-y)+data_img[(a+1)*m*3+b*3+0]*(x)*(1-y)+data_img[a*m*3+(b+1)*3+0]*(1-x)*(y)+data_img[(a+1)*m*3+(b+1)*3+0]*(x)*(y);
    res[1] = data_img[a*m*3+b*3+1]*(1-x)*(1-y)+data_img[(a+1)*m*3+b*3+1]*(x)*(1-y)+data_img[a*m*3+(b+1)*3+1]*(1-x)*(y)+data_img[(a+1)*m*3+(b+1)*3+1]*(x)*(y);
    res[2] = data_img[a*m*3+b*3+2]*(1-x)*(1-y)+data_img[(a+1)*m*3+b*3+2]*(x)*(1-y)+data_img[a*m*3+(b+1)*3+2]*(1-x)*(y)+data_img[(a+1)*m*3+(b+1)*3+2]*(x)*(y);

    // return res;
}


__global__ void main_func(int *data_img, int n, int m, int *q_img, int a, int b, float* rmsd, float q_avg, float th1, float th2){

    // int index = blockIdx.x*blockDim.x + threadIdx.x;
    // int i = index/m;
    // int j = index%m;
    int i = blockIdx.x;
    int j = blockIdx.y;

    // printf("helo\n");

    float avg1 = 0;
    float avg2 = 0;
    float avg3 = 0;

    if(i==840 && j==900){
        printf("helo\n");
    }

    // 0
    if(i+a<n && j+b<m){
        avg1 = 0;
        for(int x=i; x<i+a; ++x){
            for(int y=j; y<j+b; ++y){
                avg1 += ((float)(data_img[x*m*3+y*3+0]+data_img[x*m*3+y*3+1]+data_img[x*m*3+y*3+2]));
            }
        }
        avg1 = avg1/((float)(3*a*b));
        if(i==840 && j==900){
            printf("%f\n",abs(q_avg-avg1));
        }
        // printf("%f\n",abs(q_avg-avg1));
        if(abs(q_avg-avg1)<th2){
            if(i==840 && j==900){
                printf("%f\n",abs(q_avg-avg1));
            }
            // printf("%f\n",abs(q_avg-avg1));
            float rms1 = 0;
            for(int x=i;x<i+a;++x){
                for(int y=j;y<j+b;++y){
                    for(int z=0;z<3;++z){
                        float t = data_img[x*m*3+y*3+z]-q_img[(x-i)*b*3+(y-j)*3+z];
                        // rms1+=pow((data_img[x*m*3+y*3+z]-q_img[(x-i)*b*3+(y-j)*3+z]),2);
                        rms1+=t*t;
                    }
                }
            }
            rms1 = rms1/((float)a*b*3);
            rms1 = square_root(rms1);
            if(i==840 && j==900){
                printf("%f\n",abs(rms1));
            }
            if(rms1<th1){
                // printf("%f\n",rms1);
                rmsd[i*m*3+j*3] = (rms1);
                // printf("%f\n",rms1);
            }
            else{
                // avg1 = -1;
                rmsd[i*m*3+j*3] = -1;
            }
        } 
        else{
            // avg1 = -1;
            rmsd[i*m*3+j*3] = -1;
        }       
    }
    else{
        // avg1 = -1;
        rmsd[i*m*3+j*3] = -1;
    }

    // +45
    float xoff = (float)a/(float)square_root(2);
    float yoff = (float)b/(float)square_root(2);
    int x1 = i;
    int x2 = (float)i + ((float)(a+b))/((float)square_root(2)) + (float)1;
    int y1 = (float)j - (float)a/(float)square_root(2);
    int y2 = (float)j + (float)b/(float)square_root(2) + (float)1;
    if(y1>=0 && y2<m && x2<n){
        avg2 = 0;
        for(int x=x1;x<x2;++x){
            for(int y=y1;y<y2;++y){
                avg2 += ((float)(data_img[x*m*3+y*3+0]+data_img[x*m*3+y*3+1]+data_img[x*m*3+y*3+2]))/((float)3);
            }
        }
        avg2 = avg2/(float)(4*xoff*yoff);
        if(abs(q_avg-avg2)<th2){
            float rms2 = 0.0;
            for(int x=0;x<a;++x){
                for(int y=0;y<b;++y){
                    float p1,p2;
                    p2 = (float)j+((float)x/(float)square_root(2))-((float)y/(float)square_root(2));
                    p1 = (float)i+((float)x/(float)square_root(2))+((float)y/(float)square_root(2));
                    float* temp1;
                    bin_inter(data_img, n,m,p1,p2, temp1);
                    for(int z=0;z<3;++z){
                        rms2+=pow((temp1[z]-q_img[x*b*3+y*3+z]),2);
                    }
                }
            }
            rms2 = rms2/((float)a*b*3);
            rms2 = square_root(rms2);
            if(rms2<th1){
                rmsd[i*m*3+j*3+1] = (rms2);
                // printf("%f\n",rms2);
            }
            else{
                // avg1 = -1;
                rmsd[i*m*3+j*3+1] = -1;
            }
        } 
        else{
            // avg[i*m*3+j*3+1] = -1;
            rmsd[i*m*3+j*3+1] = -1;
        }
    }
    else{
        // avg[i*m*3+j*3+1] = -1;
        rmsd[i*m*3+j*3+1] = -1;
    }

    // -45
    x1 = (float)i - (float)b/(float)square_root(2);
    x2 = (float)i + (float)a/(float)square_root(2) + (float)1;
    y1 = j;
    y2 = (float)j + ((float)(a+b))/((float)square_root(2)) + (float)1;
    if(x1>=0 && y2<m && x2<n){
        avg3 = 0;
        for(int x=x1;x<x2;++x){
            for(int y=y1;y<y2;++y){
                avg3 += ((float)(data_img[x*m*3+y*3+0]+data_img[x*m*3+y*3+1]+data_img[x*m*3+y*3+2]))/((float)3);
            }
        }
        avg3 = avg3/(float)(4*xoff*yoff);
        if(abs(q_avg-avg3)<th2){
            float rms3 = 0.0;
            for(int x=0;x<a;++x){
                for(int y=0;y<b;++y){
                    float p1,p2;
                    p2 = (float)j+((float)x/(float)square_root(2))+((float)y/(float)square_root(2));
                    p1 = (float)i+((float)x/(float)square_root(2))-((float)y/(float)square_root(2));
                    float* temp1;
                    bin_inter(data_img, n,m,p1,p2, temp1);
                    for(int z=0;z<3;++z){
                        rms3+=pow((temp1[z]-q_img[x*b*3+y*3+z]),2);
                    }
                }
            }
            rms3 = rms3/((float)a*b*3);
            rms3 = square_root(rms3);
            if(rms3<th1){
                rmsd[i*m*3+j*3+2] = (rms3);
            }
            else{
                // avg1 = -1;
                rmsd[i*m*3+j*3+2] = -1;
            }
        } 
        else{
            // avg[i*m*3+j*3+1] = -1;
            rmsd[i*m*3+j*3+2] = -1;
        }
    }
    else{
        // avg[i*m*3+j*3+1] = -1;
        rmsd[i*m*3+j*3+2] = -1;
    }

    // printf("%f\n",rmsd[i*m*3+j*3+0]);
    // printf("%f\n",rmsd[i*m*3+j*3+1]);
    // printf("%f\n",rmsd[i*m*3+j*3+2]);

}


int main(int argc, char* argv[]){

    if(argc < 6){
        cout << "Invalid arguments"<<endl;
        exit(-1);
    }

    string data_img_file = argv[1];
    string q_img_file = argv[2];
    float th1 = stof(argv[3]);
    float th2 = stof(argv[4]);
    int n = stoi(argv[5]);

    
    int *data_img, *q_img;
    int r1,r2,r3,r4;

    pair<int,int> dim1, dim2;
    
    ifstream ifs(data_img_file);

    while(ifs.is_open()){
        int tmp;
        ifs >> r1 >> r2;

        cudaMallocManaged(&data_img, r1*r2*3*sizeof(int));

        for(int i = 0; i < r1; ++i){
            for(int j = 0; j < r2; ++j){
                for(int k = 0; k < 3; ++k){
                    ifs>>tmp;
                    // cout<<tmp<<"\n";
                    data_img[(r1-1-i)*r2*3 +j*3 +k] = tmp;
                    // cout<<img[(x-1-i)*y*3 +j*3 +k];
                }
            }
        }
        // cout<<"HELLO"<<endl;
        ifs.close();
    }
    dim1 = {r1,r2};

    ifstream ifs1(q_img_file);

    while(ifs1.is_open()){
        int tmp;
        ifs1 >> r3 >> r4;

        cudaMallocManaged(&q_img, r3*r4*3*sizeof(int));

        for(int i = 0; i < r3; ++i){
            for(int j = 0; j < r4; ++j){
                for(int k = 0; k < 3; ++k){
                    ifs1>>tmp;
                    // cout<<tmp<<"\n";
                    q_img[(r3-1-i)*r4*3 +j*3 +k] = tmp;
                    // cout<<img[(x-1-i)*y*3 +j*3 +k];
                }
            }
        }
        // cout<<"HELLO"<<endl;
        ifs1.close();
    }
    dim2 = {r3,r4};
    // dim1 = read_img(data_img_file, data_img);
    // dim2 = read_img(q_img_file, q_img);

    float q_avg = 0;
    for(int x=0;x<dim2.first;++x){
        for(int y=0;y<dim2.second;++y){
            q_avg+=((float)(q_img[x*dim2.second*3+y*3+0]+q_img[x*dim2.second*3+y*3+1]+q_img[x*dim2.second*3+y*3+2]));
        }
    }

    q_avg = q_avg/((float)(3*dim2.first*dim2.second));
    cout<<"q_avg "<<q_avg<<endl;

    // float t_avg = 0;
    // for(int x=840;x<840+dim2.first;++x){
    //     for(int y=900;y<900+dim2.second;++y){
    //         t_avg+=((float)(data_img[x*dim1.second*3+y*3+0]+data_img[x*dim1.second*3+y*3+1]+data_img[x*dim1.second*3+y*3+2]));
    //     }
    // }

    // t_avg = t_avg/((float)(3*dim2.first*dim2.second));
    // cout<<"t_avg "<<t_avg<<endl;

    float* rmsd;
    // rmsd = new float[dim1.first*dim1.second*3];
    cudaMallocManaged(&rmsd, dim1.first*dim1.second*3*sizeof(float));
    for(int i=0;i<dim1.first*dim1.second*3;++i){
        rmsd[i]=-1;
    }

    int m = dim1.second;
    int n1 = dim1.first;
    int b = dim2.second;
    int a = dim2.first;
    // cout<<n1<<" "<<m<<" "<<a<<" "<<b<<"\n";
    dim3 gd(n1,m);
    main_func<<<gd, 1>>>(data_img, n1, m, q_img, a, b, rmsd, q_avg, th1, th2);

    cudaDeviceSynchronize();

    priority_queue<pair<float, vector<int>>> pq;
    int c=0;
    for(int i=0;i<dim1.first;++i){
        for(int j=0;j<dim1.second;++j){
            for(int k=0;k<3;++k){
                float tmp = rmsd[i*m*3+j*3+k];
                if(tmp!=-1){
                    pq.push({-tmp,{i,j,k}});
                    c++;
                    // cout<<i<<" "<<j<<" "<<k<<" "<<c<<"\n";
                }
            }
        }
    }

    cout<<c<<"\n";

    ofstream ofs("output.txt");

    if(ofs.is_open()){
        for(int i=0; i<min(n,c);++i){
            auto tmp = pq.top();
            pq.pop();
            // cout<<-tmp.first<<"\n";
            ofs<<tmp.second[0]<<" "<<tmp.second[1]<<" ";
            if(tmp.second[2]==0){
                ofs<<0<<"\n";
            } 
            else if(tmp.second[2]==1){
                ofs<<45<<"\n";
            }
            else{
                ofs<<-45<<"\n";
            }
        }
        ofs.close();
    }

    cudaFree(data_img);
    cudaFree(q_img);
    cudaFree(rmsd);

    return 0;
}