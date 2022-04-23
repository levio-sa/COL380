#include <string>
#include <mpi.h>
#include <assert.h>
#include "randomizer.hpp"
#include <fstream>
#include <bits/stdc++.h>
#include <chrono>

using namespace std;

//Erase
// string ofnm;

inline void read_int(ifstream &infile, unsigned char buf[4], uint32_t &u){
    infile.read((char *)buf, sizeof(u));
    // infile.read((char *)buf, sizeof(buf[0]));
    // infile.read((char *)(buf+1), sizeof(buf[0]));
    // infile.read((char *)(buf+2), sizeof(buf[0]));
    // infile.read((char *)(buf+3), sizeof(buf[0]));
    u = (uint)buf[3] | (uint)buf[2]<<8 | (uint)buf[1]<<16 | (uint)buf[0]<<24;
}

void read_file(vector<vector<int>> &graph, std::string graph_file, int num_edges){
    ifstream infile(graph_file);
    unsigned char buf[4];
    uint32_t u = 0, v = 0;
    
    if(!infile.is_open()){
        std::cout<<"Error in opening file"<<std::endl;
        return;
    }
    
    for(int i=0; i<num_edges; ++i){
        read_int(infile, buf, u);
        read_int(infile, buf, v);
        graph[u].push_back(v);
    }
}

inline void create_file(int num_nodes, int num_rec){
    ofstream outfile("output.dat", ios::binary | ios::out);
    // ofstream outfile(ofnm, ios::binary | ios::out);
    // vector<char> nil(num_nodes*(2*num_rec+1)*4, 0); // Some efficient way
    if(!outfile.is_open()){
        cout<<"Error creating file"<<endl;
    }
    else{
        outfile.seekp(num_nodes*(2*num_rec+1)*4-1);
        outfile.write("",1);
    }
    // else{
    //     if(!outfile.write(&nil[0], nil.size())){
    //         cout<<"Error initialising file"<<endl;
    //     }
    // }
    outfile.close();
}

inline void write_int(ofstream &outfile, unsigned char buf[4], uint32_t u){
    // outfile<<u<<" ";
    // cout<<u<<"\n";
    uint32_t mask = 0xFF;
    buf[0] = (unsigned char) (u>>24 & mask);
    buf[1] = (unsigned char) (u>>16 & mask);
    buf[2] = (unsigned char) (u>>8 & mask);
    buf[3] = (unsigned char) (u & mask);

    outfile.write((char *)buf, sizeof(u));
    // outfile.write((char *)buf, sizeof(buf[0]));
    // outfile.write((char *)(buf+1), sizeof(buf[0]));
    // outfile.write((char *)(buf+2), sizeof(buf[0]));
    // outfile.write((char *)(buf+3), sizeof(buf[0]));
}

inline void write_null(ofstream &outfile, unsigned char buf[4]){
    // outfile<<"NULL ";
    buf[0] = 'N';
    buf[1] = 'U';
    buf[2] = 'L';
    buf[3] = 'L';

    outfile.write((char *)buf, 4*sizeof(buf[0]));

    // outfile.write((char *)buf, sizeof(buf[0]));
    // outfile.write((char *)(buf+1), sizeof(buf[0]));
    // outfile.write((char *)(buf+2), sizeof(buf[0]));
    // outfile.write((char *)(buf+3), sizeof(buf[0]));
}

void random_walk(int start, int end, int num_nodes, int num_steps, int num_walks, int num_rec, vector<vector<int>> &graph, Randomizer r){
    ofstream outfile("output.dat", ios::binary | ios::out | ios::in);
    // ofstream outfile(ofnm, ios::binary | ios::out | ios::in);
    
    if(!outfile.is_open()){
        cout<<"Unable to open file";
    }
    else{
        //Seek to the place where write needs to happen
        outfile.seekp(start*(2*num_rec+1)*4 + ios::beg, ios::beg);
        unsigned char buf_out[4];
        vector<int> circle(num_nodes,0);

        //Nodes processed by current processor
        for(int i=start; i<end; i++){
            //Outdegree
            write_int(outfile, buf_out, graph[i].size());
            //Container for influence scores

            //Iterating over L
            for(auto v: graph[i]){
                //num_walks walks
                for(int j=0; j<num_walks; j++){
                    //Current node
                    int w = v;
                    //num_steps steps
                    for(int k=0; k<num_steps; k++){
                        //Outdegree > 0
                        if(graph[w].size() > 0){
                            //Called only once in each step of random walk using the original node id 
                            //for which we are calculating the recommendations
                            int next_step = r.get_random_value(i);
                            //Random number indicates restart
                            if(next_step<0){
                                //Restart
                                w = v;
                            }
                            else{
                                //Deciding next step based on the output of randomizer which was already called
                                w = graph[w][next_step%graph[w].size()];
                            }
                        }
                        else{
                            //Restart
                            w = v;
                        }
                        //Increase influence
                        circle[w]+=1;
                    }
                }
            }

            //Erase i and outneighbors
            circle[i]=0;
            for(auto v: graph[i]){
                circle[v]=0;
            }

            //Sort according to influence score
            vector<pair<int,int>> vp;
            for(int pr=0; pr<num_nodes; ++pr){
                if(circle[pr]!=0){
                    vp.push_back({circle[pr],-pr});
                    circle[pr]=0;
                }
            }
            sort(vp.begin(),vp.end(),greater<pair<int,int>>());

            //Write to file
            for(int j=0; j<num_rec; ++j){
                if(j<vp.size()){
                    write_int(outfile, buf_out, -vp[j].second);
                    write_int(outfile, buf_out, vp[j].first);
                }
                else{
                    write_null(outfile, buf_out);
                    write_null(outfile, buf_out);
                }
            }
        }
    }
    outfile.close();
}

int main(int argc, char* argv[]){
    //Start Time
    // auto start_time = std::chrono::high_resolution_clock::now();
    assert(argc > 8);
    std::string graph_file = argv[1];
    int num_nodes = std::stoi(argv[2]);
    int num_edges = std::stoi(argv[3]);
    float restart_prob = std::stof(argv[4]);
    int num_steps = std::stoi(argv[5]);
    int num_walks = std::stoi(argv[6]);
    int num_rec = std::stoi(argv[7]);
    int seed = std::stoi(argv[8]);

    //Graph
    vector<vector<int>> graph(num_nodes);
    
    //Read by each rank
    read_file(graph, graph_file, num_edges);    
    
    //Only one randomizer object should be used per MPI rank, and all should have same seed
    Randomizer random_generator(seed, num_nodes, restart_prob);
    int rank, size;
    int data[1]={1};

    //Starting MPI pipeline
    MPI_Init(NULL, NULL);
    
    // Extracting Rank and Processor Count
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //Erase
    // ofnm = "output_"+to_string(size)+"_"+(string)argv[2]+"_"+(string)argv[3]+"_"+(string)argv[4]+"_"+(string)argv[5]+"_"+(string)argv[6]+"_"+(string)argv[7]+"_"+(string)argv[8]+".dat";

    MPI_Status status;

    //Create output.dat
    if(rank == 0){
        create_file(num_nodes, num_rec);
        for(int i=1; i<size; ++i)
            MPI_Send(data, 1, MPI_INT, i, 1, MPI_COMM_WORLD);
    }

    if(rank != 0){
        MPI_Recv(data, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
    }

    int start = (rank*num_nodes)/size;
    int end = ((rank+1)*num_nodes)/size;

    random_walk(start, end, num_nodes, num_steps, num_walks, num_rec, graph, random_generator);

    MPI_Finalize();

    // auto end_time = std::chrono::high_resolution_clock::now();
    // auto elapsed = chrono::duration_cast<chrono::milliseconds>(end_time - start_time);
    // if(rank == 0)
    //     cout<<"\n Elapsed "<<elapsed.count()<<"ms\n";
}