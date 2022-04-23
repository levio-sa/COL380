#include <string>
#include <mpi.h>
#include <omp.h>
#include <fstream>
#include <bits/stdc++.h>
#include <chrono>

using namespace std;

#define pq priority_queue <pair<int, float>, vector<pair<int,float>>, compare >

class compare
{
public:
    bool operator() (const pair<int, float> &a, const pair<int, float> &b){
        return a.second<b.second;
    }
};

#define pq priority_queue <pair<int, float>, vector<pair<int,float>>, compare >

float cosine_dist(int user, vector<float> &a, int ep, int D, vector<float> &b){
    float num=0.0, denom_1=0.0, denom_2=0.0;
    int user_offset = user*D, ep_offset = ep*D;
    for(int i=0; i<D; i++){
        num+=a[i+user_offset]*b[i+ep_offset];
        denom_1+=a[i+user_offset]*a[i+user_offset];
        denom_2+=b[i+ep_offset]*b[i+ep_offset];
    }
    return 1.0-(num/sqrt(denom_1)/sqrt(denom_2));
}

pq SearchLayer(int user, vector<float> &q, int topk, pq &candidates, vector<int> &indptr, vector<int> &index, vector<int> &level_offset, int lc, unordered_set<int> &visited, int D, vector<float> &vect){
    pq top_k = candidates;
    while(candidates.size()>0){
        int ep = candidates.top().first;
        //cout<<"ep"<<" "<<ep<<endl;
        candidates.pop();
        int start = indptr[ep] + level_offset[lc];
        int end = indptr[ep] + level_offset[lc+1];
        //cout<<"level "<<lc<<endl;
        for(int i=start; i<end; i++){
            int px = index[i];
            //cout<<"child "<<ep<<" "<<px<<endl;
            if(visited.find(px)!=visited.end() || px==-1){
                continue;
            }
            visited.insert(px);
            float _dist = cosine_dist(user, q, px, D, vect);
            //cout<<"distance"<<px<<" "<<_dist<<endl;
            float top_k_max = top_k.top().second;
            if (_dist>top_k_max && top_k.size()>=topk){
                continue;
            }
            top_k.push({px, _dist});
            while(top_k.size()>topk){
                top_k.pop();
            }
            candidates.push({px, _dist});
        }
    }
    return top_k;
}

pq QueryHNSW(int user, vector<float> &q, int topk, int &ep, vector<int> &indptr, vector<int> &index, vector<int> &level_offset, int &max_level, int D, vector<float> &vect){
    pq top_k;
    top_k.push({ep, cosine_dist(user, q, ep, D, vect)});
    //cout<<"HELLO "<<ep<<" "<<cosine_dist(q, vect[ep])<<endl;
    unordered_set<int> visited;
    visited.insert(ep);
    int L = max_level;
    for(int level=L; level>=0; level--){
        top_k = SearchLayer(user, q, topk, top_k, indptr, index, level_offset, level, visited, D, vect);
    }
    //cout<<"ANSWER "<<top_k.top().first<<endl;
    return top_k;
}

void read_int(int &u, string infile){
    //ifstream ifs(infile, ios::binary | ios::in);
    // ifs.read((char*)u,sizeof(u));
    // ifs.close();
    FILE * fd = fopen(infile.c_str(), "rb");
    fread(&u, 1, sizeof(u), fd);
    fclose(fd);
}

void read_vector(vector<int> &vec, string infile){
    FILE * fd = fopen(infile.c_str(), "rb");
    int temp;
    while(fread(&temp, 1, sizeof(temp), fd)==4){
        vec.push_back(temp);
    }
    fclose(fd);
}

void MPI_Parallel_Read(vector<float > &vect, string user_file, int N, int D, int rank, int size){
    //int start_embedding = N/size*rank, end_embedding = (rank!=size-1)? N/size*(rank+1): N, num_embedding_rank = end_embedding-start_embedding;
    FILE *fi = fopen(user_file.c_str(), "rb");
    //fseek(fi, (sizeof(float))*(D)*N, SEEK_SET);
    fread(&vect[0], N*D, sizeof(float), fi);
    fclose(fi);
    // vector<int> recv_size(size), recv_stride(size);
    // for(int i=0; i<size; i++){
    //     recv_size[i]= D*(((i!=size-1)? N/size*(i+1): N) - (N/size*i));
    //     recv_stride[i]= D*(N/size*i);
    // }

    // MPI_Allgatherv(MPI_IN_PLACE, recv_size[rank], MPI_FLOAT, vect.data(), recv_size.data(), recv_stride.data(), MPI_FLOAT, MPI_COMM_WORLD);
}

void MPI_Parallel_Read_Int(vector<int> &index, string user_file, int N, int rank, int size){
    //int start_embedding = N/size*rank, end_embedding = (rank!=size-1)? N/size*(rank+1): N, num_embedding_rank = end_embedding-start_embedding;
    FILE *fi = fopen(user_file.c_str(), "rb");
    //fseek(fi, (sizeof(int))*N, SEEK_SET);
    fread(&index[0], (sizeof(int))*N, sizeof(int), fi);
    fclose(fi);
    // vector<int> recv_size(size), recv_stride(size);
    // for(int i=0; i<size; i++){
    //     recv_size[i]= (((i!=size-1)? N/size*(i+1): N) - (N/size*i));
    //     recv_stride[i]= (N/size*i);
    // }

    // MPI_Allgatherv(MPI_IN_PLACE, recv_size[rank], MPI_FLOAT, index.data(), recv_size.data(), recv_stride.data(), MPI_FLOAT, MPI_COMM_WORLD);

}

void MPI_Parallel_Read_float(vector<float> &user_vect, string text_file, int &num_users, int D, int rank, int size){
    MPI_File temp_file;
    MPI_File_open(MPI_COMM_WORLD, text_file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &temp_file);
    MPI_Offset size_total;
    MPI_File_get_size(temp_file, &size_total);
    MPI_Offset start_ptr = (size_total-1)/size*rank, end_ptr = (rank!=size-1)? ((size_total-1)/size*(rank+1))+12: size_total-1,  size_ptr = end_ptr - start_ptr;
    vector<char> part_read(size_ptr);
    MPI_File_read_at_all(temp_file, start_ptr, part_read.data(), size_ptr, MPI_CHAR, MPI_STATUS_IGNORE);
    MPI_File_close(&temp_file);
    MPI_Offset start = start_ptr;
    if(rank-size+1){
        end_ptr-=12;
        while(part_read[end_ptr-start]!=' ' && part_read[end_ptr-start]!='\n'){
            end_ptr++;
        }
    }
    if(rank){
        do{
            start_ptr++;
        }while(part_read[start_ptr-1-start]!=' ' && part_read[start_ptr-1-start]!='\n');
           
    }

    vector<char> num;
    for(MPI_Offset i=start_ptr; i<end_ptr; i++){
        //if(rank==1) //cout<<part_read[i-start]<<endl;
        if(part_read[i-start]==' ' || part_read[i-start]=='\n'){
            user_vect.emplace_back(stof(string(num.begin(), num.end())));
            num.clear();
        }else{
            num.emplace_back(part_read[i-start]);
        }
    }
    if(num.size()){
         user_vect.emplace_back(stof(string(num.begin(), num.end())));
    }
    vector<long long int> count_floats(size);
    vector<int> recv_size(size), recv_stride(size);
    for(int i=0; i<size; i++){
        recv_size[i]= 1;
        recv_stride[i]= i;
    }
    count_floats[rank] = user_vect.size();
    MPI_Allgatherv(MPI_IN_PLACE, recv_size[rank], MPI_LONG_LONG_INT, count_floats.data(), recv_size.data(), recv_stride.data(), MPI_LONG_LONG_INT, MPI_COMM_WORLD);
    
    recv_size[0] = count_floats[0];
    recv_stride[0] = 0;
    for (int i=1; i<size; i++){
        recv_size[i] = count_floats[i];
        recv_stride[i] = recv_stride[i-1] + count_floats[i-1];
    }
    
    user_vect.resize(recv_stride.back()+count_floats.back());

    num_users = (long long int)(user_vect.size())/D;

    for(int i=count_floats[rank]-1; i>=0; i--){
        user_vect[i+recv_stride[rank]]=user_vect[i];
    }
    
    MPI_Allgatherv(MPI_IN_PLACE, recv_size[rank], MPI_FLOAT, user_vect.data(), recv_size.data(), recv_stride.data(), MPI_FLOAT, MPI_COMM_WORLD);
    
}

void MPI_Parallel_Write_int(FILE *fo, vector<vector<int>> &top_u_k, int rank, int size){
    vector<string> buffer(top_u_k.size());
    vector<long long int> buffer_size(top_u_k.size());
    #pragma omp parallel for
    for (int i = 0; i < top_u_k.size(); ++i){
        string temp = "";
        int j=0;
        for(; j<top_u_k[i].size()-1; j++){
            temp+=(to_string(top_u_k[i][j])+" ");
        }
        temp+=(to_string(top_u_k[i][j])+"\n");
        buffer[i] = temp;
        buffer_size[i] = temp.length();
    }

    for(int i=1; i<top_u_k.size(); i++){
        buffer_size[i]+=buffer_size[i-1];
    }

    vector<long long int> size_string(size);
    vector<int> recv_size(size), recv_stride(size);
    for(int i=0; i<size; i++){
        recv_size[i]= 1;
        recv_stride[i]= i;
    }

    size_string[rank] = buffer_size.back();

    MPI_Allgatherv(MPI_IN_PLACE, recv_size[rank], MPI_LONG_LONG_INT, size_string.data(), recv_size.data(), recv_stride.data(), MPI_LONG_LONG_INT, MPI_COMM_WORLD);
    
    
    for(int i=1; i<size; i++){
        size_string[i]+=size_string[i-1];
    }

    fseek(fo, ((rank)? size_string[rank-1]: 0), SEEK_SET);

    for (int i = 0; i < top_u_k.size(); ++i){
        fprintf(fo, "%s", buffer[i].c_str());
    }
    fclose(fo);
}


int main(int argc, char* argv[]){
    auto begin = std::chrono::high_resolution_clock::now();

    if(argc < 4){
        //cout << "Invalid arguments"<<endl;
        exit(-1);
    }
    string out_path = argv[1];
    int topk = stoi(argv[2]);
    string user_file = argv[3];
    string user_output_file = argv[4];

    //cout<< "Starting to read files"<<endl;
    
    int max_level;
    read_int(max_level, out_path+"/max_level.bin");
    //cout<<"Read max_level"<<endl;
    int ep;
    read_int(ep, out_path+"/ep.bin");
    //cout<<"Read ep"<<endl;
    //cout<<max_level<<" "<<ep<<endl;
    int L;
    read_int(L, out_path+"/L.bin");
    int D;
    read_int(D, out_path+"/D.bin");
    int I;
    read_int(I, out_path+"/I.bin");
    vector<int> level;
    read_vector(level, out_path+"/level.bin");
    //cout<<"level"<<endl;
    // vector<int> index;
    // read_vector(index, out_path+"/index.bin");
    //cout<<"index"<<endl;
    vector<int> indptr;
    read_vector(indptr, out_path+"/indptr.bin");
    //cout<<"indptr"<<endl;
    vector<int> level_offset;
    read_vector(level_offset, out_path+"/level_offset.bin");
    //cout<<"lo"<<endl;
    //vector<vector<float>> vect;
    //read_embd(D, vect, out_path+"/vect.bin");    
    //cout<<"re"<<endl;
    //vector<vector<float>> user_vect;
    int num_users;
    //read_embd_t(num_users, user_vect, user_file); 
    //cout<<"ret"<<endl;

    // for(int i=0; i<vect.size(); i++){
    //     for(int j=0; j<D; j++){
    //         //cout<<vect[i][j]<<" ";
    //     }
    //     //cout<<endl;
    // }

    // for(int i=0; i<num_users; i++){
    //     for(int j=0; j<D; j++){
    //         //cout<<user_vect[i][j]<<" ";
    //     }
    //     //cout<<endl;
    // }

    FILE* fo = fopen(user_output_file.c_str(), "w");

    int rank, size;

    //Starting MPI pipeline
    MPI_Init(NULL, NULL);
    
    // Extracting Rank and Processor Count
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    vector<int> index(I);
    MPI_Parallel_Read_Int(index, out_path+"/index.bin", I, rank, size);

    //vector<vector<float> > vect(L, vector<float>(D));
    vector<float> vect(L*D);
    MPI_Parallel_Read(vect, out_path+"/vect.bin", L, D, rank, size);

    vector<float > user_vect;
    MPI_Parallel_Read_float(user_vect, user_file, num_users, D, rank, size);
//     if(rank){
//     for(int i=0; i<L; i++){
//         for(int j=0; j<D; j++){
//             cout<<vect[i*D+j]<<" ";
//         }
//         cout<<endl;
//     }
// }

    //cout<<num_users<<endl;
    int start_user = num_users/size*rank, end_user = (rank!=size-1)? num_users/size*(rank+1): num_users, num_users_rank = end_user-start_user;
    vector<vector<int> > top_u_k(num_users_rank, vector<int>(topk));

        //cout<<"HELLO "<<start_user<<" "<<end_user<<endl;
    #pragma omp parallel for
    for(int user=start_user; user<end_user; user++){
        //cout<<"START "<<user<<endl;
        #pragma omp task shared(user_vect, topk, ep, indptr, index, level_offset, max_level, D, vect) firstprivate(user)
        {
            pq top_k = QueryHNSW(user, user_vect, topk, ep, indptr, index, level_offset, max_level, D, vect);
            int size = topk-1;
            while(top_k.size()){
                top_u_k[user-start_user][size--]=(top_k.top().first);
    			top_k.pop();
    		}
        }
    }
    #pragma omp taskwait

    MPI_Parallel_Write_int(fo, top_u_k, rank, size);

    // MPI_Barrier(MPI_COMM_WORLD);
    // int buf[1] = {1};
    // MPI_Status status;
    // if(size == 1 ){
    //     for(int i= 0; i<num_users_rank; i++){
    //         for(int j=0; j<topk; j++){
    //             cout<<top_u_k[i][j]<<" ";
    //         }
    //         cout<<endl;
    //         buf[0]=1;
    //     }
    // }
    // else if(rank == 0){
    //     for(int i= 0; i<num_users_rank; i++){
    //         for(int j=0; j<topk; j++){
    //             cout<<top_u_k[i][j]<<" ";
    //         }
    //         cout<<endl;
    //         buf[0]=1;
    //     }
    //     MPI_Send(buf, 1, MPI_INT, 1, 99, MPI_COMM_WORLD);
    // }
    // else if(rank == size-1){
    //     MPI_Recv(buf, 1, MPI_INT, rank-1, 99, MPI_COMM_WORLD, &status);
    //     for(int i= 0; i<num_users_rank; i++){
    //         for(int j=0; j<topk; j++){
    //             cout<<top_u_k[i][j]<<" ";
    //         }
    //         cout<<endl;
    //         buf[0]=1;
    //     }
    // }
    // else{
    //     MPI_Recv(buf, 1, MPI_INT, rank-1, 99, MPI_COMM_WORLD, &status);
    //     for(int i= 0; i<num_users_rank; i++){
    //         for(int j=0; j<topk; j++){
    //             cout<<top_u_k[i][j]<<" ";
    //         }
    //         cout<<endl;
    //         buf[0]=1;
    //     }
    //     MPI_Send(buf, 1, MPI_INT, rank+1, 99, MPI_COMM_WORLD);
    // }

    // FILE *fo = fopen(user_output_file.c_str(), "wb");
    // fseek(fo, 4*(topk+1)*start_user, SEEK_SET);
    // for(int i=0; i<num_users_rank; ++i){
    //     for(int j=0; j<topk; j++){
    //         fprintf(fo, "%d ", top_u_k[i][j]);
    //     }
    //     fputc('\n', fo);
    // }
    // fclose(fo);
    
    MPI_Finalize();
    auto end = std::chrono::high_resolution_clock::now();
    if(rank==0){
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        float duration = (1e-6 * (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)).count());
        cout << "Time taken " << duration << "ms" << endl;
    }
}