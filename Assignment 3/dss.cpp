#include <string>
#include <mpi.h>
#include <assert.h>
#include <chrono>
#include <bits/stdc++.h>
#include <experimental/filesystem> // or #include <filesystem> for C++17 and up
    
using namespace std;
namespace fs = std::experimental::filesystem;

void read_int(int &num, string infile){
    ifstream ifs(infile);
    if(ifs.is_open()){
        ifs >> num;
        ifs.close();
    }
    else{
        //cout<<"Error in opening "<<infile<<endl;
        exit(-1);
    }
}

void read_vector(vector<int> &vec, string infile){
    ifstream ifs(infile);
    if(ifs.is_open()){
        int tmp;
        while(ifs >> tmp){
            // //cout<<tmp<<endl;
            vec.push_back(tmp);
            if(ifs.eof()) break;
        }
        ifs.close();
    }
    else{
        //cout<<"Error in opening "<< infile <<endl;
        exit(-1);
    }
}

inline void write_int(int num, string outfile){
    ofstream ofs(outfile, ios::binary | ios::out);
    unsigned char buf[4];
    if(ofs.is_open()){
        ofs.write((char *)&num, sizeof(num));
        ofs.close();
    }
    else{
        //cout<<"Unable to open "<<outfile<<endl;
    }
}

inline void write_vector(vector<int> &vec, string outfile){
    ofstream ofs(outfile, ios::binary | ios::out);
    unsigned char buf[4];
    for(int i=0; i<vec.size(); ++i){
        ofs.write((char *)&vec[i], sizeof(vec[i]));
    }
    ofs.close();
}

void MPI_Parallel_Write_int(string text_file, FILE* fo, int rank, int size, int &I){
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

    vector<int> vect_temp;
    vector<char> num;
    for(MPI_Offset i=start_ptr; i<end_ptr; i++){
        if(part_read[i-start]==' ' || part_read[i-start]=='\n'){
            vect_temp.emplace_back(stoi(string(num.begin(), num.end())));
            num.clear();
        }else{
            num.emplace_back(part_read[i-start]);
        }
    }
    if(num.size()){
         vect_temp.emplace_back(stoi(string(num.begin(), num.end())));
    }
    vector<long long int> count_floats(size);
    vector<int> recv_size(size), recv_stride(size);
    for(int i=0; i<size; i++){
        recv_size[i]= 1;
        recv_stride[i]= i;
    }
    count_floats[rank] = vect_temp.size();
    MPI_Allgatherv(MPI_IN_PLACE, recv_size[rank], MPI_LONG_LONG_INT, count_floats.data(), recv_size.data(), recv_stride.data(), MPI_LONG_LONG_INT, MPI_COMM_WORLD);

    for (int i=0; i<size-1; i++){
        count_floats[i+1]+=count_floats[i];
    }
    if(!rank){
        I = count_floats.back();
    }
    fseek(fo, (sizeof(int))*((rank)? count_floats[rank-1]: 0), SEEK_SET);
    fwrite(vect_temp.data(), vect_temp.size(), sizeof(int), fo);
    fclose(fo);
}

void MPI_Parallel_Write_double(string text_file, FILE* fo, int rank, int size, int&L, int&D){
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
        //cout<<"BII1"<<endl;
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

    vector<float> vect_temp;
    vector<char> num;

    if(!rank){
        bool found = false;
        for(MPI_Offset i=start_ptr; i<end_ptr && !found; i++){
            //if(rank==1) cout<<part_read[i-start]<<endl;
            if(part_read[i-start]==' ' || part_read[i-start]=='\n'){
                vect_temp.emplace_back(stof(string(num.begin(), num.end())));
                if(part_read[i-start]=='\n'){
                    D = vect_temp.size();
                    found=true;
                    continue;
                }
                num.clear();
            }else{
                num.emplace_back(part_read[i-start]);
            }
        }

        vect_temp.clear();
        num.clear();
    }

    for(MPI_Offset i=start_ptr; i<end_ptr; i++){
        //if(rank==1) cout<<part_read[i-start]<<endl;
        if(part_read[i-start]==' ' || part_read[i-start]=='\n'){
            vect_temp.emplace_back(stof(string(num.begin(), num.end())));
            num.clear();
        }else{
            num.emplace_back(part_read[i-start]);
        }
    }
    if(num.size()){
         vect_temp.emplace_back(stof(string(num.begin(), num.end())));
    }
    vector<long long int> count_floats(size);
    vector<int> recv_size(size), recv_stride(size);
    for(int i=0; i<size; i++){
        recv_size[i]= 1;
        recv_stride[i]= i;
    }
    count_floats[rank] = vect_temp.size();
    MPI_Allgatherv(MPI_IN_PLACE, recv_size[rank], MPI_LONG_LONG_INT, count_floats.data(), recv_size.data(), recv_stride.data(), MPI_LONG_LONG_INT, MPI_COMM_WORLD);

    for (int i=0; i<size-1; i++){
        count_floats[i+1]+=count_floats[i];
    }
    if(!rank){
        L = (count_floats.back())/D;
    }

    // if(rank){
    //     for(int i=0; i<vect_temp.size(); i++){
    //         cout<<vect_temp[i]<<endl;
    //     }
    // }

    fseek(fo, (sizeof(float))*((rank)? count_floats[rank-1]: 0), SEEK_SET);
    fwrite(vect_temp.data(), vect_temp.size(), sizeof(float), fo);
    fclose(fo);
}

int main(int argc, char* argv[]){
    auto begin = std::chrono::high_resolution_clock::now();
    
    if(argc < 3){
        //cout<<"Invalid input";
        exit(-1);
    }

    string in_path = argv[1];
    string out_path = argv[2];

    if (!fs::is_directory(out_path) || !fs::exists(out_path)) { // Check if src folder exists
        fs::create_directory(out_path); // create src folder
    }

    int max_level;
    read_int(max_level, in_path+"/max_level.txt");
    write_int(max_level, out_path+"/max_level.bin");
    
    int ep;
    read_int(ep, in_path+"/ep.txt");
    write_int(ep, out_path+"/ep.bin");
    
    vector<int> level;
    read_vector(level, in_path+"/level.txt");
    write_vector(level, out_path+"/level.bin");

    vector<int> indptr;
    read_vector(indptr, in_path+"/indptr.txt");
    write_vector(indptr, out_path+"/indptr.bin");
    
    vector<int> level_offset;
    read_vector(level_offset, in_path+"/level_offset.txt");
    write_vector(level_offset, out_path+"/level_offset.bin");
    

    string index_file_text = in_path+"/index.txt", index_file_binary = out_path+"/index.bin";
    FILE *fo1 = fopen(index_file_binary.c_str(), "wb");

    string vector_file_text = in_path+"/vect.txt", vector_file_binary = out_path+"/vect.bin";
    FILE *fo2 = fopen(vector_file_binary.c_str(), "wb");

    int L, D, I;

    // MPI_Find_D(vector_file_text, D);

    int rank, size;
    MPI_Init(NULL, NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    MPI_Parallel_Write_int(index_file_text, fo1, rank, size, I);

    MPI_Parallel_Write_double(vector_file_text, fo2, rank, size, L, D);

    if(!rank){
        write_int(L, out_path+"/L.bin");
        write_int(D, out_path+"/D.bin");
        write_int(I, out_path+"/I.bin");
    }

    MPI_Finalize();

    auto end = std::chrono::high_resolution_clock::now();
    if(rank==0){
        auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin);
        float duration = (1e-6 * (std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)).count());
        cout << "Time taken " << duration << "ms" << endl;
    }
    return 0;
}