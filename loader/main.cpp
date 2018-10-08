#include"cnpy.h"
#include<complex>
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>
using namespace std;

int main()
{
    //load it into a new array
	vector<size_t> shape;
	cnpy::NpyArray arr;
    cnpy::npy_load("/home/isak/DNNsim/loader/bvlc_alexnet/act-conv1-0.npy", arr, shape);
    float* loaded_data = arr.data<float>();
    int i = 0;
    int j = 2;
    int k = 150;
    int l = 132;
    long index = shape[1]*shape[2]*shape[3]*i + shape[2]*shape[3]*j + shape[3]*k + l;
    cout<<loaded_data[index] << endl;
    cout<<shape[0]<<", "<<shape[1]<<", "<<shape[2]<<", "<<shape[3] << endl;

    //append the same data to file
    //npy array on file now has shape (Nz+Nz,Ny,Nx)
    //cnpy::npy_save("arr1.npy",&data[0],{Nz,Ny,Nx},"a");
}
