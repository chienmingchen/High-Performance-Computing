#include <iostream>
#include <string>

int main(int argc, char* argv[])
{
    if (argc < 2) {
        std::cerr << "error!! please check the argument" << std::endl;
        return 1;
    }

    int N = std::stoi(argv[1],nullptr,0);

    for(int i = 0; i <= N; i++)
        std::cout << i << " "; 
    std::cout << std::endl;
    return 0;
}
