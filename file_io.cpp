#include <iostream>
#include <fstream>
#include <set>

int main(int argc, char const *argv[])
{
    /* code */
    std::ifstream inputFile("input.txt");

    if (!inputFile.is_open()) {
        std::domain_error("Failed ot open the input file");
        return 1;
    }

    const int maxFileSize = 1000000;
    char* charList = new char[maxFileSize];
    int charCount = 0;
    char c;

    while (inputFile.get(c)) {
        *(charList + charCount) = c;
        ++charCount;
        if (charCount == maxFileSize) {
            std::domain_error("Input file is too large");
            return 1;
        }
    }

    inputFile.close();

    std::cout << charCount << std::endl;

    // Create alphabet
    std::set<int> alphabet;


    // Add characters to set
    for (int i = 0; i < charCount; ++i) {
        alphabet.insert(*(charList + i));
        // std::cout << *(charList + i);
    }

    // for (auto& str : alphabet) {
    //     std::cout << "'" << str << ' ' << (char)str << "' ";
    // }


    // Cleanup
    delete[] charList;

    return 0;

}
