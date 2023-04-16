#include <iostream>
#include <fstream>
#include <string.h>


// Removing problematic tokens
bool includeChar(char c) {
    if (strcmp(&c, "\n") == 0) return false;
    if ((int)c == -61) return false;
    if ((int)c == -122) return false;
    if ((int)c == -90) return false;
    if ((int)c == -61) return false;

    // if (strcmp(&c, "\n") == �) return false;
    return true;
}

int main(int argc, char const *argv[])
{

    // Read input file
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
        // decide if want character or not
        if (includeChar(c)) {
            *(charList + charCount) = c;
            ++charCount;
            if (charCount == maxFileSize) {
                std::domain_error("Input file is too large");
                return 1;
            }
        }
    }

    inputFile.close();

    std::ofstream outputFile("processed.txt");

    outputFile << charList;
    outputFile.close();
    return 0;
}