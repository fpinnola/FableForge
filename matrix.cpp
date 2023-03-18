
class Matrix
{
private:
    /* data */
    int rows, cols;
    double* data;

public:
    Matrix(/* args */);
    ~Matrix();
};

Matrix::Matrix(/* args */)
{
    rows = 0;
    cols = 0;
}

Matrix::~Matrix()
{
    // if (rows)
    if (rows > 0 && cols > 0) {
        delete[] data;
    }
}



int main(int argc, char const *argv[])
{
    /* code */
    return 0;
}
