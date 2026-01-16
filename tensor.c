#include <stdio.h>
#include <stdlib.h>

typedef struct {
    float *data;
    int rows;
    int cols;
    int size;
} Tensor;

Tensor *create_tensor(int rows, int cols)
{
    Tensor *t = (Tensor*)malloc(sizeof(Tensor));

    t->rows = rows;
    t->cols = cols;
    t->size = rows * cols;

    t->data = (float*)calloc(t->size, sizeof(float));

    return t;
}

void delete_tensor(Tensor* t) 
{
    free(t->data);
    free(t);
}

void set_value(Tensor* t, int index, float value) 
{
    if (index < t->size) {
        t->data[index] = value;
    }
}

float get_value(Tensor *t, int index)
{
    if (index < t->size) {
        return t->data[index];
    }

    return 0.0f;
}

int get_index(int row, int col, int width)
{
    return (row * width) + col;
}

Tensor *matmul(Tensor *a, Tensor *b) 
{
    if (a->cols != b->rows) {
        printf("Error: Dimension mismatch! A: %d%d, B: %dx%d\n",
            a->rows, a->cols, b->rows, b->cols);
        return NULL;
    }

    Tensor *result = create_tensor(a->rows, b->cols);

    for (int i = 0; i < a->rows; i++) {
        for (int j = 0; j < b->cols; j++) {
            float sum = 0.0f;

            for (int k = 0; k < a->cols; k++) {
                int idx_a = get_index(i, k, a->cols);
                int idx_b = get_index(k, j, b->cols);
                sum += a->data[idx_a] * b->data[idx_b];
            }

            int idx_res = get_index(i, j, result->cols);
            result->data[idx_res] = sum;
        }
    }

    return result;
}

void relu(Tensor *t) 
{
    for (int i = 0; i < t->size; i++) {
        if (t->data[i] < 0) {
            t->data[i] = 0;
        }
    }
}

void add_tensor(Tensor *a, Tensor *b) 
{
    for (int i = 0; i < a->size; i++) {
        a->data[i] += b->data[i];
    }
}