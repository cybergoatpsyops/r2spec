#include <stdint.h>

const unsigned char fixture_magic[] = {'R', '2', 'S', 'P'};

int32_t add_fixture_values(int16_t left, int16_t right)
{
    return (int32_t)left + (int32_t)right;
}

int main(void)
{
    return add_fixture_values(1, 2);
}
