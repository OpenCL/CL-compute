#include <iostream>
#include <boost/compute.hpp>

#include "opencl_kernels.hpp"

namespace bc = boost::compute;

void print_vector(const bc::vector<bc::float_> device_vector)
{    
    for(auto value : device_vector)
    {
        std::cout << value << " ";
    }
    std::cout << "\n";
}

int main (int argc, char *argv[]) 
{ 
    // Get default device and setup context
    bc::device device = bc::system::default_device();
    bc::context context(device);
    bc::command_queue queue(context, device);

    // Create data arrays on host
    bc::float_ host_x[] = { 1.0f, -2.0f, 3.0f, 4.0f, -5.0f };
    bc::float_ host_y[] = { 1.0f, 3.0f, -5.0f, 7.0f, 9.0f };

    // Create vectors on device
    bc::vector<bc::float_> device_x(
        host_x, host_x + 5, queue
    );
    bc::vector<bc::float_> device_y(
        host_y, host_y + 5, queue
    );

    std::cout << "x: ";
    print_vector(device_x);
    std::cout << "y: ";
    print_vector(device_y);

    saxpy_caching(device_x, device_y, -2.0f, queue).wait();

    std::cout << "\ny = -2.0f * x + y\n";
    std::cout << "y: ";
    print_vector(device_y);

    {
        auto e = saxpy_file(device_x, device_y, 2.0f, queue);
        e.wait();
    }

    std::cout << "\ny = 2.0f * x + y\n";
    std::cout << "y: ";
    print_vector(device_y);

    // saxpy using transform
    {
        using boost::compute::_1;
        using boost::compute::_2;
        boost::compute::transform(
            device_x.begin(), device_x.end(), // input1
            device_y.begin(), // input2
            device_y.begin(), // result
            -4.0f * _1 + _2, // lambda exp
            queue
        );
        auto e = queue.enqueue_marker();
        e.wait();
    }

    std::cout << "\ny = -4.0f * x + y\n";
    std::cout << "y: ";
    print_vector(device_y);

    return 0;
}