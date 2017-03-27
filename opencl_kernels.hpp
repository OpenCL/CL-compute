#ifndef CL_COMPUTE_OPENCL_KERNELS_HPP
#define CL_COMPUTE_OPENCL_KERNELS_HPP

#include <boost/compute/core.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/detail/sha1.hpp>
#include <boost/compute/utility/source.hpp>
#include <boost/compute/utility/program_cache.hpp>

// Using online caching
boost::compute::event saxpy_caching(const boost::compute::vector<float>& x,
                                    const boost::compute::vector<float>& y,
                                    const float a,
                                    boost::compute::command_queue& queue)
{
    const boost::compute::context &context = queue.get_context();
    const std::string kernel_name = "saxpy";

    std::string source = BOOST_COMPUTE_STRINGIZE_SOURCE(
        __kernel void saxpy(__global float *x,
                            __global float *y,
                            const float a)
        {
            const uint i = get_global_id(0);
            y[i] = a * x[i] + y[i];
        }
    );

    // Get global cache (online caching)
    boost::shared_ptr<boost::compute::program_cache> global_cache =
        boost::compute::program_cache::get_global_cache(context);

    // Set compilation options and cache key
    std::string options;
    std::string key = "__clcompute_" + kernel_name +
        static_cast<std::string>(boost::compute::detail::sha1(source));

    // Get compiled program from online cache,
    // load binary (offline caching) or compile it
    boost::compute::program program =
        global_cache->get_or_build(key, options, source.c_str(), context);

    // Get kernel
    boost::compute::kernel k = program.create_kernel(kernel_name);

    // Set arguments (C++11 variadic templates)
    k.set_args(
        x.get_buffer(),
        y.get_buffer(),
        a
    );

    return queue.enqueue_1d_range_kernel(
        k,
        0, // offset
        y.size(),
        0 // local size, 0 means the OpenCL implementation will determine
          // how to be break the global work-items into appropriate work-group
          // instances.
    );
}

// OpenCL kernel source is in a file
boost::compute::event saxpy_file(const boost::compute::vector<float>& x,
                                 const boost::compute::vector<float>& y,
                                 const float a,
                                 boost::compute::command_queue& queue)
{
    const boost::compute::context &context = queue.get_context();
    const std::string kernel_name = "saxpy";

    // Load source from file
    boost::compute::program program = 
        boost::compute::program::create_with_source_file("./kernels/saxpy.cl", context);

    // offline caching works in that case too
    program.build();

    // Get kernel
    boost::compute::kernel k = program.create_kernel(kernel_name);

    // Set arguments
    {
        size_t arg = 0;
        k.set_arg(arg++, x.get_buffer());
        k.set_arg(arg++, y.get_buffer());
        k.set_arg(arg++, a);
    }

    return queue.enqueue_1d_range_kernel(
        k,
        0, // offset
        y.size(),
        0 // local size, 0 means the OpenCL implementation will determine
          // how to be break the global work-items into appropriate work-group
          // instances.
    );
}

#endif // CL_COMPUTE_OPENCL_KERNELS_HPP