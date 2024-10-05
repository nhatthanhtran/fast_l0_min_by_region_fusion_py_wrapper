#include "../l0_norm/main_l0.cpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For standard library types
#include <pybind11/numpy.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
namespace py = pybind11;


// convert a cv::Mat to an np.array
py::array mat_to_array(const cv::Mat& im) {
    const ssize_t channels = im.channels();
    const ssize_t height = im.rows;
    const ssize_t width = im.cols;
    const ssize_t dim = sizeof(uchar) * height * width * channels;
    auto data = new uchar[dim];
    std::copy(im.data, im.data + dim, data);
    return py::array_t<uchar>(
        py::buffer_info(
            data,
            sizeof(uchar), //itemsize
            py::format_descriptor<uchar>::format(),
            channels, // ndim
            std::vector<ssize_t> { height, width, channels }, // shape
            std::vector<ssize_t> { width * channels, channels, sizeof(uchar) } // strides
        ),
        py::capsule(data, [](void* f){
            // handle releasing data
            delete[] reinterpret_cast<uchar*>(f);
        })
    );
}


// convert an np.array to a cv::Mat
cv::Mat array_to_mat(const py::array& ar) {
    if (!ar.dtype().is(py::dtype::of<uchar>())) {
        std::cout << "ERROR unsupported dtype!" << std::endl;
        return cv::Mat();
    }

    auto shape = ar.shape();
    int rows = shape[0];
    int cols = shape[1];
    int channels = shape[2];
    int type = CV_MAKETYPE(CV_8U, channels); // CV_8UC3
    cv::Mat mat = cv::Mat(rows, cols, type);
    memcpy(mat.data, ar.data(), sizeof(uchar) * rows * cols * channels);

    return mat;
}


// PYBIND11_MODULE(l0_module, m) {
//     m.def("main_l0", &main_l0);
//     // m.def("add", &add);

//     // py::class_<Example>(m, "Example")
//     //     .def(py::init<int>())
//     //     .def("getValue", &Example::getValue)
//     //     .def("setValue", &Example::setValue);
//     m.def("l0_norm", [](const py::array_t<unsigned char>& input_image, double lambda, int maxSize, int maxLoop) {
//         cv::Mat img = array_to_mat(input_image);
//         double tc = lambda*255*255;
//         // double tc = lambda;
//         cv::Mat result = l0_norm(img, tc, maxSize, maxLoop);
//         return mat_to_array(result);
//     }, py::arg("input_image"), py::arg("lambda"), py::arg("maxSize") = 32, py::arg("maxLoop") = 100);
// }




// convert a cv::Mat to an np.array
py::array mat_to_array_float(const cv::Mat& im) {
    const ssize_t channels = im.channels();
    const ssize_t height = im.rows;
    const ssize_t width = im.cols;
    const ssize_t dim = sizeof(float) * height * width * channels;
    auto data = new float[height * width * channels];
    float* dataPtr = reinterpret_cast<float*>(im.data);

    for(int i=0; i<height * width * channels; i++){
        data[i] = dataPtr[i];
    }
    return py::array_t<float>(
        py::buffer_info(
            data,
            sizeof(float), //itemsize
            py::format_descriptor<float>::format(),
            channels, // ndim
            std::vector<ssize_t> { height, width, channels }, // shape
            std::vector<ssize_t> { width * channels * sizeof (float), channels * sizeof(float), sizeof(float) } // strides
        ),
        py::capsule(data, [](void* f){
            // handle releasing data
            delete[] reinterpret_cast<float*>(f);
        })
    );
}


// convert an np.array to a cv::Mat
cv::Mat array_to_mat_float(const py::array& ar) {
    if (!ar.dtype().is(py::dtype::of<float>())) {
        std::cout << "ERROR unsupported dtype!" << std::endl;
        return cv::Mat();
    }

    auto shape = ar.shape();
    int rows = shape[0];
    int cols = shape[1];
    int channels = shape[2];
    // int type = CV_MAKETYPE(CV_8U, channels); // CV_8UC3
    int type = CV_MAKETYPE(CV_32F, channels);
    cv::Mat mat = cv::Mat(rows, cols, type);
    memcpy(mat.data, ar.data(), sizeof(float) * rows * cols * channels);
    // for (int i = 0; i < rows*cols*channels; i++){
    //     mat.data[i] = ar.data()[i];
    // }

    // auto ar_ptr = ar.unchecked<3>(); // Get an unchecked view for access
    // const float* ar_data = static_cast<const float*>(ar.data());
    // for (int i = 0; i < rows*cols*channels; i++){
    //     mat[i] = ar_data[i];
    // }
    // for (int i = 0; i < rows; ++i) {
    //     for (int j = 0; j < cols; ++j) {
    //         for (int c = 0; c < channels; ++c) {
    //             // Copy each element one at a time
    //             int index = i * cols * channels + j * channels + c;
    //             mat.at<cv::Vec<float, 3>>(i, j)[c] = ar_data[index]; // Adjust if channels > 3
    //         }
    //     }
    // }
    return mat.clone();
}


PYBIND11_MODULE(l0_module, m) {
    m.def("main_l0", &main_l0);
    // m.def("add", &add);

    // py::class_<Example>(m, "Example")
    //     .def(py::init<int>())
    //     .def("getValue", &Example::getValue)
    //     .def("setValue", &Example::setValue);
    m.def("l0_norm", [](const py::array_t<uchar>& input_image, double lambda, int maxSize, int maxLoop, bool verbose) {
        cv::Mat img = array_to_mat(input_image);
        double tc = lambda*255*255;
        // double tc = lambda;
        cv::Mat result = l0_norm(img, tc, maxSize, maxLoop, verbose);
        return mat_to_array(result);
    }, py::arg("input_image"), py::arg("lambda"), py::arg("maxSize") = 32, py::arg("maxLoop") = 100, py::arg("verbose") = false);

    m.def("l0_norm_float", [](const py::array_t<float>& input_image, double lambda, int maxSize, int maxLoop, bool verbose) {
        cv::Mat img = array_to_mat_float(input_image);
        // double tc = lambda*255*255;
        double tc = lambda;
        cv::Mat result = l0_norm_float(img, tc, maxSize, maxLoop, verbose);
        return mat_to_array_float(result);
    }, py::arg("input_image"), py::arg("lambda"), py::arg("maxSize") = 32, py::arg("maxLoop") = 100, py::arg("verbose") = false);
}




// py::array mat_to_array(const cv::Mat& im, float* data_float) {
//     const ssize_t channels = im.channels();
//     const ssize_t height = im.rows;
//     const ssize_t width = im.cols;
//     const ssize_t dim = sizeof(float) * height * width * channels;
//     auto data = new float[dim];
    
//     std::copy(data_float, data_float + height * width * channels, data);
//     // for (int i=0; i < height * width * channels; i++){
//     //     // std::cout << "ERROR unsupported dtype! " << i<<": "<<data_float[i] << std::endl;
//     //     data[i] = data_float[i];
//     // }
//     return py::array_t<float>(
//         py::buffer_info(
//             data,
//             sizeof(float), //itemsize
//             py::format_descriptor<float>::format(),
//             channels, // ndim
//             std::vector<ssize_t> { height, width, channels }, // shape
//             std::vector<ssize_t> { width * channels, channels, sizeof(float) } // strides
//         ),
//         py::capsule(data, [](void* f){
//             // handle releasing data
//             delete[] reinterpret_cast<float*>(f);
//         })
//     );
// }

// cv::Mat array_to_mat(const py::array_t<float>& ar) {
//     if (!ar.dtype().is(py::dtype::of<float>())) {
//         std::cout << "ERROR unsupported dtype!" << std::endl;
//         return cv::Mat();
//     }
//     // auto buf = ar.unchecked();
//     auto shape = ar.shape();
//     int rows = shape[0];
//     int cols = shape[1];
//     int channels = shape[2];
//     // int type = CV_MAKETYPE(CV_8U, channels); // CV_8UC3
//     int type = CV_MAKETYPE(CV_32F, channels);
//     cv::Mat mat = cv::Mat(rows, cols, type);
//     // cv::Mat mat(rows, cols, type, ar.data());
//     const float* data = ar.data();
//     memcpy(mat.data, data, sizeof(float) * rows * cols * channels);
//     float first_value_ = mat.data[0]; // Assuming it's a single-channel float image

//     // Print the first value
//     std::cout << "First value: " << first_value_ << std::endl;

//     // Access the first value
//     float first_value = data[0];

//     // Print the first value (or use it in your processing)
//     std::cout << "Second value: " << first_value << std::endl;


        
//     return mat;
// }


// PYBIND11_MODULE(l0_module, m) {
//     m.def("main_l0", &main_l0);

//     m.def("l0_norm", [](py::array_t<float>& input_image ,double lambda, int maxSize, int maxLoop) {
//         cv::Mat img = array_to_mat(input_image);
//         float* data = const_cast<float*>(input_image.data());

//         // Access the first value
//         float first_value = data[0];

//         // Print the first value (or use it in your processing)
//         std::cout << "First value of the input image: " << first_value << std::endl;


//         float first_value_ = img.data[0]; // Assuming it's a single-channel float image

//         // Print the first value
//         std::cout << "First value of the input image: " << first_value_ << std::endl;
        
        
//         double tc = lambda*255*255;
//         // double tc = lambda;
//         float* result = l0_norm_float(img, data, tc, maxSize, maxLoop);
//         return mat_to_array(img, result);
//         // return result;
//     }, py::arg("input_image"), py::arg("lambda"), py::arg("maxSize") = 32, py::arg("maxLoop") = 100);
// }









// py::array_t<unsigned char> mat_to_array(const cv::Mat& mat) {
//     py::buffer_info buf_info(
//         mat.data,
//         sizeof(unsigned char),
//         py::format_descriptor<unsigned char>::format(),
//         mat.dims,
//         {static_cast<py::ssize_t>(mat.rows), static_cast<py::ssize_t>(mat.cols)},
//         {static_cast<py::ssize_t>(mat.step[0]), static_cast<py::ssize_t>(mat.step[1])}
//     );
//     return py::array_t<unsigned char>(buf_info);
// }

// cv::Mat array_to_mat(const py::array_t<unsigned char>& arr) {
//     py::buffer_info buf_info = arr.request();
//     return cv::Mat(buf_info.shape[0], buf_info.shape[1], CV_8UC1, (unsigned char*)buf_info.ptr);
// }
