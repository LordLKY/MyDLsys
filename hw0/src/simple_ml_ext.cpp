#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

#include <cstring>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE

    /*
    NUM = X.shape[0]
    for i in range(NUM // batch):
      num = batch
      X0 = X[i * batch: (i + 1) * batch]
      y0 = y[i * batch: (i + 1) * batch]
  
      Output = np.dot(X0, theta)
      Output -= np.amax(Output, axis=1).reshape([num, 1])
      Output_exp = np.exp(Output)
      Z = Output_exp / np.sum(Output_exp, axis=1).reshape([num, 1])

      Iy = np.zeros_like(Z, dtype=np.float32)
      Iy[np.arange(Iy.shape[0]), y0] = 1

      delta = np.dot(X0.T, Z - Iy) / batch
      theta -= lr * delta
    */

    for(int i0 = 0; i0 < (m / batch); i0++){
      float delta[n * k];
      memset(delta, 0, sizeof(delta));

      for(int i1 = 0; i1 < batch; i1++){
        float output[k];
        float output_dim_max = -0xffffff;
        float output_exp_sum = 0;
        memset(output, 0, sizeof(output));
        int y_label = i0 * batch + i1;
        int x_start = y_label * n;

        for(int i2 = 0; i2 < n; i2++){
          int x_dim_start = i2 * k;
          for(int i3 = 0; i3 < k; i3++){
            output[i3] += theta[x_dim_start + i3] * X[x_start + i2];
            if(i2 == n - 1){
              output_dim_max = output_dim_max > output[i3] ? output_dim_max : output[i3];
            }
          }
        }

        for(int i2 = 0; i2 < k; i2++){
          output[i2] = std::exp(output[i2] - output_dim_max);
          output_exp_sum += output[i2];
        }

        for(int i2 = 0; i2 < k; i2++){
          output[i2] /= output_exp_sum;
        }

        output[y[y_label]] -= 1;

        for(int i2 = 0; i2 < n; i2++){
          for(int i3 = 0; i3 < k; i3++){
            delta[i2 * k + i3] += X[x_start + i2] * output[i3];
          }
        }
      }

      for(int i1 = 0; i1 < (n * k); i1++){
        theta[i1] -= delta[i1] * lr / batch;
      }
    }

    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
