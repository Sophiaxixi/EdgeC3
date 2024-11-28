### EdgeC3

The code was mainly re-written based on a distinguished and excellent source code from S. Wang, T. Tuor, T. Salonidis, K. K. Leung, C. Makaya, T. He, and K. Chan, "Adaptive federated learning in resource constrained edge computing systems," IEEE Journal on Selected Areas in Communications, vol. 37, no. 6, pp. 1205 – 1221, Jun. 2019.

#### Preparation
The code runs on Python 3 with Tensorflow version 1 (>=1.13).
Download the datasets manually and put them into the `datasets` folder.
- For MNIST dataset, download from <http://yann.lecun.com/exdb/mnist/> and put the standalone files into `datasets/mnist`.
- For CIFAR-10 dataset, download the "CIFAR-10 binary version (suitable for C programs)" from <https://www.cs.toronto.edu/~kriz/cifar.html>, extract the standalone `*.bin` files and put them into `datasets/cifar-10-batches-bin`.
- For SAR dataset, download `CSI_SAR_RT.zip` from <https://ieee-dataport.org/documents/imgfiwifi-based-activity-recognition-dateset>.

To test the code: 
- Run `server.py` and wait until you see `Waiting for incoming connections...` in the console output.
- Run 7 parallel instances of `client%d.py` on the same machine as the server. 
- You will see console outputs on both the server and clients indicating message exchanges. 

#### Code Structure

All configuration options are given in `config.py` which also explains the different setups that the code can run with.

The results are saved as CSV files in the `results/Solver` folder. 
The CSV files in the folder will be read by the cvx program.