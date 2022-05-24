>This is project for matting, which is based on c++

### 1. How to use
#### 1.1 download pytorch lib

you can find this lib from [Pytorch website](https://pytorch.org/get-started/locally/), and put it into ./3rd folder.

![download websit](./example/download_website.png)
#### 1.3 download pytorch lib

you can find this lib from [OpenCV website](https://pytorch.org/get-started/locally/), and put it into ./3rd folder.
![download websit](./example/download_opencv.png)

#### 1.4 install opencv
If you use windows, you can directly install the library (choose the location and click the next step). If you use Linux or macOS, you must compile the source code (you can find the document from ![opencv website](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html)). 


### 2. Project Structure
```
.
├── 3rd # source code
│   ├── libtorch/
│   └── ...
├── src
│   ├── Test
│   ├── MattingPlugin
│   └── ...
├── LICENSE
├── build.sh
├── clean.sh
├── CMakeLists.txt
└── README.md
```
