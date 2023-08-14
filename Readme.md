# picture-face-finder
Find faces of you and your friends in a large picture collection.

## setup
### Install dlib with gpu support
Install cuda and [cudnn](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) manually. Then install dlib with gpu support:
```cmd
git clone https://github.com/davisking/dlib.git
cd dlib
mkdir build
cd build
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -A x64
cmake --build .
cd ..
python setup.py install
```

## dlib gpu support
Be sure to [install dlib](https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/) (which face_recognition is a wrapper for) using GPU support. Otherwise it will be way to slow. 

Enabling GPU support in dlib gave some problems because I kept getting this error:
```
-- Found CUDA: C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8 (found suitable version "11.8", minimum required is "7.5") 
CMake Warning at dlib/CMakeLists.txt:637 (message):
  You have CUDA installed, but we can't use it unless you put visual studio
  in 64bit mode.


-- Found CUDA, but CMake was unable to find the cuBLAS libraries that should be part of every basic CUDA install. Your CUDA install is somehow broken or incomplete. Since cuBLAS is required for dlib to use CUDA we won't use CUDA.
-- DID NOT FIND CUDA
-- Disabling CUDA support for dlib.  DLIB WILL NOT USE CUDA
```

Which was solved by running cmake and manually setting the architecture:
```
cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -A x64
```

## face_recognition error
The face_recognition gave an error, which was fixed by commenting out the line (which is possible since we're using the cnn face detector):
```
face_detector = dlib.get_frontal_face_detector()
```

## log example
```
C:\Users\arnod\AppData\Local\Programs\Python\Python39\python.exe C:\Users\arnod\PycharmProjects\picture-face-finder\main.py 
Loading known faces: 100%|██████████| 13/13 [00:09<00:00,  1.44it/s]
Scanned 21 known faces
Warning: output already exists. Do you want to delete it? (y/n)
y
Scanning pictures: 0it [00:00, ?it/s]
Scanning pictures\b test: 100%|██████████| 2/2 [00:00<00:00,  3.00it/s]
Scanning pictures\Vrijdag 11 augustus 2023 -- 16 uur -_ start: 100%|██████████| 788/788 [09:41<00:00,  1.35it/s]
Scanning pictures\Vrijdag 11 augustus 2023 -- start-_00u: 100%|██████████| 228/228 [01:25<00:00,  2.66it/s]
Scanning pictures\Zaterdag 12 augustus 2023 -- 00u-_ 06u: 100%|██████████| 163/163 [02:57<00:00,  1.09s/it]
Scanning pictures\Zaterdag 12 augustus 2023 -- 06u-_10u: 100%|██████████| 505/505 [06:13<00:00,  1.35it/s]
Scanning pictures\Zaterdag 12 augustus 2023 -- 10u-_14u: 100%|██████████| 1005/1005 [15:23<00:00,  1.09it/s]
Scanning pictures\Zaterdag 12 augustus 2023 -- 14u-_18u: 100%|██████████| 729/729 [11:58<00:00,  1.01it/s]
Scanning pictures\Zaterdag 23 augustus 2023 -- 18u-_22u: 100%|██████████| 151/151 [02:39<00:00,  1.05s/it]
Found 221 pictures with known faces and copied them to output

Process finished with exit code 0
```

