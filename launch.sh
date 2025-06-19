rm -rf build libgstcudafilter.so

cmake -S . -B build -D USE_CUDA=ON

cmake --build build

export GST_PLUGIN_PATH=$(pwd)

ln -s ./build/libgstcudafilter-cu.so libgstcudafilter.so

gst-launch-1.0 uridecodebin uri="file://$(pwd)/$1" ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=output.mp4
#gst-launch-1.0 -e -v v4l2src ! jpegdec ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
