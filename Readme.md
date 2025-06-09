

## Mise en route

1. Installer gstreamer (si vous travaillez avec votre machine)
2. Compiler le code d'exemple avec cmake (le filtre se trouve dans le dossier ``./build``)
3. Télécharger la vidéo d'exemple ``https://cloud.lrde.epita.fr/s/tyeqDFYfXM8i3km`` (video03.avi)
4. Exporter le chemin du filtre dans la variable d'environnement ``GST_PLUGIN_PATH``
5. Ajouter un symlink vers le plugin C++ ou sa version CUDA
6. Lancer l'application du filter sur la vidéo et l'enregistrer en mp4 *dans votre afs*
7. En local, visualiser la vidéo avec *vlc*


```sh
cmake -S . -B build --preset release -D USE_CUDA=ON  # 2 (ou debug)
cmake --build build                                  # 2


wget ??? # 3
export GST_PLUGIN_PATH=$(pwd)                                         # 4
ln -s ./build/libgstcudafilter-cpp.so libgstcudafilter.so             # 5


gst-launch-1.0 uridecodebin uri=file://$(pwd)/video03.avi ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=output.mp4 #5
```

## Code

Les seuls fichiers à modifier sont normalement ``filter_impl.cu`` (version cuda) et ``filter_impl.cpp`` (version cpp). Pour basculer entre l'utilisation du filter en C++ et du filtre en CUDA, changer le lien symbolique vers le bon ``.so``.


## Uiliser *gstreamer*

### Flux depuis la webcam -> display

Si vous avez une webcam, vous pouvez lancer gstreamer pour appliquer le filter en live et afficher son FPS.

```sh
gst-launch-1.0 -e -v v4l2src ! jpegdec ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

### Flux depuis une vidéo locale -> display

Même chose pour une vidéo en locale.

```sh
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/video03.avi !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink
```

## Flux depuis une vidéo locale -> vidéo locale

Pour sauvegarder le résulat de l'application de votre filtre.

```sh
gst-launch-1.0 uridecodebin uri=file://$(pwd)/video03.avi ! videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! video/x-raw, format=I420 ! x264enc ! mp4mux ! filesink location=output.mp4
```


## Bench FPS du traitement d'une vidéo

Enfin pour bencher la vitesse de votre filtre. Regarder la sortie de la console pour voir les fps. 

```sh
gst-launch-1.0 -e -v uridecodebin uri=file://$(pwd)/video03.avi !  videoconvert ! "video/x-raw, format=(string)RGB" ! cudafilter ! videoconvert ! fpsdisplaysink video-sink=fakesink sync=false
```

## Optimizations

## Algorithm Improvements
### Improved Hysteresis Implementation

Uses bfs implementation using queues. This allows us from doing Hysteresis a max amount of 100 times, to 3-5 times.

### Improved Math

Uses cuda math functions for faster calculations
