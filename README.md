# Game of Life CUDA-CL
El proyecto se divide en 2 carpetas: experiment y src, dentro del primero estan todos los datos y gráficos usados en el informe. En el segundo se encuentra el código fuente de las 3 soluciones diseñadas.

## Instrucciones de uso
El proyecto esta hecho con CMake y requiere tanto como OpenCL 3.0 como CUDA 6.1, para compilar el proyecto se debe crear la carpeta build con el siguiente comando en la carpeta raíz del proyecto:
```
mkdir build && cd build
```

Luego se debe ejecutar el siguiente comando para compilar el proyecto:
```
cmake -B . -S ..
```

Y finalmente:
```
cmake --build .
```

Los ejecutables se encontraran dentro de la carpeta src dentro de build, donde cada habrá una carpeta con cada API y dentro de esta un ejecutable acorde a la configuración (por defecto Debug).

## Experimentos
Los experimentos se hicieron con CTest y dentro del CMakeLists.exe de experiement se pueden ver los comandos usados para cada experimento. Para ejecutar los experimentos se debe utilizar Visual Studio ya que son tests. Los resultados de estos tests se guardan dentro de la carpeta experiment/data.
