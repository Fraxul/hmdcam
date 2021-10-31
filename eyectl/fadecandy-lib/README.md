Fadecandy C++ Mini-Framework
----------------------------

This is a tiny self-contained package of C++ header files that you can use to implement blazing fast LED patterns that can run on large or small computers. There are no other dependencies or libraries to fuss with, and you can copy these headers right into your own project.

This library includes:

* Efficient [Open Pixel Control](http://openpixelcontrol.org/) client
* JSON parsing ([rapidjson](https://code.google.com/p/rapidjson/))
* Vector math ([SVL](http://www.cs.cmu.edu/~ajw/doc/svl.html))
* PNG decoding ([picopng](http://lodev.org/lodepng/))
* KD-trees for spatial search ([nanoflann](https://code.google.com/p/nanoflann/))
* Texture sampling with bilinear interpolation
* HSV color space conversion
* Particle system rendering, with floating point precision
* [Perlin Noise](http://www.algorithmic-worlds.net/info/info.php?page=pg-perlin) function
* Generalized *Effect* framework
* Main loop with smooth frame rate throttling
* Concurrent rendering on multiple CPU cores, via the EffectMixer class
* Command line parameters
* Debug output including performance metrics

Writing an Effect is like writing a GPU shader. At its core, an Effect is a function that maps LED metadata to an LED color.

The color in this case uses floating point precision, so you can chain effects without losing color precision. It's converted to 8-bit per channel just before being sent out to the Fadecandy board.

The LED metadata includes a 3D position vector as well as any other data you choose to include in your layout JSON file. The full JSON object model is available for you to use to describe the properties of each light in your project.


Usage
-----

1. Copy the `lib` directory somewhere on your project's include search path.
2. Use `Makefile` and `simple.cpp` as an example for starting your own project.
3. Write an *Effect* subclass that does awesome things.
4. Instantiate an *EffectRunner* and your *Effect*.
5. Set up default options on the *EffectRunner*.
6. Let the *EffectRunner* main loop do the rest.

License
-------

All components are under BSD-style licenses. You're free to use this code in commercial or free software, as long as you keep copyright notices in the source.

Download
--------

The latest revision of this library lives on GitHub in the [`scanlime/fadecandy`](https://github.com/scanlime/fadecandy/) repository. at [`examples/cpp/lib`](https://github.com/scanlime/fadecandy/tree/master/examples/cpp/lib).

