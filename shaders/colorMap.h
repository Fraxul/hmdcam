// Colormap shaders from https://github.com/kbinani/colormap-shaders
float jet_colormap_red(float x) { if (x < 0.7) { return 4.0 * x - 1.5; } else { return -4.0 * x + 4.5; } }
float jet_colormap_green(float x) { if (x < 0.5) { return 4.0 * x - 0.5; } else { return -4.0 * x + 3.5; } }
float jet_colormap_blue(float x) { if (x < 0.3) { return 4.0 * x + 0.5; } else { return -4.0 * x + 2.5; } }
vec3 jet_colormap(float x) {
  return vec3(
    clamp(jet_colormap_red(x), 0.0, 1.0),
    clamp(jet_colormap_green(x), 0.0, 1.0),
    clamp(jet_colormap_blue(x), 0.0, 1.0));
}

vec3 applyColormap(float x, int mode) {
  if (mode == 1)
    return jet_colormap(x);
  else
    return vec3(x); // greyscale fallback
}

