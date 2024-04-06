#ifndef INCLUDE_TONEMAPPING_ACES_LMT_BLUE_ARTIFACT_FIX
#define INCLUDE_TONEMAPPING_ACES_LMT_BLUE_ARTIFACT_FIX

//
// LMT for desaturating blue hues in order to reduce the artifact where bright
// blue colors (e.g. sirens, headlights, LED lighting, etc.) may become
// oversaturated or exhibit hue shifts as a result of clipping.
//

#include "/include/tonemapping/aces/matrices.glsl"

const mat3 correction_matrix = mat3(
     0.9404372683,  0.0083786969,  0.0005471261,
    -0.0183068787,  0.8286599939, -0.0008833746,
     0.0778696104,  0.1629613092,  1.0003362486
);

vec3 blue_light_artifact_fix(vec3 col) {
    return correction_matrix * col;// * correction_matrix;
}

#endif //INCLUDE_TONEMAPPING_ACES_LMT_BLUE_ARTIFACT_FIX
