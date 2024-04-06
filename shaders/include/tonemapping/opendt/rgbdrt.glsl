#ifndef INCLUDE_TONEMAPPING_OPENDT_RGBDRT
#define INCLUDE_TONEMAPPING_OPENDT_RGBDRT

#include "/include/utility/color.glsl"

/* RGB Display Rendering Transform (rgbDRT)
      a very simple classic per-channel display transform
    -------------------------------------------------
      v0.0.2
      Written by Jed Smith
      https://github.com/jedypod/rgbdrt

      MIT License
    -------------------------------------------------
*/

#define average 1
#define dim     2
#define dark    3

#define Lp 100.0
#define surround dim
#define invert 0



/* Custom rendering gamut for "rgbDT" display transform
    rxy 0.859 0.264
    gxy 0.137 1.12
    bxy 0.085 -0.096
    wxy 0.3127 0.329
*/
const mat3 matrix_rgbdt_to_xyz = mat3(
    vec3(0.72113842f, 0.11148937f, 0.11782813f),
    vec3(0.22163042f, 0.91144598f, -0.13307647f),
    vec3(-0.10325963f, -0.20914429f, 1.40146172f)
);
// 10% Rec.709 weighted desaturation
const mat3 matrix_rgbdt_desat = mat3(
    vec3(0.921264f, 0.0715169f, 0.00721923f),
    vec3(0.0212639f, 0.971517f, 0.00721923f),
    vec3(0.0212639f, 0.0715169f, 0.907219f)
);



/* ##########################################################################
    Display Rendering Transform
    ---------------------------
*/

vec3 rgbdrtransform(vec3 rgb) {

    const mat3 in_to_xyz = rec2020_to_xyz;
    const mat3 xyz_to_in = xyz_to_rec2020;

    const int tf = 0;

    const mat3 xyz_to_display = xyz_to_rec2020;
    const mat3 display_to_xyz = rec2020_to_xyz;

    /* Surround compensation
          Values based on a simple one-man perceptual experiment
          matching appearance of a variety of images in a dark room
          with image inset in a flat field of different illumination levels.
          dark = 0 nits (at least as close to it as an OLED tv can get).
          dim = 5 nits
          average = ~35 nits
          DO YOUR OWN EXPERIMENT! IT IS FUN!
      */

    float su;
    if (surround == average)    su = 0.9f;
    else if (surround == dim)   su = 0.95f;
    else if (surround == dark)  su = 1.0f;

    const int eotf = 0;

    // Display scale: remap peak white depending on output inverse EOTF
    const float ds = eotf == 4 ? 0.01f : eotf == 5 ? 0.1f : 100.0f/Lp;

    /* Tonescale
          Uses the "Michaelis-Menten Spring Dual-Contrast" Tonescale function
          https://colab.research.google.com/drive/10C3HvDuoAhYad1qOG2r0v8fGR-5VdpO5
      */
    // Calculate tonescale parameters
    const float fl = 0.01; // flare
    const float c0 = 1.2; // pre-tonemap contrast
    const float cs = pow(0.18, 1.0 - c0);  // pivoted contrast scale
    const float c1 = 1.1; // post-tonemap contrast
    float p = c1 * su; // surround compensation, unconstrained

    // boost peak to clip : ~32@100nits, ~75~1000nits, 100@4000nits
    const float w1 = pow(0.595 * Lp/10000.0, 0.931) + 1.037;
    const float s1 = w1 * Lp/100.0; // scale y: 1@100nits, 40@4000nits

    const float ex = -0.26; // 0.18 -> 0.1 @ 100nits
    const float eb = 0.08; // exposure boost with > Lp
    const float e0 = pow(2.0f, ex + eb * log2(s1));
    const float s0 = pow(s1/e0, 1.0f/c1);



    if (invert == 0) {
        /* Forward Display Rendering
               ----------------------------------------------------------- */

        //rgb = log2lin(rgb, tf); // Assume tf = 0
        rgb = rgb * in_to_xyz;
        rgb = rgb * inverse(matrix_rgbdt_to_xyz);
        rgb = max(vec3(0.0), rgb);

        // Tonescale
        rgb.x = rgb.x < 0.18f ? cs * pow(rgb.x, c0) : c0*(rgb.x - 0.18f) + 0.18f;
        rgb.y = rgb.y < 0.18f ? cs * pow(rgb.y, c0) : c0*(rgb.y - 0.18f) + 0.18f;
        rgb.z = rgb.z < 0.18f ? cs * pow(rgb.z, c0) : c0*(rgb.z - 0.18f) + 0.18f;
        rgb = s1 * pow(rgb / (rgb + s0), vec3(p));
        rgb = rgb * rgb / (rgb + fl);

        rgb = rgb * matrix_rgbdt_to_xyz;
        rgb = rgb * xyz_to_display;
        rgb = rgb * (matrix_rgbdt_desat);
        rgb *= ds;

        // Inverse EOTF
        rgb = clamp(rgb, 0.0, 1.0);
        float eotf_p = 2.0f + eotf * 0.2f;
        if ((eotf > 0) && (eotf < 4)) rgb = pow(rgb, vec3(1.0f / eotf_p));

    } else {
        /* Inverse Display Rendering
                 ----------------------------------------------------------- */

        float eotf_p = 2.0f + eotf * 0.2f;
        if ((eotf > 0) && (eotf < 4)) rgb = pow(rgb, vec3(eotf_p));

        rgb /= ds;
        rgb = rgb * inverse(matrix_rgbdt_desat);
        rgb = rgb * display_to_xyz;
        rgb = rgb * inverse(matrix_rgbdt_to_xyz);

        // Tonescale
        rgb = (rgb + sqrt(rgb * (4.0f * fl + rgb))) / 2.0f;
        rgb = s0/(pow(s1/rgb, vec3(1.0f/p)) - 1.0f);
        rgb.x = rgb.x < 0.18f ? pow(rgb.x/cs, 1.0f/c0) : (rgb.x - 0.18f)/c0 + 0.18f;
        rgb.y = rgb.y < 0.18f ? pow(rgb.y/cs, 1.0f/c0) : (rgb.y - 0.18f)/c0 + 0.18f;
        rgb.z = rgb.z < 0.18f ? pow(rgb.z/cs, 1.0f/c0) : (rgb.z - 0.18f)/c0 + 0.18f;

        rgb = rgb * matrix_rgbdt_to_xyz;
        rgb = rgb * xyz_to_in;
        //rgb = lin2log(rgb, tf); // Assume tf = 0
    }

    return rgb;
}



#undef average
#undef dim
#undef dark

#undef Lp
#undef surround
#undef invert

#endif // INCLUDE_TONEMAPPING_OPENDT_RGBDRT
