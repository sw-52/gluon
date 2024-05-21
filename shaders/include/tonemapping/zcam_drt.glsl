#ifndef INCLUDE_TONEMAPPING_ZCAM_DRT
#define INCLUDE_TONEMAPPING_ZCAM_DRT

/*
 *  Source: https://github.com/alexfry/output-transforms-dev/blob/main/display-transforms/nuke/ZCAM_DRT_v013.blink
 *
 *  Converted to GLSL and C# by sw-52
 *
 *  LICENSE: "/include/tonemapping/aces/license.md"
 */

// "safe" power function to avoid NANs or INFs when taking a fractional power of a negative base
// this one initially returned -pow(abs(b), e) for negative b
// but this ended up producing undesirable results in some cases
// so now it just returns 0.0 instead
#define spow(x, y) (((x) < 0.0 && (y) != floor(y)) ? 0.0 : pow(x, y))

/*float spow(float base, float exponent) {
    if (base < 0.0 && exponent != floor(exponent)) return 0.0;
    else return pow(base, exponent);
}*/


//
// Input Parameters
//

// Encoding of the Input Image
// 0: Linear
// 1: ACEScct
// 2: sRGB
// 3: BT.1886 (Gamma 2.4)
// 4: Gamma 2.6
// 5: ST2084
const int encodingIn = 0;

// Primaries of the Input Image
// 0: AP0-ACES
// 1: AP1-ACES
// 2: sRGB/Rec.709-D65
// 3: Rec.2020-D65
// 4: P3-D65
// 5: P3-DCI
const int primaries_in = 3;

// Tonescale mode
// 0: SSTS
// 1: MMSDC
// 2: Daniele Compression Curve
const int toneScaleMode = ZCAMDRT_TONESCALE_MODE; // 0 // [0 1 2]


//
// ZCAM Paramters
//

// Chomatic Adaptation Transform to Use
// 0: None
// 1: XYZ Scaling
// 2: Bradford
// 3: CAT02
// 4: Zhai2018 (two-step)
const int catType = ZCAMDRT_CAT_TYPE; // 0 // [0 1 2 3 4]

// Disable Degree of Adaptation Model for Zhai2018 CAT
// This is only effective if the limit primaries have a non-D65 white point
// since the input conversion is assumed to be fully adapted
// and the output conversion does not apply a CAT
const bool discountIlluminant = true;
// disable the degree of adaptation model for the Zhai2018 CAT\nthis is only effective if the limiting primaries do not use a D65 white point

// Reference Luminance in Cd/sqm
const float referenceLuminance = ZCAMDRT_REF_LUMINANCE; // [0 - 200] // 100.0
// the ZCAM reference luminance in Cd/sqm

// Background Luminance in Cd/sqm
const float backgroundLuminance = ZCAMDRT_BG_LUMINANCE; // [0 - 100] // 10.0
// the ZCAM background luminance in Cd/sqm

// Viewing Conditions (for output)
// 0: Dark
// 1: Dim
// 2: Average
const int viewingConditions = ZCAMDRT_SURROUND; // 2 [0 1 2]
const float viewingConditionsCoeff = viewingConditions == 0 ? 0.8 : viewingConditions == 1 ? 0.9 : 1.0;

//
// SSTS Parameters
//

// Toggle SSTS Tone Mapping
#ifdef ZCAMDRT_APPLY_TONECURVE
const bool applyTonecurve = true; // ZCAMDRT_APPLY_TONECURVE
#else
const bool applyTonecurve = false;
#endif
// toggle the SingleStageToneScale transform

// SSTS Luminances Min/Mid/Peak
const vec3 sstsLuminance = vec3(eps, 10.0, 100.0);
// min, mid & peak luminance values in Cd/sqm as parameters for the SSTS

// Toggle Highlight De-Saturation
#ifdef ZCAMDRT_HIGHLIGHT_DESAT
const bool applyHighlightDesat = true; // ZCAMDRT_HIGHLIGHT_DESAT
#else
const bool applyHighlightDesat = false;
#endif
// toggle de-saturating the highlights above SSTS mid luminance based on how much the SSTS has compressed them

// Scale the De-Saturation Applied to the Highlights
const float desatHighlights = ZCAMDRT_HIGHLIGHT_DESAT_SCALE; //  [0 - 5] // 3.5 // custom 1.5
// the amount of desaturation applied to the highlights

//
// Gamut Mapping Parameters
//

// Primaries of the Target Gamut
// 0: AP0-ACES
// 1: AP1-ACES
// 2: sRGB/Rec.709-D65
// 3: Rec.2020-D65
// 4: P3-D65
// 5: P3-DCI
const int primaries_limit = ZCAMDRT_TARGET_GAMUT; // 3


// Toggle Gamut Compression
#ifdef ZCAMDRT_GAMUT_COMPRESS
const bool applyGamutCompression = true; // ZCAMDRT_GAMUT_COMPRESS
#else
const bool applyGamutCompression = false;
#endif
// toggle the gamut compression towards the limiting primaries

// Blend Between Compressing towards
// Target Gamut Cusp Luminance (0.0)
// and SSTS Mid Luminance (1.0)
const float cuspMidBlend = ZCAMDRT_CUSP_MID_BLEND; // [0 - 1] // 0.5  // custom 0.0
// blend the lightness (J) of the focal point of the compression between the lightness of the gamut cusp at the given hue (0.0)  and the mid luminance of the SSTS (1.0)

// the distance of the compression focal point
// from the achromatic axis
// normalised to the distance of the gamut cusp
const float focusDistance = ZCAMDRT_FOCUS_DISTANCE; // [0 - 2] // 0.5 // custom 1.5
// the distance from the achromatic axis of the focal point of the compression where 0.0 is at the achromatic axis and 1.0 the distance of the gamut cusp at the given hue but on the opposite side of the achomatic axis

// Gamut Compression Fuction Parameters
// Threshold / Limit / Power
const vec3 compressionFuncParams = vec3(ZCAMDRT_COMPRESS_THRESHOLD, ZCAMDRT_COMPRESS_LIMIT, ZCAMDRT_COMPRESS_POWER); // [0 - 2] // vec3(0.75, 1.2, 1.2)
// the threshold, limit and power parameters for the PowerP compression function\nvalues below the threshold will not be compressed and values at the limit will be compressed towards the gamut boundary while the power values defines the shape of the curve

// How much the edges of the target RGB cube are smoothed when finding the gamut boundary
// in order to reduce visible contours at the gamut cusps
const float smoothCusps = ZCAMDRT_SMOOTH_CUSPS; // [0 - 1] // 0.0 // custom 0.8
// the amount by how much to smooth the edges and corners of the limiting gamut cube, except the black & white corners.

// When solving for the target gamut boundary
// how many search interval halving steps to perform
const int boundarySolvePrecision = ZCAMDRT_SOLVE_PRECISION; // 10 // [1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 25 30 40 50 75 100 150 200]
const int boundaryWhileMax = ZCAMDRT_SOLVE_PRECISION_2; //  3 // [1 2 3 4 5 6 7 8 9 10]
// the number of iterations used for finding the gamut boundary using the interval bisection method

// Number of iterations to converge on the uncompressed J value
// Because of the compression focus point changes depending on the J value of the uncompressed sample
// we cannot perfectly invert it since the original J value has now been changed by the forward compression
// we can converge on a reasonable approximation of the original J value by iterating the inverse compression
// although this is quite an expensive operation
const int inverseSolverIterations = 10;
// the number of iterations used for finding the original J & M values when applying the inverse gamut compression

//
// Output Parameters
//

// Encoding of the Output Image
// 0: Linear
// 1: ACEScct
// 2: sRGB
// 3: BT.1886 (Gamma 2.4)
// 4: Gamma 2.6
// 5: ST2084
const int encoding_out = 0;

// Primaries of the Output Image
// 0: AP0-ACES
// 1: AP1-ACES
// 2: sRGB/Rec.709-D65
// 3: Rec.2020-D65
// 4: P3-D65
// 5: P3-DCI
const int primaries_out = 3;

// Clamp output values to 0.0 - 1.0
const bool clamp_output = true;

//
// Extra Parameters
//

// Toggle Inverse Transform
const bool invert = false;

const mat3 XYZ_to_LMS_ZCAM = mat3(
     0.41478972f, 0.57999900f, 0.01464800f,
    -0.20151000f, 1.12064900f, 0.05310080f,
    -0.01660080f, 0.26480000f, 0.66847990f
);
const float zcam_rho = 1.7 * 2323.0 / pow(2.0, 5.0);


// Tonescale select
// bool mmTonescaleMode;
// OpenDRT tonescale parameters
const float Lp = sstsLuminance.z;
const float su = 2.0;
const float c0 = 1.2;
const float cs = pow(0.18, 1.0 - c0);
const float c1 = 1.1;
const float p = c1 * (0.9 + 0.05 * su);
const float w1 = pow(0.595 * Lp / 10000.0, 0.931) + 1.037;
const float s1 = w1 * Lp / 100.0;
const float ex = -0.26;
const float eb = 0.08;
const float e0 = exp2(ex + eb * log2(s1));
const float s0 = pow(s1 / e0, 1.0 / c1);
const float fl = 0.01;
const float dch = 0.55;
const float sat = max(1.0, 1.125 - 0.00025 * Lp);
const float mmScaleFactor = 100.0;

// DanieleCompressionCurve tonescale parameters
const float n = sstsLuminance.z;
const float nr = 100.0;
const float g = 1.1;
const float w = 0.84;
const float t_1 = 0.075;



// constants
const float HALF_MIN = 0.0000000596046448;
const float HALF_MAX = 65504.0;

// ZCAM vars
const float zcam_L_A = referenceLuminance * backgroundLuminance / 100.0;
const float zcam_F_b = sqrt(backgroundLuminance / referenceLuminance);
const float zcam_F_L = 0.171 * spow(zcam_L_A, rcp(3.0)) * (1.0 - exp(-48.0 / 9.0 * zcam_L_A));
const float zcam_cb = 1.15;
const float zcam_cg = 0.66;
const float zcam_c1 = 3424.0 / spow(2.0, 12.0);
const float zcam_c2 = 2413.0 / spow(2.0, 7.0);
const float zcam_c3 = 2392.0 / spow(2.0, 7.0);
const float zcam_eta = 2610.0 / spow(2.0, 14.0);
const float zcam_luminance_shift = 1.0 / (-0.20151000 + 1.12064900 + 0.05310080);
const float zcam_viewing_conditions_coeff = viewingConditions == 0 ? 0.525 :
                                            viewingConditions == 1 ? 0.59  :
                                            viewingConditions == 2 ? 0.69  : 1.0;

// CAT vars
const float cat_adaptDegree = discountIlluminant ? 1.0 : (viewingConditionsCoeff * (1.0 - (1.0 / 3.6) * exp((-zcam_L_A - 42.0) / 92.0)));



// ST2084 vars
const float st2084_m_1 = 2610.0 / 4096.0 * (1.0 / 4.0);
const float st2084_m_2 = 2523.0 / 4096.0 * 128.0;
const float st2084_c_1 = 3424.0 / 4096.0;
const float st2084_c_2 = 2413.0 / 4096.0 * 32.0;
const float st2084_c_3 = 2392.0 / 4096.0 * 32.0;
const float st2084_m_1_d = 1.0 / st2084_m_1;
const float st2084_m_2_d = 1.0 / st2084_m_2;
const float st2084_L_p = 10000.0;

// SSTS constants
const float ssts_min_stop_sdr =   -6.5;
const float ssts_max_stop_sdr =    6.5;
const float ssts_min_stop_rrt =  -15.0;
const float ssts_max_stop_rrt =   18.0;
const float ssts_min_lum_sdr  =    0.02;
const float ssts_max_lum_sdr  =   48.0;
const float ssts_min_lum_rrt  =    0.0001;
const float ssts_max_lum_rrt = 10000.0;
const int ssts_n_knots_low = 4;
const int ssts_n_knots_high = 4;
const mat3 ssts_m1 = mat3(
     0.5, -1.0,  0.5,
    -1.0,  1.0,  0.0,
     0.5,  0.5,  0.0
);

// SSTS tables
// using the vec4 type to store the two 2D vectors
const vec4 ssts_minTable = vec4(log10(ssts_min_lum_rrt), ssts_min_stop_rrt, log10(ssts_min_lum_sdr), ssts_min_stop_sdr);
const vec4 ssts_maxTable = vec4(log10(ssts_max_lum_sdr), ssts_max_stop_sdr, log10(ssts_max_lum_rrt), ssts_max_stop_rrt);
const vec4 ssts_bendsLow = vec4(ssts_min_stop_rrt, 0.18, ssts_min_stop_sdr, 0.35);
const vec4 ssts_bendsHigh = vec4(ssts_max_stop_sdr, 0.89, ssts_max_stop_rrt, 0.90);


#include "zcam_drt_table.glsl"




// m = (t.w-t.y)/(t.z-t.x)
// c = (t.y - ((t.w-t.y)/(t.z-t.x) * t.x))
// y = f * (t.w-t.y)/(t.z-t.x) + (t.y - ((t.w-t.y)/(t.z-t.x) * t.x))
#define lerp1D(t, f) ((f) * ((t).w - (t).y) / ((t).z - (t).x) + ((t).y - (((t).w - (t).y) / ((t).z - (t).x) * (t).x)))



// SSTS parameters
const vec3 ssts_min_pt = vec3(
    0.18 * spow(2.0, lerp1D(ssts_minTable, log10(sstsLuminance.x))),
    sstsLuminance.x,
    0.0
);
const vec3 ssts_mid_pt = vec3(0.18, 4.8, 1.55);
const vec3 ssts_max_pt = vec3(
    0.18f * spow(2.0, lerp1D(ssts_maxTable, log10(sstsLuminance.z))),
    sstsLuminance.z,
    0.0
);
const float ssts_knotIncLow = (log10(ssts_mid_pt.x) - log10(ssts_min_pt.x)) / 3.0;
const float ssts_knotIncHigh = (log10(ssts_max_pt.x) - log10(ssts_mid_pt.x)) / 3.0;
const float ssts_pctLow = lerp1D(ssts_bendsLow,  log2(ssts_min_pt.x / 0.18));
const float ssts_pctHigh = lerp1D(ssts_bendsHigh, log2(ssts_max_pt.x / 0.18));


// using the mat3x3 type to store the array of 6 coefficients
const mat3 ssts_coefsLow = mat3(
    (ssts_min_pt.z * (log10(ssts_min_pt.x) - 0.5 * ssts_knotIncLow)) + (log10(ssts_min_pt.y) - ssts_min_pt.z * log10(ssts_min_pt.x)),
    (ssts_min_pt.z * (log10(ssts_min_pt.x) + 0.5 * ssts_knotIncLow)) + (log10(ssts_min_pt.y) - ssts_min_pt.z * log10(ssts_min_pt.x)),
    log10(ssts_min_pt.y) + ssts_pctLow * (log10(ssts_mid_pt.y) - log10(ssts_min_pt.y)),
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x) - 0.5 * ssts_knotIncLow)) + (log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x) + 0.5 * ssts_knotIncLow)) + (log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x) + 0.5 * ssts_knotIncLow)) + (log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    0.0, 0.0, 0.0
);
const mat3 ssts_coefsHigh = mat3(
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x) - 0.5 * ssts_knotIncHigh)) + (log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x) + 0.5 * ssts_knotIncHigh)) + (log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    log10(ssts_mid_pt.y) + ssts_pctHigh * (log10(ssts_max_pt.y)-log10(ssts_mid_pt.y)),
    (ssts_max_pt.z * (log10(ssts_max_pt.x) - 0.5 * ssts_knotIncHigh)) + (log10(ssts_max_pt.y) - ssts_max_pt.z * log10(ssts_max_pt.x)),
    (ssts_max_pt.z * (log10(ssts_max_pt.x) + 0.5 * ssts_knotIncHigh)) + (log10(ssts_max_pt.y) - ssts_max_pt.z * log10(ssts_max_pt.x)),
    (ssts_max_pt.z * (log10(ssts_max_pt.x) + 0.5 * ssts_knotIncHigh)) + (log10(ssts_max_pt.y) - ssts_max_pt.z * log10(ssts_max_pt.x)),
    0.0, 0.0, 0.0
);


// matrix vars
const mat3 identity_matrix = mat3(1.0);
const mat3 XYZ_to_LMS_Bradford = mat3(
     0.8951f, 0.2664f,-0.1614f,
    -0.7502f, 1.7135f, 0.0367f,
     0.0389f,-0.0685f, 1.0296f
);
const mat3 XYZ_to_LMS_CAT02 = mat3(
     0.7328f, 0.4296f,-0.1624f,
    -0.7036f, 1.6975f, 0.0061f,
     0.0030f, 0.0136f, 0.9834f
);
const mat3 LMS_to_Izazbz = mat3(
    0.000000f, 1.0001f-eps, 0.000000f,
    3.524000f,-4.066708f, 0.542708f,
    0.199076f, 1.096799f,-1.295875f
);

const mat3 XYZ_to_RGB_input = xyz_to_rec2020;
const mat3 XYZ_to_RGB_limit = primaries_limit == 2 ? xyz_to_rec709 : xyz_to_rec2020;
const mat3 XYZ_to_RGB_output = xyz_to_rec2020;

const mat3 RGB_to_XYZ_input = rec2020_to_xyz;
const mat3 RGB_to_XYZ_limit = primaries_limit == 2 ? rec709_to_xyz : rec2020_to_xyz;
const mat3 RGB_to_XYZ_output = rec2020_to_xyz;

// white points
const vec3 d65White = vec3(1.0) * rec709_to_xyz;
const vec3 inWhite = vec3(1.0) * rec2020_to_xyz;
const vec3 refWhite = vec3(1.0) * rec2020_to_xyz;

// the maximum RGB value of the limiting gamut
const float boundaryRGB = sstsLuminance.z / referenceLuminance;

// the 1D LUT used for quickly findig the approximate limiting gamut cusp JMh coordinates
// the samples are spaced by HSV hue increments of the limiting RGB gamut
// so to find the correct entry for a given ZCAM hue (h) value
// one must search the table entries for the matching entry.z component
const int gamutCuspTableSize = 360;

// local version of the public focusDistance parameter
// this one will be clamped to a value > 0.0
// ensure positive, non-zero focus depth
// to avoid the gamut boundary search vector becoming zero for achromatic colors
// which will cause the boundary search loop to continue forever
const float focusDistanceClamped = max(0.01, focusDistance);



// "PowerP" compression function (also used in the ACES Reference Gamut Compression transform)
// values of v above  'treshold' are compressed by a 'power' function
// so that an input value of 'limit' results in an output of 1.0
float compressPowerP(float v, float threshold, float limit, float power, bool inverse) {
    float s = (limit - threshold) / pow(pow((1.0 - threshold) / (limit - threshold), -power) - 1.0, 1.0 / power);

    float vCompressed;

    if (inverse) {
        vCompressed = (v < threshold || limit < 1.0001 || v > threshold + s)
            ? v
            : threshold + s * pow(-(pow((v - threshold) / s, power) / (pow((v - threshold) / s, power) - 1.0)), 1.0 / power);
    } else {
        vCompressed = (v < threshold || limit < 1.0001)
            ? v
            : threshold + s * ((v - threshold) / s) / (pow(1.0 + pow((v - threshold) / s,power), 1.0 / power));
    }

    return vCompressed;
}

// Two-Stage chromatic adaptation transforms as proposed by Zhai, Q., & Luo, M. R. (2018)
// https://opg.optica.org/oe/fulltext.cfm?uri=oe-26-6-7724
// https://github.com/colour-science/colour/blob/e5fa0790adcc3e5df5fa42ddf2bb75214c8cf59c/colour/adaptation/zhai2018.py
vec3 CAT_Zhai2018(vec3 XYZ_b, vec3 XYZ_wb, vec3 XYZ_wd, float D_b, float D_d, mat3 M) {
    vec3 XYZ_wo = vec3(100.0);
    vec3 RGB_b = XYZ_b * M; //chckm1
    vec3 RGB_wb = XYZ_wb * M; //chckm1
    vec3 RGB_wd = XYZ_wd * M; //chckm1
    vec3 RGB_wo = XYZ_wo * M; //chckm1

    vec3 D_RGB_b = D_b * (XYZ_wb.y / XYZ_wo.y) * (RGB_wo / RGB_wb) + 1.0 - D_b;
    vec3 D_RGB_d = D_d * (XYZ_wd.y / XYZ_wo.y) * (RGB_wo / RGB_wd) + 1.0 - D_d;
    vec3 D_RGB = D_RGB_b / D_RGB_d;

    vec3 RGB_d = D_RGB * RGB_b;
    vec3 XYZ_d = RGB_d * inverse(M); //chckm1

    return XYZ_d;
}

// apply chromatic adaptation transform to 'XYZ' from 'XYZ_ws' to 'XYZ_wd' white points
// 'type' selects the cone fundamentals matrix (except for Zhai2018 which uses a 2-stage tranforms based on CATO2 fundamentals)
// 'adaptDegree' sets the degree of adaptation for the Zhai2018 model
vec3 apply_CAT(vec3 XYZ, vec3 XYZ_ws, vec3 XYZ_wd, int type, float adaptDegree) {
    mat3 XYZ_to_LMS;

    if (type == 1) {
        // XYZ Scaling
        XYZ_to_LMS = mat3(1.0);
    } else if (type == 2) {
        // Bradford
        XYZ_to_LMS = XYZ_to_LMS_Bradford;
    } else if (type == 3) {
        // CAT02
        XYZ_to_LMS = XYZ_to_LMS_CAT02;
    } else if (type == 4) {
        // Zhai2018
        return CAT_Zhai2018(XYZ, XYZ_ws, XYZ_wd, adaptDegree, adaptDegree, XYZ_to_LMS_CAT02);
    } else {
        // None
        return XYZ;
    }

    vec3 LMS_ws = XYZ_ws * XYZ_to_LMS; //chckm1
    vec3 LMS_wd = XYZ_wd * XYZ_to_LMS; //chckm1

    // if (LMS_ws.x == 0.0f)
    // {
    //   LMS_ws.x = 0.000001f;
    // }
    // if (LMS_ws.y == 0.0f)
    // {
    //   LMS_ws.y = 0.000001f;
    // }
    // if (LMS_ws.z == 0.0f)
    // {
    //   LMS_ws.z = 0.000001f;
    // }

    mat3 Mscale = mat3(1.0);
    Mscale[0][0] = LMS_wd.x / LMS_ws.x;
    Mscale[1][1] = LMS_wd.y / LMS_ws.y;
    Mscale[2][2] = LMS_wd.z / LMS_ws.z;

    mat3 M = XYZ_to_LMS * Mscale * inverse(XYZ_to_LMS); //chckm2

    return XYZ * M; //chckm1
}


// convert XYZ tristimulus values to the ZCAM intermediate Izazbz colorspace
vec3 XYZ_to_Izazbz(vec3 XYZD65) {
    vec3 XYZpD65 = XYZD65;
    XYZpD65.x = zcam_cb * XYZD65.x - (zcam_cb - 1.0) * XYZD65.z;
    XYZpD65.y = zcam_cg * XYZD65.y - (zcam_cg - 1.0) * XYZD65.x;
    vec3 LMS = XYZpD65 * XYZ_to_LMS_ZCAM; //chckm1
    vec3 LMSp = vec3(0.0);
    LMSp.x = spow((zcam_c1 + zcam_c2 * spow((LMS.x/10000.0), zcam_eta)) / (1.0 + zcam_c3 * spow((LMS.x/10000.0), zcam_eta)), zcam_rho);
    LMSp.y = spow((zcam_c1 + zcam_c2 * spow((LMS.y/10000.0), zcam_eta)) / (1.0 + zcam_c3 * spow((LMS.y/10000.0), zcam_eta)), zcam_rho);
    LMSp.z = spow((zcam_c1 + zcam_c2 * spow((LMS.z/10000.0), zcam_eta)) / (1.0 + zcam_c3 * spow((LMS.z/10000.0), zcam_eta)), zcam_rho);
    vec3 Izazbz = LMSp * LMS_to_Izazbz; //chckm1
    // return vec3(LMS_to_Izazbz[0][0], LMS_to_Izazbz[0][1], LMS_to_Izazbz[0][2]);
    return Izazbz;
}

// convert the ZCAM intermediate Izazbz colorspace to XYZ tristimulus values
vec3 Izazbz_to_XYZ(vec3 Izazbz) {
    vec3 LMSp = Izazbz * inverse(LMS_to_Izazbz); //chckm1
    vec3 LMS = vec3(0.0);
    LMS.x = 10000.0 * spow((zcam_c1 - spow(LMSp.x, 1.0 / zcam_rho)) / (zcam_c3 * spow(LMSp.x, 1.0 / zcam_rho) - zcam_c2), 1.0 / zcam_eta);
    LMS.y = 10000.0 * spow((zcam_c1 - spow(LMSp.y, 1.0 / zcam_rho)) / (zcam_c3 * spow(LMSp.y, 1.0 / zcam_rho) - zcam_c2), 1.0 / zcam_eta);
    LMS.z = 10000.0 * spow((zcam_c1 - spow(LMSp.z, 1.0 / zcam_rho)) / (zcam_c3 * spow(LMSp.z, 1.0 / zcam_rho) - zcam_c2), 1.0 / zcam_eta);
    vec3 XYZpD65 = LMS * inverse(XYZ_to_LMS_ZCAM); //chckm1
    vec3 XYZD65 = XYZpD65;
    XYZD65.x = (XYZpD65.x + (zcam_cb - 1.0) * XYZpD65.z) / zcam_cb;
    XYZD65.y = (XYZpD65.y + (zcam_cg - 1.0) * XYZD65.x) / zcam_cg;
    return XYZD65;
}

// convert the ZCAM intermediate Izazbz colorspace to the ZCAM J (lightness), M (colorfulness) and h (hue) correlates
// needs the Iz values of the reference white and the viewing conditions parameters
vec3 Izazbz_to_JMh(vec3 Izazbz, float refWhiteIz, int viewingConditions) {
    vec3 JMh = vec3(0.0);
    const float zcam_F_s = zcam_viewing_conditions_coeff;

    JMh.z = mod(degrees(atan(Izazbz.z, Izazbz.y)) + 360.0, 360.0);
    float ez = 1.015 + cos(radians(89.038 + JMh.z));
    const float Qz_exponent = (1.6 * zcam_F_s) / pow(zcam_F_b, 0.12);
    const float Qz_mult = 2700.0 * pow(zcam_F_s, 2.2) * pow(zcam_F_b, 0.5) * pow(zcam_F_L, 0.2);
    float Qz  = spow(Izazbz.x,   Qz_exponent) * Qz_mult;
    float Qzw = spow(refWhiteIz, Qz_exponent) * Qz_mult;
    JMh.x = 100.0 * (Qz / Qzw);
    JMh.y = 100.0 * spow((spow(Izazbz.y, 2.0) + spow(Izazbz.z, 2.0)), 0.37) * ((spow(ez, 0.068) * pow(zcam_F_L, 0.2)) / (pow(zcam_F_b, 0.1) * pow(refWhiteIz, 0.78)));

    return JMh;
    // return vec3(Qz, Qzw, JMh.z);
}

// convert the ZCAM J (lightness), M (colorfulness) and h (hue) correlates to the ZCAM intermediate Izazbz colorspace
// needs the Iz values of the reference white and the viewing conditions parameters
vec3 JMh_to_Izazbz(vec3 JMh, float refWhiteIz, int viewingConditions) {
    const float zcam_F_s = zcam_viewing_conditions_coeff;
    const float Qzm = spow(zcam_F_s, 2.2) * spow(zcam_F_b, 0.5) * spow(zcam_F_L, 0.2);
    float Qzw = 2700.0 * spow(refWhiteIz, (1.6 * zcam_F_s) / spow(zcam_F_b, 0.12)) * Qzm;
    const float Izp = spow(zcam_F_b, 0.12) / (1.6 * zcam_F_s);
    const float Izd = 2700.0 * 100.0 * Qzm;
    float ez = 1.015 + cos(radians(89.038 + JMh.z));
    float hzr = radians(JMh.z);
    float Czp = spow((JMh.y * spow(refWhiteIz, 0.78) * spow(zcam_F_b, 0.1)) / (100.0 * spow(ez, 0.068) * spow(zcam_F_L, 0.2)), 50.0 / 37.0);

    return vec3(spow((JMh.x * Qzw) / Izd, Izp), Czp * cos(hzr), Czp * sin(hzr));
}

// convert XYZ tristimulus values to the ZCAM J (lightness), M (colorfulness) and h (hue) correlates
// needs XYZ tristimulus values for the reference white and a D65 white as well as the viewing conditions as parameters
vec3 XYZ_to_ZCAM_JMh(vec3 XYZ, vec3 refWhite, vec3 d65White, int viewingConditions) {
    vec3 refWhiteIzazbz = XYZ_to_Izazbz(refWhite*referenceLuminance/refWhite.y);
    return Izazbz_to_JMh(XYZ_to_Izazbz(apply_CAT(XYZ, refWhite, d65White, catType, cat_adaptDegree)), refWhiteIzazbz.x, viewingConditions);
}

// convert the ZCAM J (lightness), M (colorfulness) and h (hue) correlates to XYZ tristimulus values
// needs XYZ tristimulus values for the reference white and a D65 white as well as the viewing conditions as parameters
vec3 ZCAM_JMh_to_XYZ(vec3 JMh, vec3 refWhite, vec3 d65White, int viewingConditions) {
    vec3 refWhiteIzazbz = XYZ_to_Izazbz(refWhite * referenceLuminance / refWhite.y);
    return apply_CAT(Izazbz_to_XYZ(JMh_to_Izazbz(JMh, refWhiteIzazbz.x, viewingConditions)), d65White, refWhite, catType, cat_adaptDegree);
}

// check if the 3D point 'v' is inside a cube with the dimensions cubeSize x cubeSize x cubeSize
// the 'smoothing' parameter rounds off the edges and corners of the cube with the exception of the 0,0,0 and cubeSize x cubeSize x cubeSize corners
// a smoothing value of 0.0 applies no smoothing and 1.0 the maximum amount (smoothing values > 1.0 result in undefined behavior)
int isInsideCube(vec3 v, float cubeSize, float smoothing) {
    vec3 normv = v / cubeSize;

    float minv = min_of(normv);
    float maxv = max_of(normv);

    if (smoothing <= 0.0) {
        // when not smoothing we can use a much simpler test
        if (minv < 0.0 || maxv > 1.0) return 0;
        else return 1;
    }

    vec3 clamped = normv;

    float radius = smoothing / 2.0;

    radius = clamp(radius * maxv * (1.0 - minv), 0.0, radius);

    clamped.x = clamp(normv.x, radius, 1.0 - radius);
    clamped.y = clamp(normv.y, radius, 1.0 - radius);
    clamped.z = clamp(normv.z, radius, 1.0 - radius);

    if (length(normv - clamped) > radius) return 0;
    else return 1;
}

// convert ACEScct encoded values to linear
float ACEScct_to_linear(float v) {
    return v > 0.155251141552511 ? spow(2.0, v * 17.52 - 9.72) : (v - 0.0729055341958355) / 10.5402377416545;
}

// encode linear values as ACEScct
float linear_to_ACEScct(float v) {
    return v > 0.0078125 ? (log2(v) + 9.72) / 17.52 : 10.5402377416545 * v + 0.0729055341958355;
}


// convert sRGB gamma encoded values to linear
float sRGB_to_linear(float v) {
    return v < 0.04045 ? v / 12.92 : spow((v + 0.055) / 1.055, 2.4);
}

// encode linear values as sRGB gamma
float linear_to_sRGB(float v) {
    return v <= 0.0031308 ? 12.92 * v : 1.055 * (spow(v, 1.0 / 2.4)) - 0.055;
}

// convert ST2084 PQ encoded values to linear
float ST2084_to_linear(float v) {
    float V_p = spow(v, st2084_m_2_d);
    return spow((max(0.0, V_p - st2084_c_1) / (st2084_c_2 - st2084_c_3 * V_p)), st2084_m_1_d)*st2084_L_p;
}

// encode linear values as ST2084 PQ
float linear_to_ST2084(float v) {
    float Y_p = spow(max(0.0, v) / st2084_L_p, st2084_m_1);
    return spow((st2084_c_1 + st2084_c_2 * Y_p) / (st2084_c_3 * Y_p + 1.0), st2084_m_2);
}

// decode value 'v' with the inverse of the selected encoding fuction to luminance
float encodingToLuminance(int encoding, float v) {
    if (encoding == 1) {
        // ACEScct
        return ACEScct_to_linear(v) * referenceLuminance;
    } else if (encoding == 2) {
        // sRGB
        return sRGB_to_linear(v) * referenceLuminance;
    } else if (encoding == 3) {
        // BT.1886 (Gamma 2.4)
        return spow(v, 2.4) * referenceLuminance;
    } else if (encoding == 4) {
        // Gamma 2.6
        return spow(v, 2.6) * referenceLuminance;
    } else if (encoding == 5) {
        // ST2084
        return ST2084_to_linear(v);
    } else {
        // Linear
        // default
        return v * referenceLuminance;
    }
}

// decode the components of a 3D vector 'v' with the inverse of the selected encoding fuction to luminance
vec3 encodingToLuminance3(int encoding, vec3 v) {
    vec3 lin;
    lin.x = encodingToLuminance(encoding, v.x);
    lin.y = encodingToLuminance(encoding, v.y);
    lin.z = encodingToLuminance(encoding, v.z);

    return lin;
}

// encode the linear luminance value 'v' with the encoding fuction selected by 'encoding'
float luminanceToEncoding(int encoding, float v) {
    if (encoding == 1) {
        // ACEScct
        return linear_to_ACEScct(v / referenceLuminance);
    } else if (encoding == 2) {
        // sRGB
        return linear_to_sRGB(v / referenceLuminance);
    } else if (encoding == 3) {
        // BT.1886 (Gamma 2.4)
        return spow(v / referenceLuminance, 1.0 / 2.4);
    } else if (encoding == 4) {
        // Gamma 2.6
        return spow(v / referenceLuminance, 1.0 / 2.6);
    } else if (encoding == 5) {
        // ST2084
        return linear_to_ST2084(v);
    } else {
        // Linear
        // default
        return v / referenceLuminance;
    }
}

// encode the linear luminance value components of a 3D vector 'v' with the encoding fuction selected by 'encoding'
vec3 luminanceToEncoding3(int encoding, vec3 v) {
    vec3 enc;
    enc.x = luminanceToEncoding(encoding, v.x);
    enc.y = luminanceToEncoding(encoding, v.y);
    enc.z = luminanceToEncoding(encoding, v.z);

    return enc;
}

// convert RGB values in the input colorspace to the ZCAM intermediate Izazbz colorspace
vec3 input_RGB_to_Izazbz(vec3 inputRGB) {
    // clamp input to +/- HALF_MAX range (to remove inf values, etc.)
    inputRGB = clamp(inputRGB, -HALF_MAX, HALF_MAX);

    // convert to linear XYZ luminance values
    vec3 luminanceRGB = encodingToLuminance3(encodingIn, inputRGB);
    vec3 luminanceXYZ = luminanceRGB * RGB_to_XYZ_input; //chckm1

    // assuming 'fully adapted', dark' viewing conditions for input image (does that make sense?)
    return XYZ_to_Izazbz(apply_CAT(luminanceXYZ, inWhite, d65White, catType, 1.0));
    // return apply_CAT(luminanceXYZ, inWhite, d65White, catType, 1.0);
}

// convert values in the ZCAM intermediate Izazbz colorspace to RGB values in the input colorspace
vec3 Izazbz_to_input_RGB(vec3 Izazbz) {
    vec3 luminanceXYZ = Izazbz_to_XYZ(Izazbz);
    luminanceXYZ = apply_CAT(luminanceXYZ, d65White, inWhite, catType, 1.0);
    vec3 luminanceRGB = luminanceXYZ * XYZ_to_RGB_input; //chckm1
    vec3 RGB = luminanceToEncoding3(encodingIn, luminanceRGB);
    return RGB;
}

// convert RGB values in the output colorspace to the ZCAM J (lightness), M (colorfulness) and h (hue) correlates
vec3 output_RGB_to_JMh(vec3 RGB) {
    vec3 luminanceRGB = encodingToLuminance3(encoding_out, RGB);
    vec3 XYZ = luminanceRGB * RGB_to_XYZ_output; //chckm1
    vec3 JMh = XYZ_to_ZCAM_JMh(XYZ, refWhite, d65White, viewingConditions);
    return JMh;
}

// convert ZCAM J (lightness), M (colorfulness) and h (hue) correlates to  RGB values in the output colorspace
vec3 JMh_to_output_RGB(vec3 JMh) {
    vec3 luminanceXYZ = ZCAM_JMh_to_XYZ(JMh, refWhite, d65White, viewingConditions);
    vec3 luminanceRGB = luminanceXYZ * XYZ_to_RGB_output; //chckm1
    vec3 outputRGB = luminanceToEncoding3(encoding_out, luminanceRGB);

    if (clamp_output) outputRGB = clamp01(outputRGB);

    return outputRGB;
}

// convert linear RGB values with the limiting primaries to ZCAM J (lightness), M (colorfulness) and h (hue) correlates
vec3 limit_RGB_to_JMh(vec3 RGB) {
    vec3 luminanceRGB = RGB * boundaryRGB * referenceLuminance;
    vec3 XYZ = luminanceRGB * RGB_to_XYZ_limit; //chckm1
    vec3 JMh = XYZ_to_ZCAM_JMh(XYZ, refWhite, d65White, viewingConditions);
    return JMh;
}

// convert ZCAM J (lightness), M (colorfulness) and h (hue) correlates to linear RGB values with the limiting primaries
vec3 JMh_to_limit_RGB(vec3 JMh) {
    vec3 luminanceXYZ = ZCAM_JMh_to_XYZ(JMh, refWhite, d65White, viewingConditions);
    vec3 luminanceRGB = luminanceXYZ * XYZ_to_RGB_output; //chckm1
    vec3 RGB = luminanceRGB / boundaryRGB / referenceLuminance;
    return RGB;
}


// convert HSV cylindrical projection values to RGB
vec3 HSV_to_RGB(vec3 HSV) {
    float C = HSV.z * HSV.y;
    float X = C * (1.0 - abs(mod(HSV.x * 6.0, 2.0) - 1.0));
    float m = HSV.z - C;

    vec3 RGB;
    RGB.x = (HSV.x<1.0/6.0?  C :HSV.x<2.0/6.0?  X :HSV.x<3.0/6.0?0.0 :HSV.x<4.0/6.0?0.0 :HSV.x<5.0/6.0?  X :  C) + m;
    RGB.y = (HSV.x<1.0/6.0?  X :HSV.x<2.0/6.0?  C :HSV.x<3.0/6.0?  C :HSV.x<4.0/6.0?  X :HSV.x<5.0/6.0?0.0 :0.0) + m;
    RGB.z = (HSV.x<1.0/6.0?0.0 :HSV.x<2.0/6.0?0.0 :HSV.x<3.0/6.0?  X :HSV.x<4.0/6.0?  C :HSV.x<5.0/6.0?  C :  X) + m;
    return RGB;
}

// convert RGB to HSV cylindrical projection values
vec3 RGB_to_HSV(vec3 RGB) {
    float cmax = max_of(RGB);
    float cmin = min_of(RGB);
    float delta = cmax - cmin;

    vec3 HSV;
    HSV.x = delta == 0.0 ? 0.0
           : cmax == RGB.x ? (mod((RGB.y - RGB.z) / delta + 6.0, 6.0)) / 6.0
           : cmax == RGB.y ? (((RGB.z - RGB.x) / delta + 2.0) / 6.0)
           :                 (((RGB.x - RGB.y) / delta + 4.0) / 6.0);

    HSV.y = cmax == 0.0 ? 0.0 : delta / cmax;
    HSV.z = cmax;
    return HSV;
}


// retrieve the JM coordinates of the limiting gamut cusp at the hue slice 'h'
// cusps are very expensive to compute
// and the DRT is only using them for lightness mapping
// which does not require a high degree of accuracy
// so instead we use a pre-computed table of cusp points
// sampled at 1 degree hue intervals of the the RGB target gamut
// and lerp between them to get the approximate J & M values
vec2 cuspFromTable(float h, vec3[gamutCuspTableSize] gamutCuspTable) {
    vec3 lo;
    vec3 hi;

    if (h <= gamutCuspTable[0].z) {
        lo = gamutCuspTable[gamutCuspTableSize-1];
        lo.z = lo.z - 360.0;
        hi = gamutCuspTable[0];
    } else if (h >= gamutCuspTable[gamutCuspTableSize-1].z) {
        lo = gamutCuspTable[gamutCuspTableSize-1];
        hi = gamutCuspTable[0];
        hi.z = hi.z + 360.0;
    } else {
        for (int i = 1; i < gamutCuspTableSize; ++i) {
            if (h <= gamutCuspTable[i].z) {
                lo = gamutCuspTable[i-1];
                hi = gamutCuspTable[i];
                break;
            }
        }
    }

    float t = (h - lo.z) / (hi.z - lo.z);

    float cuspJ = mix(lo.x, hi.x, t);
    float cuspM = mix(lo.y, hi.y, t);

    return vec2(cuspJ, cuspM);
}


// find the JM coordinates of the smoothed boundary of the limiting gamut in ZCAM at the hue slice 'h'
// by searching along the line defined by 'JMSource' and 'JMFocus'
// the function will search outwards from where the line intersects the achromatic axis with a staring incement of 'startStepSize'
// once the boundary has been crossed it will search in the opposite direction with half the step size
// and will repeat this as as many times as is set by the 'prec' (precision) paramter
vec2 findBoundary(
    vec2 JMSource,
    vec2 JMFocus,
    float h,
    vec3 XYZw,
    vec3 XYZd65,
    mat3 XYZ_to_RGB,
    float smoothing,
    int prec,
    float startStepSize,
    float limitJmax,
    float limitMmax
) {

    vec2 achromaticIntercept = vec2(JMFocus.x - (((JMSource.x-JMFocus.x) / (JMSource.y-JMFocus.y)) * JMFocus.y), 0.0);

    if (achromaticIntercept.x <= 0.0 || achromaticIntercept.x >= limitJmax) return achromaticIntercept;

    float stepSize = startStepSize;
    vec2 unitVector = normalize(achromaticIntercept - JMFocus);
    vec2 JMtest = achromaticIntercept;
    int searchOutwards = 1;

    for (int i = 0; i < prec; ++i) {
        for (int j = 0; j < boundaryWhileMax; ++j) {
            JMtest += unitVector * stepSize;
            int inside = isInsideCube(
                (ZCAM_JMh_to_XYZ(vec3(JMtest.x, JMtest.y, h), XYZw, XYZd65, viewingConditions) / referenceLuminance) * XYZ_to_RGB, //chckm1
                boundaryRGB,
                smoothing
            );

            if (searchOutwards == 1) {
                if (JMtest.x < 0.0 || JMtest.x > limitJmax || JMtest.y > limitMmax || inside != 1) {
                    searchOutwards = 0;
                    stepSize = -abs(stepSize) / 2.0;
                    break;
                }
            } else {
                if (JMtest.y < 0.0 || inside == 1) {
                    searchOutwards = 1;
                    stepSize = abs(stepSize) / 2.0;
                    break;
                }
            }
        }
    }

    vec2 JMboundary = vec2(clamp(JMtest.x, 0.0, limitJmax), clamp(JMtest.y, 0.0, limitMmax));

    return JMboundary;
}

// apply the forward ACES SingleStageToneScale (SSTS) transform to the linear 'x' input value and return a luminance value
float forwardSSTS(float x, vec3 minPt, vec3 midPt, vec3 maxPt) {
    // Check for negatives or zero before taking the log. If negative or zero,
    // set to HALF_MIN.
    float logx = log10(max(x, HALF_MIN));

    float logy;

    if (logx <= log10(minPt.x)) {
        logy = logx * minPt.z + (log10(minPt.y) - minPt.z * log10(minPt.x));
    } else if ((logx > log10(minPt.x)) && (logx < log10(midPt.x))) {
        float knot_coord = 3.0 * (logx - log10(minPt.x)) / (log10(midPt.x) - log10(minPt.x));
        int j = int(knot_coord);
        float t = knot_coord - float(j);
        vec3 cf = vec3(ssts_coefsLow[j / 3][j % 3], ssts_coefsLow[(j + 1) / 3][(j + 1) % 3], ssts_coefsLow[(j + 2) / 3][(j + 2) % 3]);
        vec3 monomials = vec3(t * t, t, 1.0);
        logy = dot(monomials, cf * ssts_m1); //chckm1
    } else if ((logx >= log10(midPt.x)) && (logx < log10(maxPt.x))) {
        float knot_coord = 3.0 * (logx-log10(midPt.x)) / (log10(maxPt.x) - log10(midPt.x));
        int j = int(knot_coord);
        float t = knot_coord - float(j);
        vec3 cf = vec3(ssts_coefsHigh[j / 3][j % 3], ssts_coefsHigh[(j + 1) / 3][(j + 1) % 3], ssts_coefsHigh[(j + 2) / 3][(j + 2) % 3]);
        vec3 monomials = vec3(t * t, t, 1.0);
        logy = dot(monomials, cf * ssts_m1); //chckm1
    } else {
        logy = logx * maxPt.z + (log10(maxPt.y) - maxPt.z * log10(maxPt.x));
    }

    // Fix NaNs in bright spots
    logy = min(logy, 10.0 - HALF_MIN);

    return spow(10.0, logy);
}

// Michalis Menton Dual Spring Curve
float forwardMmTonescale(float x) {
    float tc = 0.0;
    if (x < 0.18) tc = cs * spow(x, c0);
    else        tc = c0 * (x - 0.18) + 0.18;

    float ts = s1 * spow((tc / (s0 + tc)), p);
    float tf = ts * ts / (ts + fl);
    float ccf = spow(s0 / (x + s0), dch) * sat;

    return tf;
}

// Daniele's Compression Curve
// https://www.desmos.com/calculator/fihdxfot6s
float forwardDanieleCompressionCurve(float x) {
    float m0 = n / nr;
    float m  = 0.5 * (m0 + sqrt(m0 * (m0 + 4.0 * t_1)));
    float s_1 = w * pow(m, rcp(g));

    // Scale Data
    //   x = x / n;

    // Ref Version
    float f = pow(((max(0.0, x)) / (x + s_1)), g) * m;
    float h = max(0.0,((pow(f, 2.0)) / (f + t_1)));

    //  Scale Data
    //   h = h * n;

    return h;
}

// apply the inverse ACES SingleStageToneScale (SSTS) transfomr to the 'x' luminance value and return an linear value
float inverseSSTS(float y, vec3 minPt, vec3 midPt, vec3 maxPt) {
    float KNOT_INC_LOW  = (log10(midPt.x) - log10(minPt.x)) / 3.0;
    float KNOT_INC_HIGH = (log10(maxPt.x) - log10(midPt.x)) / 3.0;

    // KNOT_Y is luminance of the spline at each knot
    float KNOT_Y_LOW[4];

    for (int i = 0; i < 4; i++) {
        KNOT_Y_LOW[i] = (ssts_coefsLow[i/3][i%3] + ssts_coefsLow[(i+1)/3][(i+1)%3]) / 2.0;
    }

    float KNOT_Y_HIGH[4];

    for (int i = 0; i < 4; i++) {
        KNOT_Y_HIGH[i] = (ssts_coefsHigh[i/3][i%3] + ssts_coefsHigh[(i+1)/3][(i+1)%3]) / 2.0;
    }

    float logy = log10(max(y, 0.0000000001));

    float logx;

    if (logy <= log10(minPt.y)) {
        logx = log10(minPt.x);
    } else if ((logy > log10(minPt.y)) && (logy <= log10(midPt.y))) {
        int j;
        vec3 cf = vec3(0.0);

        if (logy > KNOT_Y_LOW[0] && logy <= KNOT_Y_LOW[1]) {
            cf.x = ssts_coefsLow[0][0];
            cf.y = ssts_coefsLow[0][1];
            cf.z = ssts_coefsLow[0][2];
            j = 0;
        } else if (logy > KNOT_Y_LOW[1] && logy <= KNOT_Y_LOW[2]) {
            cf.x = ssts_coefsLow[0][1];
            cf.y = ssts_coefsLow[0][2];
            cf.z = ssts_coefsLow[1][0];
            j = 1;
        } else if (logy > KNOT_Y_LOW[2] && logy <= KNOT_Y_LOW[3]) {
            cf.x = ssts_coefsLow[0][2];
            cf.y = ssts_coefsLow[1][0];
            cf.z = ssts_coefsLow[1][1];
            j = 2;
        }

        vec3 tmp = cf * ssts_m1; // chckm1

        float a = tmp.x;
        float b = tmp.y;
        float c = tmp.z;
        c = c - logy;

        float d = sqrt(b * b - 4.0 * a * c);

        float t = (2.0 * c) / (-d - b);

        logx = log10(minPt.x) + (t + j) * KNOT_INC_LOW;

    } else if ((logy > log10(midPt.y)) && (logy < log10(maxPt.y))) {
        int j;
        vec3 cf = vec3(0.0);

        if (logy >= KNOT_Y_HIGH[0] && logy <= KNOT_Y_HIGH[1]) {
            cf.x = ssts_coefsHigh[0][0];
            cf.y = ssts_coefsHigh[0][1];
            cf.z = ssts_coefsHigh[0][2];
            j = 0;
        } else if (logy > KNOT_Y_HIGH[1] && logy <= KNOT_Y_HIGH[2]) {
            cf.x = ssts_coefsHigh[0][1];
            cf.y = ssts_coefsHigh[0][2];
            cf.z = ssts_coefsHigh[1][0];
            j = 1;
        } else if (logy > KNOT_Y_HIGH[2] && logy <= KNOT_Y_HIGH[3]) {
            cf.x = ssts_coefsHigh[0][2];
            cf.y = ssts_coefsHigh[1][0];
            cf.z = ssts_coefsHigh[1][1];
            j = 2;
        }

        vec3 tmp = cf * ssts_m1; // chckm1

        float a = tmp.x;
        float b = tmp.y;
        float c = tmp.z;
        c = c - logy;

        float d = sqrt(b * b - 4.0 * a * c);

        float t = (2.0 * c) / (-d - b);

        logx = log10(midPt.x) + (t + j) * KNOT_INC_HIGH;

    } else {
        logx = log10(maxPt.x);
    }

    return spow(10.0, logx);
}

// convert Iz to luminance
// note that the PQ fuction used for Iz differs from the ST2084 function by replacing m_2 with rho
// it also includes a luminance shift caused by the 2nd row-sum of the XYZ to LMS matrix not adding up to 1.0
float IzToLuminance(float Iz) {
    float V_p = spow(Iz, 1.0 / zcam_rho);
    float luminance = spow((max(0.0, V_p - st2084_c_1) / (st2084_c_2 - st2084_c_3 * V_p)), st2084_m_1_d)*st2084_L_p * zcam_luminance_shift;
    return luminance;
}

// convert luminance to Iz
// note that the PQ fuction used for Iz differs from the ST2084 function by replacing m_2 with rho
// it also includes a luminance shift caused by the 2nd row-sum of the XYZ to LMS matrix not adding up to 1.0
float luminanceToIz(float luminance) {
    float Y_p = spow((luminance/zcam_luminance_shift) / st2084_L_p, st2084_m_1);
    float Iz = spow((st2084_c_1 + st2084_c_2 * Y_p) / (st2084_c_3 * Y_p + 1.0), zcam_rho);
    return Iz;
}

// calculate a scale factor for colorfulness
// based on the difference between the original and tone scaled (TS) Iz values
// we are only interested in the differences above mid grey
// so we first offset the original Iz values to align 18% it with the mid point of the IzTS value
float highlightDesatFactor(float Iz, float IzTS) {
    float linear = IzToLuminance(Iz) / referenceLuminance;

    // no highlight desat below SSTS mid point
    if (linear < 0.18) return 1.0;

    float IzMid   = luminanceToIz(0.18 * referenceLuminance);
    float IzMidTS = luminanceToIz(sstsLuminance.y);

    float IzAligned = Iz + IzMidTS - IzMid;

    float desatFactor = 1.0 - clamp01(compressPowerP(
        (log10(max(HALF_MIN, IzAligned)) - log10(max(HALF_MIN, IzTS))) * desatHighlights,
        compressionFuncParams.x,
        HALF_MAX,
        compressionFuncParams.z,
        false
    ));

    return desatFactor;
}

vec3 forwardTonescale(vec3 inputIzazbz, mat3 ssts_params) {
    vec3 refWhiteIzazbz = XYZ_to_Izazbz(refWhite*referenceLuminance/refWhite.y);

    if (!applyTonecurve && !applyHighlightDesat) {
        // nothing to do here except converting to JMh
        return Izazbz_to_JMh(inputIzazbz, refWhiteIzazbz.x, 0);
    }

    float linear = IzToLuminance(inputIzazbz.x) / referenceLuminance;

    float luminanceTS = 50.0;

    // switch for applying the different tonescale compression functions
    if (toneScaleMode == 1) {
        luminanceTS = forwardMmTonescale(linear) * mmScaleFactor;
    } else if (toneScaleMode == 2) {
        luminanceTS = forwardDanieleCompressionCurve(linear) * mmScaleFactor;
    } else {
        luminanceTS = forwardSSTS(linear, ssts_params[0], ssts_params[1], ssts_params[2]);
    }

    float IzTS = luminanceToIz(luminanceTS);

    vec3 outputIzazbz = inputIzazbz;

    if (applyTonecurve) outputIzazbz.x = IzTS;

    // convert the result to JMh
    vec3 outputJMh = Izazbz_to_JMh(outputIzazbz, refWhiteIzazbz.x, 0);

    if (applyHighlightDesat) {
        float factM = highlightDesatFactor(inputIzazbz.x, IzTS);
        outputJMh.y = outputJMh.y * factM;
    }

    return outputJMh;
}

vec3 compressGamut(vec3 inputJMh, bool invert, float distanceGainCalcJ, vec3[gamutCuspTableSize] gamutCuspTable, float limitJmax, float limitMmax) {
    if (!applyGamutCompression) return inputJMh;

    float sstsMidJ = XYZ_to_ZCAM_JMh(refWhite * sstsLuminance.y, refWhite, d65White, viewingConditions).x;
    vec2 JMinput = vec2(inputJMh.x, inputJMh.y);
    vec2 JMcusp = cuspFromTable(inputJMh.z, gamutCuspTable);

    float focusJ = mix(JMcusp.x, sstsMidJ, cuspMidBlend);

    float focusDistanceGain = 1.0;

    if (distanceGainCalcJ > focusJ) {
        focusDistanceGain = (limitJmax - focusJ) / max(0.0001, (limitJmax - min(limitJmax, distanceGainCalcJ)));
    } else {
        focusDistanceGain = focusJ / max(0.0001, distanceGainCalcJ);
    }

    vec2 JMfocus = vec2(focusJ, -JMcusp.y * focusDistanceClamped * focusDistanceGain);
    vec2 vecToFocus = JMfocus - JMinput;
    vec2 achromaticIntercept = vec2(JMfocus.x - (((JMinput.x-JMfocus.x) / (JMinput.y-JMfocus.y)) * JMfocus.y), 0.0);

    // to reduce the number of expensive boundary finding iterations needed
    // we taking an educated guess at a good starting step size
    // based on how far the sample is either above or below the gamut cusp
    float cuspToTipRatio;
    if (JMinput.x > JMcusp.x) {
        cuspToTipRatio = (JMinput.x - JMcusp.x) / (limitJmax - JMcusp.x);
    } else {
        cuspToTipRatio = (JMcusp.x - JMinput.x) / (JMcusp.x);
    }

    float startStepSize = mix(JMcusp.y / 3.0, 0.1, cuspToTipRatio);
    vec2 JMboundary = findBoundary(JMinput, JMfocus, inputJMh.z, refWhite, d65White, XYZ_to_RGB_limit, smoothCusps, boundarySolvePrecision, startStepSize, limitJmax, limitMmax);
    float normFact = 1.0 / max(0.0001, length(JMboundary - achromaticIntercept));
    float v = length(JMinput - achromaticIntercept) * normFact;
    float vCompressed = compressPowerP(v, compressionFuncParams.x, compressionFuncParams.y, compressionFuncParams.z, invert);
    vec2 JMcompressed = vec2(0.0);

    // hack to stop nan values after compression
    if (JMinput.y != 0.0) {
        JMcompressed = achromaticIntercept + normalize(JMinput - achromaticIntercept) * vCompressed / normFact;
    } else JMcompressed = JMinput;

    return vec3(JMcompressed.x, JMcompressed.y, inputJMh.z);
}

// apply the forward gamut compression to the limiting primaries
vec3 compressGamutForward(vec3 JMh, vec3[gamutCuspTableSize] gamutCuspTable, float limitJmax, float limitMmax) {
    vec3 JMhcompressed = compressGamut(JMh, false, JMh.x, gamutCuspTable, limitJmax, limitMmax);
    // Hack to deal with weird zero values on output
    // JMhcompressed.x = min(300.0, JMhcompressed.x);
    return JMhcompressed;
}

void zcamdrt_init(
    inout mat3 ssts_params,
    inout vec3[360] gamutCuspTable,
    inout float limitJmax,
    inout float limitMmax
) {

    ssts_params[0] = ssts_min_pt;
    ssts_params[1] = ssts_mid_pt;
    ssts_params[2] = ssts_max_pt;

    float ssts_expShift = log2(inverseSSTS(sstsLuminance.y, ssts_min_pt, ssts_mid_pt, ssts_max_pt)) - log2(0.18);
    ssts_params[0].x = spow(2.0, (log(ssts_params[0].x) / log(2.0) - ssts_expShift));
    ssts_params[1].x = spow(2.0, (log(0.18            ) / log(2.0) - ssts_expShift));
    ssts_params[2].x = spow(2.0, (log(ssts_params[2].x) / log(2.0) - ssts_expShift));

    // the 'gamutCuspTableUnsorted' table is populated
    // in increments of H of the limiting gamut HSV space starting at H=0.0
    // since it is unlikely that HSV.H=0 and JMh.h=0 line up
    // the entries are then wrap-around shifted
    // so that the 'gamutCuspTable' starts with the lowest JMh.h value
    vec3 gamutCuspTableUnsorted[360];

    //
    // solving the RGB cusp from JMh is very expensive
    // instead we go the other way and start with a RGB cusp sweep
    // which is easily calculated by converting via HSV (Hue, 1.0, 1.0)
    // we then convert each cusp to JMh and add them to a table
    //

    for (int i = 0; i < gamutCuspTableSize; ++i) {
        float hNorm = float(i) / gamutCuspTableSize;
        vec3 RGB = HSV_to_RGB(vec3(hNorm, 1.0, 1.0));
        gamutCuspTableUnsorted[i] = limit_RGB_to_JMh(RGB);
    }

    int minhIndex = 0;
    for (int i = 1; i < gamutCuspTableSize; ++i) {
        if (gamutCuspTableUnsorted[i].z < gamutCuspTableUnsorted[minhIndex].z) {
            minhIndex = i;
        }
    }

    for (int i = 0; i < gamutCuspTableSize; ++i) {
        gamutCuspTable[i] = gamutCuspTableUnsorted[(minhIndex+i)%gamutCuspTableSize];
    }

    // calculate the maximum expected J & M values for the given limit gamut
    // these are used as limiting values for the gamut boundary searches

    // limitJmax (asumed to match limitRGB white)
    limitJmax = limit_RGB_to_JMh(vec3(1.0)).x;


    // limitMmax (assumed to coincide with one of the RGBCMY corners of the limitRGB cube)
    vec3 gamutCornersTable[6];
    gamutCornersTable[0] = limit_RGB_to_JMh(vec3(1.0f, 0.0f, 0.0f));
    gamutCornersTable[1] = limit_RGB_to_JMh(vec3(1.0f, 1.0f, 0.0f));
    gamutCornersTable[2] = limit_RGB_to_JMh(vec3(0.0f, 1.0f, 0.0f));
    gamutCornersTable[3] = limit_RGB_to_JMh(vec3(0.0f, 1.0f, 1.0f));
    gamutCornersTable[4] = limit_RGB_to_JMh(vec3(0.0f, 0.0f, 1.0f));
    gamutCornersTable[5] = limit_RGB_to_JMh(vec3(1.0f, 0.0f, 1.0f));

    limitMmax = 0.0;
    for (int i = 0; i < 6; ++i) {
        limitMmax = max(limitMmax, gamutCornersTable[i].y);
    }

}

/*void init() // Precompute this
{
    HALF_MIN = 0.0000000596046448f;
    HALF_MAX = 65504.0f;

    zcam_L_A = referenceLuminance * backgroundLuminance / 100.0f;
    zcam_F_b = sqrt(backgroundLuminance/referenceLuminance);
    zcam_F_L = 0.171f*spow(zcam_L_A, 1.0f/3.0f) * (1.0f-exp(-48.0f/9.0f*zcam_L_A));

    if (discountIlluminant) cat_adaptDegree = 1.0f;
    else {
        float viewingConditionsCoeff = 1.0f;
        if ( viewingConditions == 0 ) viewingConditionsCoeff = 0.8f;
        else if ( viewingConditions == 1 ) viewingConditionsCoeff = 0.9f;
        else if ( viewingConditions == 2 ) viewingConditionsCoeff = 1.0f;

        cat_adaptDegree = viewingConditionsCoeff * (1.0f - (1.0f / 3.6f) * exp((-zcam_L_A - 42.0f) / 92.0f));
    }

    zcam_cb  = 1.15f;
    zcam_cg  = 0.66f;
    zcam_c1  = 3424.0f / spow(2.0f,12.0f);
    zcam_c2  = 2413.0f / spow(2.0f, 7.0f);
    zcam_c3  = 2392.0f / spow(2.0f, 7.0f);
    zcam_eta = 2610.0f / spow(2.0f,14.0f);
    // zcam_rho = 1.7f * 2323.0f / pow(2.0f,5.0f);
    zcam_luminance_shift = 1.0f / (-0.20151000f + 1.12064900f + 0.05310080f);

    zcam_viewing_conditions_coeff = 1.0f;

    if ( viewingConditions == 0 ) zcam_viewing_conditions_coeff = 0.525f;
    else if ( viewingConditions == 1 ) zcam_viewing_conditions_coeff = 0.59f;
    else if ( viewingConditions == 2 ) zcam_viewing_conditions_coeff = 0.69f;

    st2084_m_1=2610.0f / 4096.0f * (1.0f / 4.0f);
    st2084_m_2=2523.0f / 4096.0f * 128.0f;
    st2084_c_1=3424.0f / 4096.0f;
    st2084_c_2=2413.0f / 4096.0f * 32.0f;
    st2084_c_3=2392.0f / 4096.0f * 32.0f;
    st2084_m_1_d = 1.0f / st2084_m_1;
    st2084_m_2_d = 1.0f / st2084_m_2;
    st2084_L_p = 10000.0f;

    ssts_min_stop_sdr =  -6.5f;
    ssts_max_stop_sdr =   6.5f;
    ssts_min_stop_rrt = -15.0f;
    ssts_max_stop_rrt =  18.0f;
    ssts_min_lum_sdr = 0.02f;
    ssts_max_lum_sdr = 48.0f;
    ssts_min_lum_rrt = 0.0001f;
    ssts_max_lum_rrt = 10000.0f;
    ssts_n_knots_low = 4;
    ssts_n_knots_high = 4;

    ssts_minTable = float4(log10(ssts_min_lum_rrt), ssts_min_stop_rrt, log10(ssts_min_lum_sdr), ssts_min_stop_sdr);
    ssts_maxTable = float4(log10(ssts_max_lum_sdr), ssts_max_stop_sdr, log10(ssts_max_lum_rrt), ssts_max_stop_rrt);
    ssts_bendsLow = float4(ssts_min_stop_rrt, 0.18f, ssts_min_stop_sdr, 0.35f);
    ssts_bendsHigh = float4(ssts_max_stop_sdr, 0.89f, ssts_max_stop_rrt, 0.90f);


    ssts_min_pt.x = 0.18f * spow(2.0f, lerp1D(ssts_minTable, log10(sstsLuminance.x)));
    ssts_min_pt.y = sstsLuminance.x;
    ssts_min_pt.z = 0.0f;

    ssts_mid_pt = vec3(0.18f, 4.8f, 1.55f);

    ssts_max_pt.x = 0.18f * spow(2.0f, lerp1D(ssts_maxTable, log10(sstsLuminance.z)));
    ssts_max_pt.y = sstsLuminance.z;
    ssts_max_pt.z = 0.0f;

    ssts_knotIncLow  = (log10(ssts_mid_pt.x) - log10(ssts_min_pt.x)) / 3.0f;
    ssts_knotIncHigh = (log10(ssts_max_pt.x) - log10(ssts_mid_pt.x)) / 3.0f;
    ssts_pctLow  = lerp1D(ssts_bendsLow,  log2(ssts_min_pt.x / 0.18f));
    ssts_pctHigh = lerp1D(ssts_bendsHigh, log2(ssts_max_pt.x / 0.18f));


    float ssts_coefsLow_data[] = {
    (ssts_min_pt.z * (log10(ssts_min_pt.x)-0.5f*ssts_knotIncLow)) + ( log10(ssts_min_pt.y) - ssts_min_pt.z * log10(ssts_min_pt.x)),
    (ssts_min_pt.z * (log10(ssts_min_pt.x)+0.5f*ssts_knotIncLow)) + ( log10(ssts_min_pt.y) - ssts_min_pt.z * log10(ssts_min_pt.x)),
    log10(ssts_min_pt.y) + ssts_pctLow*(log10(ssts_mid_pt.y)-log10(ssts_min_pt.y)),
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x)-0.5f*ssts_knotIncLow)) + ( log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x)+0.5f*ssts_knotIncLow)) + ( log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x)+0.5f*ssts_knotIncLow)) + ( log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    0.0f, 0.0f, 0.0f };

    float sssts_coefsHigh_data[] = {
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x)-0.5f*ssts_knotIncHigh)) + ( log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    (ssts_mid_pt.z * (log10(ssts_mid_pt.x)+0.5f*ssts_knotIncHigh)) + ( log10(ssts_mid_pt.y) - ssts_mid_pt.z * log10(ssts_mid_pt.x)),
    log10(ssts_mid_pt.y) + ssts_pctHigh*(log10(ssts_max_pt.y)-log10(ssts_mid_pt.y)),
    (ssts_max_pt.z * (log10(ssts_max_pt.x)-0.5f*ssts_knotIncHigh)) + ( log10(ssts_max_pt.y) - ssts_max_pt.z * log10(ssts_max_pt.x)),
    (ssts_max_pt.z * (log10(ssts_max_pt.x)+0.5f*ssts_knotIncHigh)) + ( log10(ssts_max_pt.y) - ssts_max_pt.z * log10(ssts_max_pt.x)),
    (ssts_max_pt.z * (log10(ssts_max_pt.x)+0.5f*ssts_knotIncHigh)) + ( log10(ssts_max_pt.y) - ssts_max_pt.z * log10(ssts_max_pt.x)),
    0.0f, 0.0f, 0.0f };

    ssts_coefsLow.setArray(ssts_coefsLow_data);
    ssts_coefsHigh.setArray(sssts_coefsHigh_data);

    ssts_paramMin = ssts_min_pt;
    ssts_paramMid = ssts_mid_pt;
    ssts_paramMax = ssts_max_pt;
    ssts_expShift = log2(inverseSSTS(sstsLuminance.y, ssts_min_pt, ssts_paramMid, ssts_max_pt)) - log2(0.18f);
    ssts_paramMin.x = spow(2.0f, (log(ssts_paramMin.x) / log(2.0f) - ssts_expShift));
    ssts_paramMid.x = spow(2.0f, (log(0.18f          ) / log(2.0f) - ssts_expShift));
    ssts_paramMax.x = spow(2.0f, (log(ssts_paramMax.x) / log(2.0f) - ssts_expShift));



    // Blink does not seem to support initialising multidimensional arrays
    // So instead of being able to index the matrix data directly from one
    // we need to use long if/else statements to populate the
    // input, limit & output primary matrices
    // (maybe there is a better way?)

    /*float XYZ_to_Rec709_D65_matrix_data[]=
    {
    3.2409699419f, -1.5373831776f, -0.4986107603f,
    -0.9692436363f,  1.8759675015f,  0.0415550574f,
    0.0556300797f, -0.2039769589f,  1.0569715142f,
    };

    float XYZ_to_Rec2020_D65_matrix_data[]=
    {
    1.7166511880f, -0.3556707838f, -0.2533662814f,
    -0.6666843518f,  1.6164812366f,  0.0157685458f,
    0.0176398574f, -0.0427706133f,  0.9421031212f,
    };

    float XYZ_to_P3_D65_matrix_data[]=
    {
    2.4934969119f, -0.9313836179f, -0.4027107845f,
    -0.8294889696f,  1.7626640603f,  0.0236246858f,
    0.0358458302f, -0.0761723893f,  0.9568845240f,
    };

    float XYZ_to_P3_DCI_matrix_data[]=
    {
    2.7253940305f, -1.0180030062f, -0.4401631952f,
    -0.7951680258f,  1.6897320548f,  0.0226471906f,
    0.0412418914f, -0.0876390192f,  1.1009293786f
    };

    // populate the input primaries matrix
    if ( primaries_in == 0 ) XYZ_to_RGB_input.setArray(XYZ_to_AP0_ACES_matrix_data);
    else if ( primaries_in == 1 ) XYZ_to_RGB_input.setArray(XYZ_to_AP1_ACES_matrix_data);
    else if ( primaries_in == 2 ) XYZ_to_RGB_input.setArray(XYZ_to_Rec709_D65_matrix_data);
    else if ( primaries_in == 3 ) XYZ_to_RGB_input.setArray(XYZ_to_Rec2020_D65_matrix_data);
    else if ( primaries_in == 4 ) XYZ_to_RGB_input.setArray(XYZ_to_P3_D65_matrix_data);
    else if ( primaries_in == 5 ) XYZ_to_RGB_input.setArray(XYZ_to_P3_DCI_matrix_data);
    else XYZ_to_RGB_input.setArray(identity_matrix_data);

    // populate the limiting primaries matrix
    if ( primaries_limit == 0 ) XYZ_to_RGB_limit.setArray(XYZ_to_AP0_ACES_matrix_data);
    else if ( primaries_limit == 1 ) XYZ_to_RGB_limit.setArray(XYZ_to_AP1_ACES_matrix_data);
    else if ( primaries_limit == 2 ) XYZ_to_RGB_limit.setArray(XYZ_to_Rec709_D65_matrix_data);
    else if ( primaries_limit == 3 ) XYZ_to_RGB_limit.setArray(XYZ_to_Rec2020_D65_matrix_data);
    else if ( primaries_limit == 4 ) XYZ_to_RGB_limit.setArray(XYZ_to_P3_D65_matrix_data);
    else if ( primaries_limit == 5 ) XYZ_to_RGB_limit.setArray(XYZ_to_P3_DCI_matrix_data);
    else XYZ_to_RGB_limit.setArray(identity_matrix_data);

    // populate the output primaries matrix
    if ( primaries_out == 0 ) XYZ_to_RGB_output.setArray(XYZ_to_AP0_ACES_matrix_data);
    else if ( primaries_out == 1 ) XYZ_to_RGB_output.setArray(XYZ_to_AP1_ACES_matrix_data);
    else if ( primaries_out == 2 ) XYZ_to_RGB_output.setArray(XYZ_to_Rec709_D65_matrix_data);
    else if ( primaries_out == 3 ) XYZ_to_RGB_output.setArray(XYZ_to_Rec2020_D65_matrix_data);
    else if ( primaries_out == 4 ) XYZ_to_RGB_output.setArray(XYZ_to_P3_D65_matrix_data);
    else if ( primaries_out == 5 ) XYZ_to_RGB_output.setArray(XYZ_to_P3_DCI_matrix_data);
    else XYZ_to_RGB_output.setArray(identity_matrix_data);

    RGB_to_XYZ_input = XYZ_to_RGB_input.invert();
    RGB_to_XYZ_limit = XYZ_to_RGB_limit.invert();
    RGB_to_XYZ_output = XYZ_to_RGB_output.invert();*//*


    //
    // solving the RGB cusp from JMh is very expensive
    // instead we go the other way and start with a RGB cusp sweep
    // which is easily calculated by converting via HSV (Hue, 1.0, 1.0)
    // we then convert each cusp to JMh and add them to a table
    //

    gamutCuspTableSize = 360;

    for (int i = 0; i < gamutCuspTableSize; ++i ) {
        float hNorm = float(i) / (gamutCuspTableSize);
        vec3 RGB = HSV_to_RGB(vec3(hNorm, 1.0f, 1.0f));
        gamutCuspTableUnsorted[i] = limit_RGB_to_JMh(RGB);
    }

    int minhIndex = 0;
    for (int i = 1; i < gamutCuspTableSize; ++i) {
        if (gamutCuspTableUnsorted[i].z <  gamutCuspTableUnsorted[minhIndex].z) {
            minhIndex = i;
        }
    }


    for ( int i = 0; i < gamutCuspTableSize; ++i ) {
        gamutCuspTable[i] = gamutCuspTableUnsorted[(minhIndex+i)%gamutCuspTableSize];
    }

    // calculate the maximum expected J & M values for the given limit gamut
    // these are used as limiting values for the gamut boundary searches

    // limitJmax (asumed to match limitRGB white)
    limitJmax = limit_RGB_to_JMh(vec3(1.0)).x;


    // limitMmax (assumed to coincide with one of the RGBCMY corners of the limitRGB cube)
    vec3 gamutCornersTable[6];
    gamutCornersTable[0] = limit_RGB_to_JMh(vec3(1.0f, 0.0f, 0.0f));
    gamutCornersTable[1] = limit_RGB_to_JMh(vec3(1.0f, 1.0f, 0.0f));
    gamutCornersTable[2] = limit_RGB_to_JMh(vec3(0.0f, 1.0f, 0.0f));
    gamutCornersTable[3] = limit_RGB_to_JMh(vec3(0.0f, 1.0f, 1.0f));
    gamutCornersTable[4] = limit_RGB_to_JMh(vec3(0.0f, 0.0f, 1.0f));
    gamutCornersTable[5] = limit_RGB_to_JMh(vec3(1.0f, 0.0f, 1.0f));

    limitMmax = 0.0f;
    for (int i = 0; i < 6; ++i) {
        limitMmax = max(limitMmax, gamutCornersTable[i].y);
    }

}*/


vec3 zcamdrtransform(vec3 srcRGB) {
    // [0] = ssts_paramMin, [1] = ssts_paramMid, [2] = ssts_paramMax
    //mat3 ssts_params;
    //vec3[360] gamutCuspTable;
    // the maximum lightness value of the limiting gamut
    //float limitJmax;
    // the maximum colorfulness value of the limiting gamut
    //float limitMmax;

    //zcamdrt_init(ssts_params, gamutCuspTable, limitJmax, limitMmax);

    vec3 dstRGB;

    vec3 inputIzazbz = input_RGB_to_Izazbz(srcRGB);
    vec3 JMh = forwardTonescale(inputIzazbz, ssts_params);
    JMh = compressGamutForward(JMh, gamutCuspTable, limitJmax, limitMmax);
    dstRGB = JMh_to_output_RGB(JMh);
    // dstRGB = Izazbz_to_JMh(inputIzazbz, 0.31334, 0);

    return dstRGB;
}

#endif // INCLUDE_TONEMAPPING_ZCAM_DRT
