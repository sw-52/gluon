#ifndef INCLUDE_TONEMAPPING_ACES_LMT_OTHER
#define INCLUDE_TONEMAPPING_ACES_LMT_OTHER



#include "/include/tonemapping/aces/matrices.glsl"
#include "/include/utility/color.glsl"

const float X_BRK = 0.0078125;
const float Y_BRK = 0.155251141552511;
const float A = 10.5402377416545;
const float B = 0.0729055341958355;
const float sqrt3over4 = 0.433012701892219;

struct Chromaticities {
    vec2 red;
    vec2 green;
    vec2 blue;
    vec2 white;
};

const Chromaticities AP0 = Chromaticities(vec2(0.7347, 0.2653), vec2(0.0, 1.0), vec2(0.0001, -0.077), vec2(0.32168, 0.33767));
const Chromaticities AP1 = Chromaticities(vec2(0.713, 0.293), vec2(0.165, 0.83), vec2(0.128, 0.044), vec2(0.32168, 0.33767));
const Chromaticities REC709_PRI = Chromaticities(vec2(0.64, 0.33), vec2(0.3, 0.6), vec2(0.15, 0.06), vec2(0.3127, 0.329));


float interpolate1D_2(vec2[2] table, float p) {
    if(p <= table[0].x ) return table[0].y;
    if(p >= table[1].x ) return table[1].y;
    for(int i = 0; i < 1; ++i ) {
        if(table[i].x <= p && p < table[i + 1].x ) {
            float s = (p - table[i].x) / (table[i + 1].x - table[i].x);
            return table[i].y * (1.0 - s ) + table[i+1].y * s;
        }
    }
    return 0.0;
}

mat3 RGBtoXYZ(Chromaticities N) {
    mat3 M = mat3(
        vec3(N.red.x, N.red.y, 1.0 - (N.red.x + N.red.y)),
        vec3(N.green.x, N.green.y, 1.0 - (N.green.x + N.green.y)),
        vec3(N.blue.x, N.blue.y, 1.0 - (N.blue.x + N.blue.y))
    );
    vec3 wh = vec3(N.white.x / N.white.y, 1.0, (1.0 - (N.white.x + N.white.y)) / N.white.y);
    wh = wh * inverse(M);
    mat3 WH = mat3(vec3(wh.x, 0.0, 0.0), vec3(0.0, wh.y, 0.0), vec3(0.0, 0.0, wh.z));
    M = WH * M;
    return M;
}

mat3 XYZtoRGB(Chromaticities N) {
    mat3 M = inverse(RGBtoXYZ(N));
    return M;
}

float lin_to_ACEScct(float lin) {
    if (lin <= X_BRK) return A * lin + B;
    else return (log2(lin) + 9.72) / 17.52;
}

float ACEScct_to_lin(float acescct) {
    if (acescct > Y_BRK) return exp2(acescct * 17.52 - 9.72);
    else return (acescct - B) / A;
}

vec3 ACES_to_ACEScct(vec3 aces) {
    vec3 ap1_lin = aces * ap0_to_ap1; // (RGBtoXYZ(AP0) * XYZtoRGB(AP1));
    vec3 acescct;
    acescct.x = lin_to_ACEScct(ap1_lin.x); acescct.y = lin_to_ACEScct(ap1_lin.y); acescct.z = lin_to_ACEScct(ap1_lin.z);
    return acescct;
}

vec3 ACEScct_to_ACES(vec3 acescct) {
    vec3 ap1_lin;
    ap1_lin.x = ACEScct_to_lin(acescct.x); ap1_lin.y = ACEScct_to_lin(acescct.y); ap1_lin.z = ACEScct_to_lin(acescct.z);
    return ap1_lin * ap1_to_ap0; // (RGBtoXYZ(AP1) * XYZtoRGB(AP0));
}

float uncenter_hue(float hueCentered, float centerH) {
    float hue = hueCentered + centerH;
    if (hue < 0.0) hue = hue + 360.0;
    else if (hue > 360.0) hue = hue - 360.0;
    return hue;
}

mat3 calc_sat_adjust_matrix(float sat, vec3 rgb2Y) {
    vec3 v = (1.0 - sat) * rgb2Y;
    mat3 M = mat3(v, v, v) + mat3(sat);
    return M;
}

vec3 ASCCDL_inACEScct(vec3 aces_in, vec3 SLOPE, vec3 OFFSET, vec3 POWER, float SAT) {
    vec3 acescct = ACES_to_ACEScct(aces_in);
    acescct.x = pow(clamp01((acescct.x * SLOPE.x) + OFFSET.x), 1.0 / POWER.x);
    acescct.y = pow(clamp01((acescct.y * SLOPE.y) + OFFSET.y), 1.0 / POWER.y);
    acescct.z = pow(clamp01((acescct.z * SLOPE.z) + OFFSET.z), 1.0 / POWER.z);
    float luma = dot(acescct, luminance_weights_rec709);
    float satClamp = max0(SAT);
    acescct.x = luma + satClamp * (acescct.x - luma);
    acescct.y = luma + satClamp * (acescct.y - luma);
    acescct.z = luma + satClamp * (acescct.z - luma);
    return ACEScct_to_ACES(acescct);
}

vec3 gamma_adjust_linear(vec3 rgb_in, float GAMMA, float PIVOT) {
    float SCALAR = PIVOT / pow(PIVOT, GAMMA);
    vec3 rgb_out = rgb_in;
    if (rgb_in.x > 0.0) rgb_out.x = pow(rgb_in.x, GAMMA) * SCALAR;
    if (rgb_in.y > 0.0) rgb_out.y = pow(rgb_in.y, GAMMA) * SCALAR;
    if (rgb_in.z > 0.0) rgb_out.z = pow(rgb_in.z, GAMMA) * SCALAR;
    return rgb_out;
}

vec3 sat_adjust(vec3 rgb_in, float sat_factor) {
    vec3 RGB2Y = vec3(
        RGBtoXYZ(REC709_PRI)[0].y,
        RGBtoXYZ(REC709_PRI)[1].y,
        RGBtoXYZ(REC709_PRI)[2].y
    );
    mat3 SAT_MAT = calc_sat_adjust_matrix(sat_factor, RGB2Y);
    return rgb_in * SAT_MAT;
}

const mat3 matrix_rgb_2_yab = transpose(mat3(
	vec3(1.0/3.0, 1.0/2.0, 0.0),
	vec3(1.0/3.0, -1.0/4.0, sqrt3over4),
	vec3(1.0/3.0, -1.0/4.0, -sqrt3over4)
));

vec3 rgb_2_yab(vec3 rgb) {
	vec3 yab = rgb * matrix_rgb_2_yab;
	return yab;
}

vec3 yab_2_rgb(vec3 yab) {
    vec3 rgb = yab * inverse(matrix_rgb_2_yab);
    return rgb;
}

vec3 yab_2_ych(vec3 yab) {
    vec3 ych = yab;
    float yo = yab.y * yab.y + yab.z * yab.z;
    ych.y = sqrt(yo);
    ych.z = atan(yab.z, yab.y) * (180.0 / pi);
    if (ych.z < 0.0) ych.z += 360.0;
    return ych;
}

vec3 ych_2_yab(vec3 ych) {
    vec3 yab;
    yab.x = ych.x;
    float h = ych.z * degree;
    yab.y = ych.y * cos(h);
    yab.z = ych.y * sin(h);
    return yab;
}

vec3 rgb_2_ych(vec3 rgb) {
	return yab_2_ych(rgb_2_yab(rgb));
}

vec3 ych_2_rgb(vec3 ych) {
	return yab_2_rgb(ych_2_yab(ych));
}

vec3 scale_C_at_H(vec3 rgb, float centerH, float widthH, float percentC) {
	vec3 new_rgb = rgb;
	vec3 ych = rgb_2_ych(rgb);
	if (ych.y > 0.0) {
		float centeredHue = center_hue(ych.z, centerH);
		float f_H = cubic_basis_shaper(centeredHue, widthH);
		if (f_H > 0.0) {
			vec3 new_ych = ych;
			new_ych.y = ych.y * (f_H * (percentC - 1.0) + 1.0);
			new_rgb = ych_2_rgb(new_ych);
		} else {
			new_rgb = rgb;
		}
	}
	return new_rgb;
}

vec3 rotate_H_in_H(vec3 rgb, float centerH, float widthH, float degreesShift) {
	vec3 ych = rgb_2_ych(rgb);
	vec3 new_ych = ych;
	float centeredHue = center_hue(ych.z, centerH);
	float f_H = cubic_basis_shaper(centeredHue, widthH);
	float old_hue = centeredHue;
	float new_hue = centeredHue + degreesShift;
	vec2 table[2] = vec2[](vec2(0.0, old_hue), vec2(1.0, new_hue));
	float blended_hue = interpolate1D_2(table, f_H);
	if (f_H > 0.0) new_ych.z = uncenter_hue(blended_hue, centerH);
	return ych_2_rgb(new_ych);
}

vec3 scale_C(vec3 rgb, float percentC) {
    vec3 ych = rgb_2_ych(rgb);
    ych.y = ych.y * percentC;
    return ych_2_rgb(ych);
}

vec3 overlay_f3(vec3 a, vec3 b) {
    float LUMA_CUT = lin_to_ACEScct(0.5);
    float luma = dot(a, luminance_weights_rec709);
    //float luma = 0.2126 * a.x + 0.7152 * a.y + 0.0722 * a.z;
    vec3 outp;
    if (luma < LUMA_CUT) outp = 2.0 * a * b;
    else outp = 1.0 - (2.0 * (1.0 - a) * (1.0 - b));
    return outp;
}

vec3 lmt_pfe(vec3 aces) {
	float SCALEC = 0.7;
    vec3 SLOPE   = vec3(1.0, 1.0, 0.94);
    vec3 OFFSET  = vec3(0.0, 0.0, 0.02);
    vec3 POWER   = vec3(1.0, 1.0, 1.0);
    float SAT    = 1.0;
    vec2 GAMMA   = vec2(1.5, 0.18);

    vec3 ROT1    = vec3(0.0, 30.0, 5.0);
    vec3 ROT2    = vec3(80.0, 60.0, -15.0);
    vec3 ROT3    = vec3(52.0, 50.0, -14.0);
    vec3 SCL1    = vec3(45.0, 40.0, 1.4);
    vec3 ROT4    = vec3(190.0, 40.0, 30.0);
    vec3 SCL2    = vec3(240.0, 120.0, 1.4);

#if ACES_PFE_PRESET == ACES_PFE_PRESET_NATURAL
	SCALEC = 1.0;
	SLOPE.b = 1.0;
	OFFSET.b = 0.0;
	GAMMA.x = 1.0;

	ROT1.z = 0.0;
	ROT2.z = 0.0;
	ROT3.z = 0.0;
	SCL1.z = 1.0;
	ROT4.z = 0.0;
	SCL2.z = 1.0;
#elif ACES_PFE_PRESET == ACES_PFE_PRESET_CUSTOM
	SCALEC = ACES_PFE_SCALEC;
	SLOPE = vec3(ACES_PFE_SLOPE_R, ACES_PFE_SLOPE_G, ACES_PFE_SLOPE_B);
	OFFSET = vec3(ACES_PFE_OFFSET_R, ACES_PFE_OFFSET_G, ACES_PFE_OFFSET_B);
	POWER = vec3(ACES_PFE_POWER_R, ACES_PFE_POWER_G, ACES_PFE_POWER_B);
	SAT = ACES_PFE_SAT;
	GAMMA = vec2(ACES_PFE_GAMMA1, ACES_PFE_GAMMA2);

    ROT1 = vec3(ACES_PFE_ROTATEH11, ACES_PFE_ROTATEH12, ACES_PFE_ROTATEH13);
    ROT2 = vec3(ACES_PFE_ROTATEH21, ACES_PFE_ROTATEH22, ACES_PFE_ROTATEH23);
    ROT3 = vec3(ACES_PFE_ROTATEH31, ACES_PFE_ROTATEH32, ACES_PFE_ROTATEH33);
    SCL1 = vec3(ACES_PFE_SCALECH11, ACES_PFE_SCALECH12, ACES_PFE_SCALECH13);
    ROT4 = vec3(ACES_PFE_ROTATEH41, ACES_PFE_ROTATEH42, ACES_PFE_ROTATEH43);
    SCL2 = vec3(ACES_PFE_SCALECH21, ACES_PFE_SCALECH22, ACES_PFE_SCALECH23);
#endif

	aces = scale_C(aces, SCALEC);

    aces = ASCCDL_inACEScct(aces, SLOPE, OFFSET, POWER, SAT);
    if(GAMMA.x != 1.0) aces = gamma_adjust_linear(aces, GAMMA.x, GAMMA.y);
#ifndef ACES_PFE_BYPASS
    aces = rotate_H_in_H(aces, ROT1.x, ROT1.y, ROT1.z);
    aces = rotate_H_in_H(aces, ROT2.x, ROT2.y, ROT2.z);
    aces = rotate_H_in_H(aces, ROT3.x, ROT3.y, ROT3.z);
    aces = scale_C_at_H(aces,  SCL1.x, SCL1.y, SCL1.z);
    aces = rotate_H_in_H(aces, ROT4.x, ROT4.y, ROT4.z);
    aces = scale_C_at_H(aces,  SCL2.x, SCL2.y, SCL2.z);
#endif
    return aces;
}

vec3 lmt_bleach(vec3 aces) {
    vec3 a, b, blend;
    a = sat_adjust(aces, 0.9);
    a *= 2.0;
    b = sat_adjust(aces, 0.0);
    b = gamma_adjust_linear(b, 1.2, 0.18);
    a = ACES_to_ACEScct(a);
    b = ACES_to_ACEScct(b);
    blend = overlay_f3(a, b);
    aces = ACEScct_to_ACES(blend);
    return aces;
}

#endif //INCLUDE_TONEMAPPING_ACES_LMT_OTHER
