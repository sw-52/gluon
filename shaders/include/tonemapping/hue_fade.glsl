#ifndef INCLUDE_TONEMAPPING_HUE_FADE
#define INCLUDE_TONEMAPPING_HUE_FADE

// From shadertoy by tiusic -  https://www.shadertoy.com/view/fdtGDN

vec3 mid3_(vec3 v) { return v.y < v.z ? vec3(0, v.y, 0) : v.x < v.z ? vec3(0, 0, v.z) : vec3(v.x, 0, 0); }
vec3 mid3(vec3 v)  { return v.x < v.y ? mid3_(v) : mid3_(v.yxz).yxz; }
vec3 max3v(vec3 v) { return v.x < v.y ? (v.y < v.z ? vec3(0, 0, v.z) : vec3(0, v.y, 0)) : (v.x < v.z ? vec3(0, 0, v.z) : vec3(v.x, 0, 0)); }
float max3(vec3 v) { return max(max(v.x, v.y), v.z); }
vec3 lerp(vec3 a, vec3 b, float t) { return (1. - t) * a + t * b; }
vec3 qerp(vec3 a, vec3 b, float t) { return sqrt(1. - t * t) * a + t * b; }


float tri(float x) {
    x = mod(2. * x, 2.);
    return x < 1. ? 3. * x - 1. : 5. - 3. * x;
}

vec3 hue(float h) {
    return clamp01(vec3(tri(h+0.5), tri(h+1./6.), tri(h-1./6.)));
}

vec3 baseColor(vec2 uv, float t) {
    return 3. * uv.y * lerp(hue(uv.x), vec3(1), 0.5 + 0.5 * sin(t));
}

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp01(p - K.xxx), c.y);
}

float qq(float q, float p) { return pow(p, 4. * q * (q - 1.0)); }

vec3 hdrr(vec3 c) {
    float l = min(c.g, c.b);
    float m = (c.r - l);
    if (m < eps) return c;

    float k = (max(c.g, c.b) - l) / m;
    vec3 d = c + (1. - k) * vec3(0, 1.1 * max(c.r - 0.7, 0.), max(c.r - 1.2, 0.));

    if (c.g > c.b) d.b += k * max(c.g - 0.8, 0.);
    else           d.g += k * max(c.b - 0.8, 0.);

    float j = 0.5 + 0.5 * (c.g > c.b ? k : -k);
    float g = clamp(log(m / l), 0., 1.);
    const float J = 0.38;

    if (j < J) {
        d.g *= qq(j / J, 1. - g * 0.2);
        d.b /= qq(j / J, 1. - g * 0.2);
    } else {
        d.b *= qq((j - J) / (1. - J), 1. - g * 0.4);
        d.g /= qq((j - J) / (1. - J), 1. - g * 0.2);
    }

    if (m < 0.3) {
        return lerp(c, d, m / 0.3);
    }

    return d;
}

vec3 hdr_hue_fade(vec3 c) {
    if (c.r > c.g && c.r > c.b) {
        c = hdrr(c);
    } else if (c.g > c.b && c.g > c.r) {
        c = hdrr(c.gbr).brg;
    } else if (c.b > c.r && c.b > c.g) {
        c = hdrr(c.brg).gbr;
    }
    return c;
}

#endif // INCLUDE_TONEMAPPING_HUE_FADE
