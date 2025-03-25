#ifndef INCLUDE_LIGHT_LPV_LIGHT_COLORS
#define INCLUDE_LIGHT_LPV_LIGHT_COLORS

#include "/include/utility/color.glsl"
#include "/include/light/colors/blocklight_color.glsl"

const vec3[64] light_color = vec3[64](
	vec3(1.00, 1.00, 1.00) * 12.0, //  0 | Strong white light
	vec3(1.00, 1.00, 1.00) *  6.0, //  1 | Medium white light
	vec3(1.00, 1.00, 1.00) *  1.0, //  2 | Weak white light
	vec3(1.00, 0.55, 0.27) * 12.0, //  3 | Strong golden light
	vec3(1.00, 0.57, 0.30) *  8.0, //  4 | Medium golden light
	vec3(1.00, 0.57, 0.30) *  4.0, //  5 | Weak golden light
	vec3(1.00, 0.18, 0.10) *  5.0, //  6 | Redstone components
	vec3(1.00, 0.30, 0.10) * 24.0, //  7 | Lava
	vec3(1.00, 0.45, 0.10) *  9.0, //  8 | Medium orange light
	vec3(1.00, 0.63, 0.15) *  4.0, //  9 | Brewing stand
	vec3(1.00, 0.57, 0.30) * 12.0, // 10 | Medium golden light (Jack o' Lantern)
	vec3(0.45, 0.73, 1.00) *  6.0, // 11 | Soul lights
	vec3(0.45, 0.73, 1.00) * 14.0, // 12 | Beacon
	vec3(0.75, 1.00, 0.83) *  3.0, // 13 | Sculk
	vec3(0.75, 1.00, 0.83) *  1.0, // 14 | End portal frame
	vec3(0.60, 0.10, 1.00) *  4.0, // 15 | Pink glow
	vec3(0.75, 1.00, 0.50) *  1.0, // 16 | Sea pickle
	vec3(1.00, 0.50, 0.25) *  4.0, // 17 | Nether plants
	vec3(1.00, 0.57, 0.30) *  8.0, // 18 | Medium golden light (Candles)
	vec3(1.00, 0.65, 0.30) *  8.0, // 19 | Ochre froglight
	vec3(0.86, 1.00, 0.44) *  8.0, // 20 | Verdant froglight
	vec3(0.75, 0.44, 1.00) *  8.0, // 21 | Pearlescent froglight
	vec3(0.60, 0.10, 1.00) *  2.0, // 22 | Enchanting table
	vec3(0.75, 0.44, 1.00) *  4.0, // 23 | Amethyst cluster
	vec3(0.75, 0.44, 1.00) *  4.0, // 24 | Calibrated sculk sensor
	vec3(0.75, 1.00, 0.83) *  6.0, // 25 | Active sculk sensor
	vec3(1.00, 0.18, 0.10) *  3.3, // 26 | Redstone block
#ifdef COLORED_LIGHTS_EMERALD_EMISSION
	vec3(0.10, 1.00, 0.10) *  3.3, // 27 | Emerald block
#else
	vec3(0.0),
#endif
#ifdef COLORED_LIGHTS_LAPIS_EMISSION
	vec3(0.10, 0.10, 1.00) *  3.3, // 28 | Lapis block
#else
	vec3(0.0),
#endif
	vec3(1.00, 1.00, 1.00) * 32.0, // 29 | Lightning rod
	vec3(0.60, 0.10, 1.00) * 12.0, // 30 | Nether portal
	vec3(0.0),                     // 31 | End portal
	vec3(1.0, 0.1, 0.1) *  8.0,    // 32 | Red
	vec3(1.0, 0.5, 0.1) *  8.0,    // 33 | Orange
	vec3(1.0, 1.0, 0.1) *  8.0,    // 34 | Yellow
	vec3(0.7, 0.4, 0.0) *  8.0,    // 35 | Brown
	vec3(0.1, 1.0, 0.1) *  8.0,    // 36 | Green
	vec3(0.5, 1.0, 0.5) *  8.0,    // 37 | Lime
	vec3(0.1, 0.1, 1.0) *  8.0,    // 38 | Blue
	vec3(0.5, 0.5, 1.0) *  8.0,    // 39 | Light blue
	vec3(0.1, 1.0, 1.0) *  8.0,    // 40 | Cyan
	vec3(0.7, 0.1, 1.0) *  8.0,    // 41 | Purple
	vec3(1.0, 0.1, 1.0) *  8.0,    // 42 | Magenta
	vec3(1.0, 0.5, 1.0) *  8.0,    // 43 | Pink
	vec3(0.1, 0.1, 0.1) *  8.0,    // 44 | Black
	vec3(0.9, 0.9, 0.9) *  8.0,    // 45 | White
	vec3(0.3, 0.3, 0.3) *  8.0,    // 46 | Gray
	vec3(0.7, 0.7, 0.7) *  8.0,    // 47 | Light gray
	vec3(1.00, 1.00, 1.00) * 32.0, // 48 | Lightning rod
	vec3(0.0),  // 49 | Unused
	vec3(0.0),  // 50 | Unused
	vec3(0.0),  // 51 | Unused
	vec3(0.0),  // 52 | Unused
	vec3(0.0),  // 53 | Unused
	vec3(0.0),  // 54 | Unused
	vec3(0.0),  // 55 | Unused
	vec3(0.0),  // 56 | Unused
	vec3(0.0),  // 57 | Unused
	vec3(0.0),  // 58 | Unused
	vec3(0.0),  // 59 | Unused
	vec3(0.0),  // 60 | Unused
	vec3(0.0),  // 61 | Unused
	vec3(0.0),  // 62 | Unused
	vec3(0.0)   // 63 | Unused
);

const vec3[16] tint_color = vec3[16](
	vec3(1.0, 0.1, 0.1), // Red
	vec3(1.0, 0.5, 0.1), // Orange
	vec3(1.0, 1.0, 0.1), // Yellow
	vec3(0.7, 0.4, 0.0), // Brown
	vec3(0.1, 1.0, 0.1), // Green
	vec3(0.5, 1.0, 0.5), // Lime
	vec3(0.1, 0.1, 1.0), // Blue
	vec3(0.5, 0.5, 1.0), // Light blue
	vec3(0.1, 1.0, 1.0), // Cyan
	vec3(0.7, 0.1, 1.0), // Purple
	vec3(1.0, 0.1, 1.0), // Magenta
	vec3(1.0, 0.5, 1.0), // Pink
	vec3(0.1, 0.1, 0.1), // Black
	vec3(0.9, 0.9, 0.9), // White
	vec3(0.3, 0.3, 0.3), // Gray
	vec3(0.7, 0.7, 0.7)  // Light gray
);

bool is_candle(uint block_id) {
	return 232u <= block_id && block_id < 300u;
}

bool is_fallback(uint block_id) {
	return 300u <= block_id && block_id < 332u;
}

bool is_redstone_wire(uint block_id) {
	return 332u <= block_id && block_id < 347u;
}

float get_candle_intensity(uint level) {
	//return level > 0 ? (level > 1 ? (level > 2 ? 10.0 : 8.0) : 6.0) : 3.0;
	return sqr(float(level + 1u)) + 1u;
}

vec3 get_light_color(uint id) {
	if (id < 64) return light_color[id];
	else if (is_candle(id)) {
		if(id > 295) { // Uncolored Candle
			return light_color[18u] / 8.0 * get_candle_intensity(id - 296u);
		}

		id -= 232u;
		uint level = uint(floor(float(id) / 16.0));
		float intensity = get_candle_intensity(level);

	#ifdef COLORED_LIGHTS_COLORED_CANDLES
		return tint_color[id - level * 16u] * intensity;
	#else
		return light_color[18u] / 8.0 * intensity;
	#endif
	}
	else if (is_fallback(id)) {
		return blocklight_color * 0.8 * (id - 300u); // Blocklight 15 = Light intensity 12.0
	}
#ifdef COLORED_LIGHTS_REDSTONE_WIRE
	else if (is_redstone_wire(id)) {
		float intensity = float(id - 331u);
		return light_color[6] * intensity * 0.15;
	}
#endif
	return vec3(0.0);
}

#endif // INCLUDE_LIGHT_LPV_LIGHT_COLORS
