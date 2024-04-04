#ifndef INCLUDE_LIGHT_COLORS_SPACE_COLOR
#define INCLUDE_LIGHT_COLORS_SPACE_COLOR

#include "/include/sky/atmosphere.glsl"
#include "/include/utility/color.glsl"

//uniform float moon_phase_brightness;

// Magic brightness adjustment so that auto exposure isn't needed
float get_sun_exposure() {
	const float base_scale = 7.0 * SUN_I;
	return base_scale;
}

vec3 get_light_color() {
	vec3 light_color  = vec3(mix(get_sun_exposure(), 0.0, step(0.5, sunAngle)));
	     light_color *= sunlight_color;
	     light_color *= clamp01(rcp(0.02) * light_dir.y); // fade away during day/night transition
		 light_color *= 1.0 - 0.25 * pulse(abs(light_dir.y), 0.15, 0.11);

	return light_color;
}

vec3 get_ambient_color() {
	return vec3(0.2);
}

#endif // INCLUDE_LIGHT_COLORS_SPACE_COLOR
