#ifndef INCLUDE_FOG_AIR_FOG_VL
#define INCLUDE_FOG_AIR_FOG_VL

#include "/include/light/distortion.glsl"
#include "/include/sky/atmosphere.glsl"
#include "/include/utility/encoding.glsl"
#include "/include/utility/phase_functions.glsl"
#include "/include/utility/random.glsl"
#include "/include/utility/space_conversion.glsl"

#if defined COLORED_LIGHTS && defined COLORED_LIGHTS_FOG
#include "/include/light/lpv/blocklight.glsl"
#endif

#ifdef PROGRAM_DEFERRED0
	#define NOISE_3D colortex6
#else
	#define NOISE_3D colortex0
#endif

const uint  air_fog_min_step_count    = AIR_FOG_MIN_STEPS; // 25  8
const uint  air_fog_max_step_count    = AIR_FOG_MAX_STEPS; // 50  25
const float air_fog_step_count_growth = AIR_FOG_STEP_GROWTH; //  0.1
const float air_fog_volume_top        = AIR_FOG_VOLUME_TOP; // 320.0
const float air_fog_volume_bottom     = SEA_LEVEL - 24.0;
const vec2  air_fog_falloff_start     = vec2(AIR_FOG_RAYLEIGH_FALLOFF_START, AIR_FOG_MIE_FALLOFF_START) + SEA_LEVEL;
const vec2  air_fog_falloff_half_life = vec2(AIR_FOG_RAYLEIGH_FALLOFF_HALF_LIFE, AIR_FOG_MIE_FALLOFF_HALF_LIFE);

vec2 air_fog_density(vec3 world_pos) {
	const vec2 mul = -rcp(air_fog_falloff_half_life);
	const vec2 add = -mul * air_fog_falloff_start;

	vec2 density = exp2(min(world_pos.y * mul + add, 0.0));

	// fade away below sea level
	density *= linear_step(air_fog_volume_bottom, SEA_LEVEL, world_pos.y);

#ifdef AIR_FOG_CLOUDY_NOISE
	const vec3 wind = 0.03 * vec3(1.0, 0.0, 0.7);

	float noise = texture(NOISE_3D, 0.015 * world_pos + wind * frameTimeCounter).x;

	density.y *= 2.0 * sqr(1.0 - noise);
#endif

	return density;
}

mat2x3 raymarch_air_fog(vec3 world_start_pos, vec3 world_end_pos, bool sky, float skylight, float dither) {
	const uint air_fog_multiple_scattering_iterations = FOG_MULTIPLE_SCATTERING_ITERATIONS; // 4

	vec3 world_dir = world_end_pos - world_start_pos;

	float length_sq = length_squared(world_dir);
	float norm = inversesqrt(length_sq);
	float ray_length = length_sq * norm;
	world_dir *= norm;

	vec3 shadow_start_pos = transform(shadowModelView, world_start_pos - cameraPosition);
	     shadow_start_pos = project_ortho(shadowProjection, shadow_start_pos);

	vec3 shadow_dir = mat3(shadowModelView) * world_dir;
	     shadow_dir = diagonal(shadowProjection).xyz * shadow_dir;

	float distance_to_lower_plane = (air_fog_volume_bottom - eyeAltitude) / world_dir.y;
	float distance_to_upper_plane = (air_fog_volume_top    - eyeAltitude) / world_dir.y;
	float distance_to_volume_start, distance_to_volume_end;

	if (eyeAltitude < air_fog_volume_bottom) {
		// Below volume
		distance_to_volume_start = distance_to_lower_plane;
		distance_to_volume_end = world_dir.y < 0.0 ? -1.0 : distance_to_upper_plane;
	} else if (eyeAltitude < air_fog_volume_top) {
		// Inside volume
		distance_to_volume_start = 0.0;
		distance_to_volume_end = world_dir.y < 0.0 ? distance_to_lower_plane : distance_to_upper_plane;
	} else {
		// Above volume
		distance_to_volume_start = distance_to_upper_plane;
		distance_to_volume_end = world_dir.y < 0.0 ? distance_to_upper_plane : -1.0;
	}

#ifdef DISTANT_HORIZONS
    float fog_end = float(dhRenderDistance);
#else
    float fog_end = far;
#endif

	if (distance_to_volume_end < 0.0) return mat2x3(vec3(0.0), vec3(1.0));

	ray_length = sky ? distance_to_volume_end : ray_length;
	ray_length = clamp(ray_length - distance_to_volume_start, 0.0, fog_end);

	uint step_count = uint(float(air_fog_min_step_count) + air_fog_step_count_growth * ray_length);
	     step_count = min(step_count, air_fog_max_step_count);

	float step_length = ray_length * rcp(float(step_count));

	vec3 world_step = world_dir * step_length;
	vec3 world_pos  = world_start_pos + world_dir * (distance_to_volume_start + step_length * dither);

	vec3 shadow_step = shadow_dir * step_length;
	vec3 shadow_pos  = shadow_start_pos + shadow_dir * (distance_to_volume_start + step_length * dither);

	vec3 transmittance = vec3(1.0);

	mat2x3 light_sun = mat2x3(0.0); // Rayleigh, mie
	mat2x3 light_sky = mat2x3(0.0); // Rayleigh, mie
	mat2x3 light_lpv = mat2x3(0.0); // Rayleigh, mie

	for (int i = 0; i < step_count; ++i, world_pos += world_step, shadow_pos += shadow_step) {
		vec3 shadow_screen_pos = distort_shadow_space(shadow_pos) * 0.5 + 0.5;

#if defined SHADOW && !defined PROGRAM_DEFERRED0
	 	ivec2 shadow_texel = ivec2(shadow_screen_pos.xy * shadowMapResolution * MC_SHADOW_QUALITY);

	#ifdef AIR_FOG_COLORED_LIGHT_SHAFTS
		float depth0 = texelFetch(shadowtex0, shadow_texel, 0).x;
		float depth1 = texelFetch(shadowtex1, shadow_texel, 0).x;
		vec3  color = clamp01(texelFetch(shadowcolor0, shadow_texel, 0).rgb * 4.0);
		float color_weight = step(depth0, shadow_screen_pos.z) * step(eps, max_of(color));

		color = color * color_weight + (1.0 - color_weight);

		vec3 shadow = step(shadow_screen_pos.z, depth1) * color;
		     shadow = (clamp01(shadow_screen_pos) == shadow_screen_pos) ? shadow : vec3(1.0);
	#else
		float depth1 = texelFetch(shadowtex1, shadow_texel, 0).x;
		float shadow = step(float(clamp01(shadow_screen_pos) == shadow_screen_pos) * shadow_screen_pos.z, depth1);
	#endif

	#if defined CLOUD_SHADOWS && defined FOG_CLOUD_SHADOWS && defined WORLD_OVERWORLD
		shadow *= get_cloud_shadows(colortex8, world_pos - cameraPosition);
	#endif
#elif defined CLOUD_SHADOWS && defined FOG_CLOUD_SHADOWS && defined WORLD_OVERWORLD
	float shadow = get_cloud_shadows(colortex8, world_pos - cameraPosition);
#else
		#define shadow 1.0
#endif

#if defined COLORED_LIGHTS && defined COLORED_LIGHTS_FOG
		vec3 lpv_color = get_lpv_linear(world_pos - cameraPosition) + 0.0;
		//lpv_color = lpv_color * exp2(lpv_color * 0.4) + 1.0;
		//lpv_color = sqr(lpv_color) * rcp(0.2 * lpv_color + 1.0) + 1.0;
		//lpv_color = exp2(min(lpv_color, 20000.0));
		
		//lpv_color = pow(vec3(1.2), 4.0 * lpv_color);
		//const vec4 lpv_fog_params = vec4(-1.341e-2, 6.223e-1, 3.663e-2, -1.623e-1);

		//lpv_color = pow(lpv_fog_params.x * lpv_color, vec3(3.0)) + sqr(lpv_fog_params.y * lpv_color) + lpv_fog_params.z * lpv_color + 1.0;
	#ifdef AIR_FOG_CLOUDY_NOISE
			const mat3 wind = mat3(
				vec3(1.0, 0.0, 0.7) * 0.10,
				vec3(0.4, 0.0, -0.6) * 0.10,
				vec3(-0.8, 0.0, 0.1) * 0.10
			);

			float noise  = texture(NOISE_3D, 0.05 * world_pos + wind[0] * frameTimeCounter).x * 0.75;
			      noise += texture(NOISE_3D, 0.075 * world_pos + wind[1] * frameTimeCounter).x * 0.75;
				  noise -= texture(NOISE_3D, 0.075 * world_pos + wind[0] * frameTimeCounter + 1234.453).x * 0.75;
				  noise += texture(NOISE_3D, 0.15 * world_pos + wind[2] * frameTimeCounter).x * 1.0;
				  noise *= rcp(1.75);

			//vec3 lpv_light_dir = normalize(get_lpv_direction_fog(world_pos - cameraPosition/*, hash2(fract(world_pos))*/));
			// mat3 lpv_gradient = get_lpv_gradient_rgb(world_pos - cameraPosition + vec3(0.0, 0.6, 0.0), 1.1);

			// Fix opposite hues being treated as light sources
			/*vec3 hsl = rgb_to_hsl(lpv_color);
			vec3 primary_hues = vec3(
				rgb_to_hsl(vec3(1.0, 0.0, 0.0)).x,
				rgb_to_hsl(vec3(0.0, 1.0, 0.0)).x,
				rgb_to_hsl(vec3(0.0, 0.0, 1.0)).x
			);*/
			/*vec3 hsl_weighted = vec3(
				isolate_hue_norm(hsl, primary_hues.r, rcp(5.0)),
				isolate_hue_norm(hsl, primary_hues.g, rcp(5.0)),
				isolate_hue_norm(hsl, primary_hues.b, rcp(5.0))
			);*/
			/*vec3 hsl_weighted = vec3(
				isolate_hue_norm(rgb_to_hsl(lpv_gradient[0]), hsl.x, rcp(5.0)),
				isolate_hue_norm(rgb_to_hsl(lpv_gradient[1]), hsl.x, rcp(5.0)),
				isolate_hue_norm(rgb_to_hsl(lpv_gradient[2]), hsl.x, rcp(5.0))
			);
			hsl_weighted = mix(vec3(1.0), hsl_weighted, hsl.y);
			hsl_weighted *= clamp01(hsl.z * 2.0);*/

			/*vec3 hue_gradient = vec3(
				isolate_hue_norm(rgb_to_hsl(lpv_gradient[0]), hsl.x, 1.0),
				isolate_hue_norm(rgb_to_hsl(lpv_gradient[1]), rgb_to_hsl(lpv_gradient[1]).x, 1.0),
				isolate_hue_norm(rgb_to_hsl(lpv_gradient[2]), rgb_to_hsl(lpv_gradient[2]).x, 1.0)

			);*/
			//lpv_gradient = transpose(lpv_gradient);

			// vec3 lpv_light_dir = lpv_color * lpv_gradient;
			// lpv_light_dir = length(lpv_light_dir) >= eps ? normalize(lpv_light_dir) : vec3(0.0);

			//lpv_color *= 25.0 /* pow((1.0 - noise), 3.0)*/ * pow(max0(dot(lpv_light_dir, vec3(0.0, -1.0, 0.0))), 4.0);
			//lpv_color = pow(max0(dot((lpv_light_dir), vec3(0.0, -1.0, 0.0))), 3.0) * lpv_color * 5.0 /* sqr(noise)*/;
			lpv_color = max0(lpv_color * sqr(noise) * 5.0);
			//if (any(isnan(lpv_color))) lpv_color = vec3(0.0);
	#endif
#else
		#define lpv_color 0.0
#endif

		vec2 density = air_fog_density(world_pos) * step_length;

		vec3 step_optical_depth = air_fog_coeff[1] * density;
		vec3 step_transmittance = exp(-step_optical_depth);
		vec3 step_transmitted_fraction = (1.0 - step_transmittance) / max(step_optical_depth, eps);

		vec3 visible_scattering = step_transmitted_fraction * transmittance;

		light_sun[0] += visible_scattering * density.x * shadow;
		light_sun[1] += visible_scattering * density.y * shadow;
		light_sky[0] += visible_scattering * density.x;
		light_sky[1] += visible_scattering * density.y;
		light_lpv[0] += visible_scattering * density.x * lpv_color;
		light_lpv[1] += visible_scattering * density.y * lpv_color;

		transmittance *= step_transmittance;

#if defined COLORED_LIGHTS && defined COLORED_LIGHTS_FOG
		/*vec3 lpv_light_dir = normalize(get_lpv_direction_clouds(world_pos - cameraPosition, hash2(fract(world_pos))));
		float lpv_LoV = dot(world_dir, lpv_light_dir);
		float lpv_mie_phase = 0.7 * henyey_greenstein_phase(lpv_LoV, 0.5) + 0.3 * henyey_greenstein_phase(lpv_LoV, -0.2);

		float lpv_scatter_amount = 1.0;

		for (int i = 0; i < air_fog_multiple_scattering_iterations; ++i) {
			scattering += lpv_scatter_amount * (light_sun * vec2(isotropic_phase, mie_phase)) * light_color;
			scattering += lpv_scatter_amount * (light_lpv * vec2(isotropic_phase, mie_phase));

			scatter_amount *= 0.5;
			mie_phase = mix(mie_phase, isotropic_phase, 0.3);
		}*/
#endif
	}

	light_sun[0] *= air_fog_coeff[0][0];
	light_sun[1] *= air_fog_coeff[0][1];
	light_sky[0] *= air_fog_coeff[0][0] * eye_skylight;
	light_sky[1] *= air_fog_coeff[0][1] * eye_skylight;
	light_lpv[0] *= air_fog_coeff[0][0] /* 0.99 + 0.01*/;
	light_lpv[1] *= air_fog_coeff[0][1] /* 0.99 + 0.01*/;

	if (!sky) {
		// Skylight falloff
		light_sky[0] *= skylight;
		light_sky[1] *= skylight;
	}

	float LoV = dot(world_dir, light_dir);
	float mie_phase = 0.7 * henyey_greenstein_phase(LoV, 0.5) + 0.3 * henyey_greenstein_phase(LoV, -0.2);

	/*
	// Single scattering
	vec3 scattering  = light_color * (light_sun * vec2(isotropic_phase, mie_phase));
	     scattering += ambient_color * (light_sky * vec2(isotropic_phase));
	/*/
	// Multiple scattering
	vec3 scattering = vec3(0.0);
	float scatter_amount = 1.0;

#ifdef PROGRAM_DEFERRED0
	vec3 ambient_color = ambient_color_fog;
#endif

	scattering += light_sky * vec2(isotropic_phase) * ambient_color;
	//scattering += light_lpv;

	for (int i = 0; i < air_fog_multiple_scattering_iterations; ++i) {
		scattering += scatter_amount * (light_sun * vec2(isotropic_phase, mie_phase)) * light_color;
		scattering += scatter_amount * (light_lpv * vec2(isotropic_phase, mie_phase));

		scatter_amount *= 0.5;
		mie_phase = mix(mie_phase, isotropic_phase, 0.3);
	}
	//*/

	scattering *= 1.0 - blindness;

	return mat2x3(scattering, transmittance);
}

#endif // INCLUDE_FOG_AIR_FOG_VL
