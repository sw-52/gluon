/*
--------------------------------------------------------------------------------

  Photon Shader by SixthSurge

  program/composite0.glsl:
  Calculate volumetric fog

--------------------------------------------------------------------------------
*/

#include "/include/global.glsl"


//----------------------------------------------------------------------------//
#if defined vsh

out vec2 uv;

flat out vec3 ambient_color;
flat out vec3 light_color;

#if defined WORLD_OVERWORLD
flat out mat2x3 air_fog_coeff[2];
#endif

// ------------
//   Uniforms
// ------------

uniform sampler2D colortex4; // Sky map, lighting color palette

uniform float rainStrength;
uniform float sunAngle;

uniform int worldTime;
uniform int worldDay;

uniform vec3 sun_dir;

uniform float eye_skylight;

uniform vec2 view_res;

uniform float biome_temperate;
uniform float biome_arid;
uniform float biome_snowy;
uniform float biome_taiga;
uniform float biome_jungle;
uniform float biome_swamp;
uniform float biome_may_rain;
uniform float biome_may_snow;
uniform float biome_temperature;
uniform float biome_humidity;

uniform float time_sunrise;
uniform float time_noon;
uniform float time_sunset;
uniform float time_midnight;

#if defined WORLD_OVERWORLD
#define WEATHER_FOG
#include "/include/misc/weather.glsl"
#endif

void main() {
	uv = gl_MultiTexCoord0.xy;

	int lighting_color_x = SKY_MAP_LIGHT_X;
	light_color   = texelFetch(colortex4, ivec2(lighting_color_x, 0), 0).rgb;
	ambient_color = texelFetch(colortex4, ivec2(lighting_color_x, 1), 0).rgb;

#if defined WORLD_OVERWORLD
	mat2x3 rayleigh_coeff = air_fog_rayleigh_coeff(), mie_coeff = air_fog_mie_coeff();
	air_fog_coeff[0] = mat2x3(rayleigh_coeff[0], mie_coeff[0]);
	air_fog_coeff[1] = mat2x3(rayleigh_coeff[1], mie_coeff[1]);
#endif

	vec2 vertex_pos = gl_Vertex.xy;
	gl_Position = vec4(vertex_pos * 2.0 - 1.0, 0.0, 1.0);
}

#endif
//----------------------------------------------------------------------------//



//----------------------------------------------------------------------------//
#if defined fsh

layout (location = 0) out vec3 fog_scattering;
layout (location = 1) out vec3 fog_transmittance;

/* RENDERTARGETS: 6,7 */

in vec2 uv;

flat in vec3 ambient_color;
flat in vec3 light_color;

#if defined WORLD_OVERWORLD
flat in mat2x3 air_fog_coeff[2];
#endif

// ------------
//   Uniforms
// ------------

uniform sampler2D noisetex;

uniform sampler3D colortex0; // 3D worley noise
uniform sampler2D colortex1; // gbuffer data
uniform sampler2D colortex2; // gbuffer 1
uniform sampler2D colortex3; // translucent color
uniform sampler2D colortex4; // sky map

uniform sampler2D depthtex0;
uniform sampler2D depthtex1;

#ifndef WORLD_NETHER
#ifdef SHADOW
uniform sampler2D shadowtex0;
uniform sampler2D shadowtex1;
#ifdef SHADOW_COLOR
uniform sampler2D shadowcolor0;
#endif
#endif
#endif

#ifdef CLOUD_SHADOWS
uniform sampler2D colortex8; // cloud shadow map
#endif

uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferProjectionInverse;

uniform mat4 shadowModelView;
uniform mat4 shadowModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowProjectionInverse;

#ifdef DISTANT_HORIZONS
uniform int dhRenderDistance;
#endif

uniform vec3 cameraPosition;

uniform float near;
uniform float far;

uniform float blindness;
uniform float eyeAltitude;
uniform float rainStrength;
uniform float wetness;

uniform float sunAngle;
uniform float frameTimeCounter;

uniform int isEyeInWater;
uniform int worldTime;
uniform int frameCounter;

uniform float world_age;

uniform vec3 light_dir;
uniform vec3 sun_dir;
uniform vec3 moon_dir;

uniform vec2 view_res;
uniform vec2 view_pixel_size;
uniform vec2 taa_offset;

uniform float eye_skylight;
uniform float desert_sandstorm;

uniform float time_sunrise;
uniform float time_noon;
uniform float time_sunset;
uniform float time_midnight;

#ifdef COLORED_LIGHTS
uniform sampler3D light_sampler_a;
uniform sampler3D light_sampler_b;
#endif

// ------------
//   Includes
// ------------

#if defined WORLD_OVERWORLD
#ifdef CLOUD_SHADOWS
#include "/include/light/cloud_shadows.glsl"
#endif

#include "/include/fog/air_fog_vl.glsl"
#endif

#if defined WORLD_END //|| defined WORLD_NETHER
#include "/include/fog/end_fog_vl.glsl"
#endif

#include "/include/fog/water_fog_vl.glsl"

#include "/include/misc/distant_horizons.glsl"
#include "/include/misc/water_normal.glsl"
#include "/include/utility/encoding.glsl"
#include "/include/utility/random.glsl"
#include "/include/utility/space_conversion.glsl"

vec3 apply_translucent(vec3 solid_color, vec4 translucent_color) {
	if (TRANSLUCENT_TINT > 0.0) {
		vec3 t_hsl = rgb_to_hsl(translucent_color.rgb);
		solid_color *= mix(vec3(1.0), hsl_to_rgb(vec3(t_hsl.xy, 0.5)), TRANSLUCENT_TINT * t_hsl.y);
	}

	return solid_color * (1.0 - translucent_color.a) + translucent_color.rgb * translucent_color.a;
}


/*mat2x3 apply_translucent_fog(inout fog, vec4 translucent_color) {

}*/

void main() {
	ivec2 fog_texel  = ivec2(gl_FragCoord.xy);
	ivec2 view_texel = ivec2(gl_FragCoord.xy * taau_render_scale * rcp(VL_RENDER_SCALE));

	float depth0        = texelFetch(depthtex0, view_texel, 0).x;
	float depth1        = texelFetch(depthtex1, view_texel, 0).x;
	vec4 gbuffer_data_0 = texelFetch(colortex1, view_texel, 0);

#ifdef DISTANT_HORIZONS
    mat4 projection_matrix, projection_matrix_inverse;
	mat4 back_projection_matrix, back_projection_matrix_inverse;
	float dh_depth  = texelFetch(dhDepthTex, view_texel, 0).x;
	float dh_depth1 = texelFetch(dhDepthTex1, view_texel, 0).x;
	bool front_is_dh_terrain = is_distant_horizons_terrain(depth0, dh_depth);
    bool back_is_dh_terrain  = is_distant_horizons_terrain(depth1, dh_depth1);

	if (front_is_dh_terrain) depth0 = dh_depth;
	if (back_is_dh_terrain)  depth1 = dh_depth1;

    /*if (depth0 == 1.0) {
        is_dh_terrain = true;
        depth0 = dh_depth;
        projection_matrix = dhProjection;
        projection_matrix_inverse = dhProjectionInverse;
    } else {
        is_dh_terrain = false;
        projection_matrix = gbufferProjection;
        projection_matrix_inverse = gbufferProjectionInverse;
    }

	if (depth1 == 1.0) {
		is_dh_terrain = true;
		depth1 = dh_depth1;
		back_projection_matrix = dhProjection;
        back_projection_matrix_inverse = dhProjectionInverse;
	} else {
		is_dh_terrain = false;
		back_projection_matrix = gbufferProjection;
        back_projection_matrix_inverse = gbufferProjectionInverse;
	}*/
#else
	#define front_is_dh_terrain       false
	#define back_is_dh_terrain        false
    //#define projection_matrix         gbufferProjection
    //#define projection_matrix_inverse gbufferProjectionInverse
	//#define back_projection_matrix         gbufferProjection
    //#define back_projection_matrix_inverse gbufferProjectionInverse
#endif

	float skylight = unpack_unorm_2x8(gbuffer_data_0.w).y;
	uint material_mask = uint(255.0 * unpack_unorm_2x8(gbuffer_data_0.y).y);

	bool is_translucent = depth0 < depth1;
	bool is_water = material_mask == 1u;

	vec3 view_pos  = screen_to_view_space(vec3(uv, depth0), true, front_is_dh_terrain);
	vec3 scene_pos = view_to_scene_space(view_pos);
	vec3 world_pos = scene_pos + cameraPosition;

	vec3 view_back_pos  = screen_to_view_space(vec3(uv, depth1), true, back_is_dh_terrain);
	vec3 scene_back_pos = view_to_scene_space(view_back_pos);
	vec3 world_back_pos = scene_back_pos + cameraPosition;

	float dither = texelFetch(noisetex, fog_texel & 511, 0).b;
	      dither = r1(frameCounter, dither);

	vec3 world_start_pos = gbufferModelViewInverse[3].xyz + cameraPosition;
	vec3 world_end_pos   = world_pos;

	vec3 world_back_start_pos = world_end_pos;
#ifdef DISTANT_HORIZONS
	if (/*!front_is_dh_terrain &&*/ back_is_dh_terrain) {
		/*vec3 view_back_start_pos  = screen_to_view_space(vec3(uv, dh_depth), true, true);
		vec3 scene_back_start_pos = view_to_scene_space(view_pos);
		     world_back_start_pos = scene_back_start_pos + cameraPosition;*/
			 //world_back_start_pos = world_start_pos;
	}
#endif
#if defined WATER_REFRACTION || (defined TRANSLUCENT_REFRACTION && defined NORMAL_MAPPING)
	vec3 flat_normal = decode_unit_vector(unpack_unorm_2x8(gbuffer_data_0.z));

	vec4 gbuffer_data_1 = texelFetch(colortex2, view_texel, 0);
	vec3 normal = decode_unit_vector(gbuffer_data_1.xy);

	bool has_refraction = normal != flat_normal && depth0 != depth1;
	vec2 refracted_uv;
	bool refracted = false;

	mat3 tbn = get_tbn_matrix(flat_normal);

	vec3 world_dir; float view_dist;
	length_normalize(scene_pos - gbufferModelViewInverse[3].xyz, world_dir, view_dist);
	float layer_dist = abs(view_dist - length(view_back_pos));

	if (is_water) {
#ifdef WATER_REFRACTION

	// Water waves

	#ifdef WATER_WAVES
		if (flat_normal.y > 0.01 && isEyeInWater == 0
		 || flat_normal.y < 0.01 && isEyeInWater != 0
		) {
			vec2 coord = world_pos.xz;

			bool flowing_water = abs(flat_normal.y) < 0.99;
			vec2 flow_dir = flowing_water ? normalize(flat_normal.xz) : vec2(0.0);

	#ifdef WATER_PARALLAX
			vec3 tangent_dir = world_dir * tbn;
			coord = get_water_parallax_coord(tangent_dir, coord, flow_dir, flowing_water);
	#endif

			normal = tbn * get_water_normal(world_pos, flat_normal, coord, flow_dir, skylight, flowing_water);

		}
	#endif

		vec3 tangent_normal = normal * tbn;

		refracted_uv = uv + tangent_normal.xy * rcp(max(view_dist, 1.0)) * min(layer_dist, 8.0) * (0.1 * WATER_REFRACTION_INTENSITY);

		//vec4 refracted_color = texture(colortex3, refracted_uv * taau_render_scale).rgba;

		// Make sure the refracted object is behind water
		float refracted_data  = texelFetch(colortex1, ivec2(refracted_uv * taau_render_scale * view_res), 0).y;
		uint  refracted_mask  = uint(unpack_unorm_2x8(refracted_data).y * 255.0);

		if (refracted_mask == 1) refracted = true;
#endif
	} else if (has_refraction) {
#if defined TRANSLUCENT_REFRACTION && defined NORMAL_MAPPING
		vec3 tangent_normal = normal * tbn;

		// light gray glass
	#ifdef LIGHT_GRAY_GLASS_REFRACTION_BOOST
		bool has_refraction_boost = material_mask == 179u;
	#else
		const bool has_refraction_boost = false;
	#endif
		refracted_uv = uv + tangent_normal.xy * rcp(clamp(pow(view_dist * 0.5, 1.5), 0.1, 1.0)) * clamp(layer_dist, 1.0, 8.0) * (0.02 * TRANSLUCENT_REFRACTION_INTENSITY)
			 * (1.0 + float(has_refraction_boost) * 3.0) * (1.0 - float(material_mask == 62u) * 0.75);


		if ((length(tangent_normal.xy) > 0.02 || has_refraction_boost) && uv != refracted_uv) {
			if (refracted_uv != clamp01(refracted_uv)) {
				refracted_uv = 1.0 - abs(mod(refracted_uv, 2.0) - 1.0);
			}
			refracted = true;
		}
#endif
	}

	if (refracted) {
		ivec2 refracted_texel = ivec2(refracted_uv * taau_render_scale * view_res);
			//float refracted_depth0   = texelFetch(depthtex0, refracted_texel, 0).x;
			float refracted_depth1   = texelFetch(depthtex1, refracted_texel, 0).x;

		#ifdef DISTANT_HORIZONS
			float refracted_dh_depth = texelFetch(dhDepthTex, refracted_texel, 0).x;
			bool refracted_is_dh_terrain = is_distant_horizons_terrain(refracted_depth1, dh_depth);
			if (refracted_is_dh_terrain) refracted_depth1 = refracted_dh_depth;
		#else
			#define refracted_is_dh_terrain false
		#endif

			// Make sure the refracted pos is behind
			/*float refracted_depth = texelFetch(depthtex1, ivec2(refracted_uv * taau_render_scale * view_res), 0).x;
			refracted_depth = linearize_depth(refracted_depth);
			if (refracted_depth >= linearize_depth(depth0)) {*/
			
				vec3 refracted_view_pos  = screen_to_view_space(vec3(uv, depth0), true, false);
				vec3 refracted_scene_pos = view_to_scene_space(refracted_view_pos);
				world_back_start_pos = refracted_scene_pos + cameraPosition;

				vec3 refracted_view_back_pos = screen_to_view_space(vec3(refracted_uv, refracted_depth1), true, refracted_is_dh_terrain);
				vec3 refracted_scene_back_pos = view_to_scene_space(refracted_view_back_pos);
				world_back_pos = refracted_scene_back_pos + cameraPosition;
				// view_texel * view_pixel_size

				//vec4 refracted_gbuffer_data = texelFetch(colortex1, view_texel, 0);
				//skylight = 1.0;
			//}
	}
#endif


/*
#if defined WATER_REFRACTION || (defined TRANSLUCENT_REFRACTION && defined NORMAL_MAPPING)
	//vec3 flat_normal = decode_unit_vector(unpack_unorm_2x8(gbuffer_data_0.z));

	//vec4 gbuffer_data_1 = texelFetch(colortex2, view_texel, 0);
	//vec3 normal = decode_unit_vector(gbuffer_data_1.xy);

	//bool has_refraction = normal != flat_normal && depth0 != depth1;
	if (has_refraction) {
		mat3 tbn = get_tbn_matrix(flat_normal);
		vec3 tangent_normal = normal * tbn;

		vec3 world_dir; float view_dist;
		length_normalize(scene_pos - gbufferModelViewInverse[3].xyz, world_dir, view_dist);
		float layer_dist = abs(view_dist - length(view_back_pos));

		// light gray glass
	#ifdef LIGHT_GRAY_GLASS_REFRACTION_BOOST
		bool has_refraction_boost = material_mask == 179u;
	#else
		const bool has_refraction_boost = false;
	#endif
		vec2 refracted_uv = uv;
	#ifdef WATER_REFRACTION
		if (is_water) {
			refracted_uv += tangent_normal.xy * rcp(max(view_dist, 1.0)) * min(layer_dist, 8.0) * (0.1 * WATER_REFRACTION_INTENSITY);
		} else
	#else
		if (!is_water)
	#endif
		{
		#if defined TRANSLUCENT_REFRACTION && defined NORMAL_MAPPING
			refracted_uv += tangent_normal.xy * rcp(clamp(pow(view_dist * 0.5, 1.5), 0.1, 1.0)) * clamp(layer_dist, 1.0, 8.0) * (0.02 * TRANSLUCENT_REFRACTION_INTENSITY)
				 * (1.0 + float(has_refraction_boost) * 3.0) * (1.0 - float(material_mask == 62u) * 0.75);
		#endif
		}

		/*if (is_water) {
		#ifdef WATER_REFRACTION
			refracted_uv += tangent_normal.xy * rcp(max(view_dist, 1.0)) * min(layer_dist, 8.0) * (0.1 * WATER_REFRACTION_INTENSITY);
		#endif
		} else {
		#if defined TRANSLUCENT_REFRACTION && defined NORMAL_MAPPING
			refracted_uv += tangent_normal.xy * rcp(clamp(pow(view_dist * 0.5, 1.5), 0.1, 1.0)) * clamp(layer_dist, 1.0, 8.0) * (0.02 * TRANSLUCENT_REFRACTION_INTENSITY)
				 * (1.0 + float(has_refraction_boost) * 3.0) * (1.0 - float(material_mask == 62u) * 0.75);
		#endif
		}*/

		/*if ((length(tangent_normal.xy) > 0.02 || has_refraction_boost) && uv != refracted_uv) {
			if (refracted_uv != clamp01(refracted_uv)) {
				refracted_uv = 1.0 - abs(mod(refracted_uv, 2.0) - 1.0);
			}

			ivec2 refracted_texel = ivec2(refracted_uv * taau_render_scale * view_res);
			//float refracted_depth0   = texelFetch(depthtex0, refracted_texel, 0).x;
			float refracted_depth1   = texelFetch(depthtex1, refracted_texel, 0).x;

		#ifdef DISTANT_HORIZONS
			float refracted_dh_depth = texelFetch(dhDepthTex, refracted_texel, 0).x;
			bool refracted_is_dh_terrain = is_distant_horizons_terrain(refracted_depth1, dh_depth);
			if (refracted_is_dh_terrain) refracted_depth1 = refracted_dh_depth;
		#else
			#define refracted_is_dh_terrain false
		#endif*/

			// Make sure the refracted pos is behind
			/*float refracted_depth = texelFetch(depthtex1, ivec2(refracted_uv * taau_render_scale * view_res), 0).x;
			refracted_depth = linearize_depth(refracted_depth);
			if (refracted_depth >= linearize_depth(depth0)) {*/
			
				/*vec3 refracted_view_pos  = screen_to_view_space(vec3(refracted_uv, depth0), true, front_is_dh_terrain);
				vec3 refracted_scene_pos = view_to_scene_space(refracted_view_pos);
				world_back_start_pos = refracted_scene_pos + cameraPosition;

				vec3 refracted_view_back_pos = screen_to_view_space(vec3(refracted_uv, refracted_depth1), true, refracted_is_dh_terrain);
				vec3 refracted_scene_back_pos = view_to_scene_space(refracted_view_back_pos);
				world_back_pos = refracted_scene_back_pos + cameraPosition;
				// view_texel * view_pixel_size

				//vec4 refracted_gbuffer_data = texelFetch(colortex1, view_texel, 0);
				skylight = 1.0;
			//}
		}
	}
#endif*/

	is_translucent = is_water || is_translucent;
#ifdef DISTANT_HORIZONS
	is_translucent = is_translucent || dh_depth != dh_depth1;
#endif
	is_translucent = is_translucent && distance(world_back_start_pos, world_back_pos) > 1e-3;

	// Volumetric lighting

#if defined VL
	switch (isEyeInWater) {
		case 0:
			#if defined WORLD_OVERWORLD
			mat2x3 fog = raymarch_air_fog(world_start_pos, world_end_pos, depth0 == 1.0, skylight, dither);
			#elif defined WORLD_NETHER
			//mat2x3 fog = mat2x3(vec3(1.0, 0.4, 0.0), vec3(0.2, 1.0, 1.0));
			/*mat2x3 fog = raymarch_end_fog(world_start_pos, world_end_pos, depth0 == 1.0, dither);
			fog[1] *= 1.0 - clamp01(length(fog[0])); //vec3(length(fog[0]), 0.0, 0.0);*/
			mat2x3 fog = mat2x3(vec3(0.0), vec3(1.0));
			#elif defined WORLD_END
			mat2x3 fog = raymarch_end_fog(world_start_pos, world_end_pos, depth0 == 1.0, dither);
			#else
			mat2x3 fog = mat2x3(vec3(0.0), vec3(1.0));
			#endif
			//fog = mat2x3(vec3(0.0), vec3(1.0));

			fog_scattering    = fog[0];
			fog_transmittance = fog[1];

			if (is_translucent) {
				mat2x3 fog_t = mat2x3(vec3(0.0), vec3(1.0));;
				if (!is_water) {
					#if defined WORLD_OVERWORLD
					fog_t = raymarch_air_fog(world_back_start_pos, world_back_pos, depth1 == 1.0, skylight, dither);
					#elif defined WORLD_NETHER
					fog_t = mat2x3(vec3(0.0), vec3(1.0));
					#elif defined WORLD_END
					fog_t = raymarch_end_fog(world_back_start_pos, world_back_pos, depth1 == 1.0, dither);
					#else
					fog_t = mat2x3(vec3(0.0), vec3(1.0));
					#endif
				} else if (!front_is_dh_terrain) {
					fog_t = raymarch_water_fog(world_back_start_pos, world_back_pos, depth1 == 1.0, false, dither);
				}

				if ((max_of(fog_t[0]) > eps || min_of(fog_t[1]) < (1.0 - eps)) && !is_water) {
					vec4 translucent_color = texelFetch(colortex3, view_texel, 0).rgba;
					fog_t[0] = apply_translucent(fog_t[0], translucent_color);
					fog_t[1] = mix(fog_t[1], vec3(1.0), translucent_color.a);
				}

				fog_scattering    += fog_t[0] * fog_transmittance;
				fog_transmittance *= fog_t[1];
			}

			break;

		case 1:
			mat2x3 water_fog = raymarch_water_fog(world_start_pos, world_end_pos, depth0 == 1.0, true, dither);

			fog_scattering    = water_fog[0];
			fog_transmittance = water_fog[1];

			if (is_translucent) {
				mat2x3 fog_t = mat2x3(vec3(0.0), vec3(1.0));;
				if (is_water) {
					#if defined WORLD_OVERWORLD
					fog_t = raymarch_air_fog(world_back_start_pos, world_back_pos, depth1 == 1.0, skylight, dither);
					#elif defined WORLD_NETHER
					fog_t = mat2x3(vec3(0.0), vec3(1.0));
					#elif defined WORLD_END
					fog_t = raymarch_end_fog(world_back_start_pos, world_back_pos, depth1 == 1.0, dither);
					#else
					fog_t = mat2x3(vec3(0.0), vec3(1.0));
					#endif
				} else {
					fog_t = raymarch_water_fog(world_back_start_pos, world_back_pos, depth1 == 1.0, true, dither);
				}

				if ((max_of(fog_t[0]) > eps || min_of(fog_t[1]) < (1.0 - eps))) {
					vec4 translucent_color = texelFetch(colortex3, view_texel, 0).rgba;
					fog_t[0] = apply_translucent(fog_t[0], translucent_color);
					fog_t[1] = mix(fog_t[1], vec3(1.0), translucent_color.a);
				}

				fog_scattering    += fog_t[0] * fog_transmittance;
				fog_transmittance *= fog_t[1];
			}

			break;

		default:
			fog_scattering    = vec3(0.0);
			fog_transmittance = vec3(1.0);

			break;

		// Prevent potential game crash due to empty switch statement
		case -1:
			break;
	}
#endif
}

#endif
//----------------------------------------------------------------------------//
