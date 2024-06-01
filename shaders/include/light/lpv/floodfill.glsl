#ifndef INCLUDE_LIGHT_LPV_FLOODFILL
#define INCLUDE_LIGHT_LPV_FLOODFILL

#include "voxelization.glsl"
#include "/include/light/lpv/light_colors.glsl"

bool is_emitter(uint block_id) {
    bool isInRange = 32u <= block_id && block_id < 64u;

    // Handle lapis and emerald block light
    #ifndef LAPIS_BLOCK_LIGHT
        isInRange = isInRange && block_id != 60u;
    #endif
    #ifndef EMERALD_BLOCK_LIGHT
        isInRange = isInRange && block_id != 59u;
    #endif

    return isInRange;
}

bool is_translucent(uint block_id) {
	return 164u <= block_id && block_id < 180u;
}

// Workaround for emitter ids 61 and >=64 not working in compute - TODO
bool is_custom(uint block_id) {
	return 64u <= block_id && block_id < 96u;
}
bool is_candle(uint block_id) {
	return 264u <= block_id && block_id < 332u;
}

float get_candle_intensity(uint level) {
	//return level > 0 ? (level > 1 ? (level > 2 ? 10.0 : 8.0) : 6.0) : 3.0;
	return sqr(float(level + 1u)) + 1u;
}

vec3 get_emitted_light(uint block_id) {
	if (is_emitter(block_id)) {
		return texelFetch(light_data_sampler, ivec2(int(block_id) - 32, 0), 0).rgb;
	} else if (is_custom(block_id)) {
		return light_color[block_id - 32u];
	} else if (is_candle(block_id)) {
		if(block_id > 327) { // Uncolored Candle
			return light_color[18u] / 8.0 * get_candle_intensity(block_id - 328u);
		}
		block_id -= 264u;
		uint level = uint(floor(float(block_id) / 16.0));
		float intensity = get_candle_intensity(level);

		#ifdef COLORED_CANDLE_LIGHTS
			return tint_color[block_id - level * 16u] * intensity;
		#else
			return light_color[18u] / 8.0 * intensity;
		#endif
	} else {
		return vec3(0.0);
	}
}

vec3 get_tint(uint block_id, bool is_transparent) {
	if (is_translucent(block_id)) {
		return texelFetch(light_data_sampler, ivec2(int(block_id) - 164, 1), 0).rgb;
	} else {
		return vec3(is_transparent);
	}
}

ivec3 clamp_to_voxel_volume(ivec3 pos) {
	return clamp(pos, ivec3(0), voxel_volume_size - 1);
}

vec3 gather_light(sampler3D light_sampler, ivec3 pos) {
	const ivec3[6] face_offsets = ivec3[6](
		ivec3( 1,  0,  0),
		ivec3( 0,  1,  0),
		ivec3( 0,  0,  1),
		ivec3(-1,  0,  0),
		ivec3( 0, -1,  0),
		ivec3( 0,  0, -1)
	);

	return texelFetch(light_sampler, pos, 0).rgb +
	       texelFetch(light_sampler, clamp_to_voxel_volume(pos + face_offsets[0]), 0).xyz +
	       texelFetch(light_sampler, clamp_to_voxel_volume(pos + face_offsets[1]), 0).xyz +
	       texelFetch(light_sampler, clamp_to_voxel_volume(pos + face_offsets[2]), 0).xyz +
	       texelFetch(light_sampler, clamp_to_voxel_volume(pos + face_offsets[3]), 0).xyz +
	       texelFetch(light_sampler, clamp_to_voxel_volume(pos + face_offsets[4]), 0).xyz +
	       texelFetch(light_sampler, clamp_to_voxel_volume(pos + face_offsets[5]), 0).xyz;
}

void update_lpv(writeonly image3D light_img, sampler3D light_sampler) {
	ivec3 pos = ivec3(gl_GlobalInvocationID);
	ivec3 previous_pos = ivec3(vec3(pos) - floor(previousCameraPosition) + floor(cameraPosition));

	uint block_id      = texelFetch(voxel_sampler, pos, 0).x;
	bool transparent   = block_id == 0u || block_id >= 1024u;
	     block_id      = block_id & 1023;
	vec3 light_avg     = gather_light(light_sampler, previous_pos) * rcp(7.0);
	vec3 emitted_light = get_emitted_light(block_id);
	     emitted_light = sqr(emitted_light) * sign(emitted_light);
	vec3 tint          = sqr(get_tint(block_id, transparent));

	vec3 light = emitted_light + light_avg * tint;

	imageStore(light_img, pos, vec4(light, 0.0));
}

#endif // INCLUDE_LIGHT_LPV_FLOODFILL
