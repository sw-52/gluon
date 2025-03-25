#if !defined INCLUDE_LIGHT_DIRECTIONAL_LIGHTMAPS
#define INCLUDE_LIGHT_DIRECTIONAL_LIGHTMAPS

// Based on Ninjamike's implementation in shaderLABS #snippets

vec2 get_directional_lightmaps(vec3 normal, vec3 scene_pos, vec2 light_levels) {
	vec2 lightmap_mul = vec2(1.0);

	vec2 lightmap_gradient; vec3 lightmap_dir;
	mat2x3 pos_gradient = mat2x3(dFdx(scene_pos), dFdy(scene_pos));

	// Blocklight

	lightmap_gradient = vec2(dFdx(light_levels.x), dFdy(light_levels.x));
	lightmap_dir = pos_gradient * lightmap_gradient;

	if (length_squared(lightmap_gradient) > 1e-12) {
		lightmap_mul.x = (clamp01(dot(normalize(lightmap_dir), normal) + 0.8) * DIRECTIONAL_LIGHTMAPS_INTENSITY + (1.0 - DIRECTIONAL_LIGHTMAPS_INTENSITY)) * inversesqrt(sqrt(light_levels.x) + eps);
	}

	// Skylight

	lightmap_gradient = vec2(dFdx(light_levels.y), dFdy(light_levels.y));
	lightmap_dir = pos_gradient * lightmap_gradient;

	if (length_squared(lightmap_gradient) > 1e-12) {
		lightmap_mul.y = (clamp01(dot(normalize(lightmap_dir), normal) + 0.8) * DIRECTIONAL_LIGHTMAPS_INTENSITY + (1.0 - DIRECTIONAL_LIGHTMAPS_INTENSITY)) * inversesqrt(sqrt(light_levels.y) + eps);
	}

	return lightmap_mul;
}

vec3 get_directional_lpv(vec3 normal, vec3 scene_pos, vec3 lpv_blocklight) {
	vec3 blocklight_mul = vec3(1.0);
	//lpv_blocklight = pow(lpv_blocklight, vec3(rcp(8.0)));

	mat3x2 blocklight_gradient; mat3 blocklight_dir;
	mat2x3 pos_gradient = mat2x3(dFdx(scene_pos), dFdy(scene_pos));

	// Blocklight

	blocklight_gradient = transpose(mat2x3(dFdx(lpv_blocklight), dFdy(lpv_blocklight)));
	/*blocklight_gradient = mat3x2(
		vec2(dFdx(lpv_blocklight.x), dFdy(lpv_blocklight.x)),
		vec2(dFdx(lpv_blocklight.y), dFdy(lpv_blocklight.y)),
		vec2(dFdx(lpv_blocklight.z), dFdy(lpv_blocklight.z))
	);*/
	blocklight_dir = pos_gradient * blocklight_gradient;
	//blocklight_dir = mat3(normalize(blocklight_dir[0]), normalize(blocklight_dir[1]), normalize(blocklight_dir[2]));
	//blocklight_dir /= determinant(blocklight_dir);

	if (max_of(vec3(length_squared(blocklight_gradient[0]), length_squared(blocklight_gradient[1]), length_squared(blocklight_gradient[2]))) > 1e-12) {
		blocklight_mul = (clamp01(normalize(normal * blocklight_dir) + 0.8) * DIRECTIONAL_LIGHTMAPS_INTENSITY + (1.0 - DIRECTIONAL_LIGHTMAPS_INTENSITY)) * inversesqrt(sqrt(lpv_blocklight) + eps);
	}
	return blocklight_mul;
}

vec3 get_directional_lpv_full(vec3 normal, vec3 scene_pos, vec3 lpv_blocklight) {
	vec3 blocklight_mul = vec3(1.0);

	mat3x2 blocklight_gradient; mat3 blocklight_dir;
	mat2x3 pos_gradient = mat2x3(dFdx(scene_pos), dFdy(scene_pos));

	// Blocklight

	blocklight_gradient = transpose(mat2x3(dFdx(lpv_blocklight), dFdy(lpv_blocklight)));
	/*blocklight_gradient = mat3x2(
		vec2(dFdx(lpv_blocklight.x), dFdy(lpv_blocklight.x)),
		vec2(dFdx(lpv_blocklight.y), dFdy(lpv_blocklight.y)),
		vec2(dFdx(lpv_blocklight.z), dFdy(lpv_blocklight.z))
	);*/
	blocklight_dir = pos_gradient * blocklight_gradient;
	//blocklight_dir = mat3(normalize(blocklight_dir[0]), normalize(blocklight_dir[1]), normalize(blocklight_dir[2]));
	//blocklight_dir /= determinant(blocklight_dir);

	if (max_of(vec3(length_squared(blocklight_gradient[0]), length_squared(blocklight_gradient[1]), length_squared(blocklight_gradient[2]))) > 1e-12) {
		blocklight_mul = (clamp01(normalize(normal * blocklight_dir) + 0.8) * DIRECTIONAL_LIGHTMAPS_INTENSITY + (1.0 - DIRECTIONAL_LIGHTMAPS_INTENSITY)) * inversesqrt(sqrt(lpv_blocklight) + eps);
	}
	return blocklight_mul;
}

#endif // INCLUDE_LIGHT_DIRECTIONAL_LIGHTMAPS
