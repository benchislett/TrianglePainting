#pragma once

int clampi(int i, int min, int max) {
	if (i < min) {
		return min;
	}
	else if (i > max) {
		return max;
	}
	else {
		return i;
	}
}

float clamp(float f, float min, float max) {
	if (f < min) {
		return min;
	}
	else if (f > max) {
		return max;
	}
	else {
		return f;
	}
}

float min(float x, float y) {
	return x < y ? x : y;
}

float max(float x, float y) {
	return x > y ? x : y;
}