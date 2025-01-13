#pragma once

#include "colours.h"

RGBA01 composit_over_premultiplied_01(const RGBA01& background, const RGBA01 &foreground);
RGBA01 composit_over_straight_01(const RGBA01& background, const RGBA01 &foreground);

RGBA255 composit_over_premultiplied_255(const RGBA255& background, const RGBA255 &foreground);
RGBA255 composit_over_straight_255(const RGBA255& background, const RGBA255 &foreground);
